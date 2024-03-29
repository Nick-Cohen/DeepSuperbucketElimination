#ifndef Function_HXX_INCLUDED
#define Function_HXX_INCLUDED

#include <climits>
#include <stdlib.h>
#include <string>
#include <vector>

#include "Utils/Mutex.h"
#include "Utils/Sort.hxx"
#include "Problem/Globals.hxx"
//#include "Problem/Workspace.hxx"

#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/script.h>

#ifdef INCLUDE_TORCH
#include "Net.h"
#endif // INCLUDE_TORCH

namespace BucketElimination { class Bucket ; class MiniBucket ; }

#define ARE_Function_Type_BayesianCPT "BayesianCPT"
#define ARE_Function_Probabilities "Probabilities"

#define ARE_Function_Type_RealCost "Cfn"
#define ARE_Function_RealCost "RealCost"

#define ARE_Function_Type_Const "Const"
#define ARE_Function_Const "Const"

#define ARE_Function_TableType double
#define ARE_Function_TableTypeString "double"

/*
	Addressing scheme in the table is as follows. Store sequentially values of argument value combinations 
	in order: 0,...,0; 0,...,1; 0,...,2; ...; 0,...,k-1; 0,...,1,0; 0,...,1,1; etc.
	In other words, as a k-base number starting from 0,...,0 and adding 1 at a time.
	E.g. a value combination (a,b) of a function (X, Y) has address b+K[Y]*a.
	Given a function f(X1, ..., Xm), and a combination of argument values, a formula to compute an index in the table is 
	v[Xm] + v[Xm-1]*K[Xm] + v[Xm-2]*K[Xm]*K[Xm-1] + ...
*/

namespace ARE
{

class ARP ;
class Function ;
class Workspace ;

typedef Function* (*FnConstructor)(void) ;

/*
struct VarValAssignment
{
	int32_t _Var ;
	int32_t _Value ;
	// initialize : VarValAssignment v{varidx, value} ;
} ;

inline int32_t FindVarValueAssignment(int32_t Var, const std::vector<VarValAssignment> & VVA)
{
	// start looking from the end, since VVA is typically context assignment from root down, 
	// therefore assuming that Var is a fn in a bucket, it is more likely towards the end of VVA.
	for (int32_t i = VVA.size()-1 ; i >= 0 ; --i) {
		if (Var == VVA[i]._Var) 
			return VVA[i]._Value ;
		}
	return -1 ;
}
*/

class Function
{
protected :
	Workspace *_Workspace ;
	ARP *_Problem ;
	// index of this function wrt the owner
	// 1) for original functions of the problem, index in the problem array of functions
	// 2) for (M)BE generated functions -(V+1) where V is the var that generated the fn
	int32_t _IDX ;
	// type of the function; user/derived classes should fill this in; e.g. "BayesianCPT"
	std::string _Type ;
	// file with the function contents (table); could be NULL (e.g. Bayesian table loaded from UAI file); or could a binary table for the function.
	std::string _FileName ;
	// a flag indicating that this function is not relevant to the query, and can be skipped.
	// e.g. a CPT that contain no evidence variables, and that has no evidence variable descendants.
	bool _IsQueryIrrelevant ;
	// if case this fn is a bayesian CPT, the child variable.
	// normally child var in a CPT is the last variable, but sometimes we may reorder the fn scope, 
	// and may thus lose track of the child var.
	int32_t _BayesianCPTChildVariable ;
public :
	inline Workspace *WS(void) const { return _Workspace ; }
	inline ARP *Problem(void) const { return _Problem ; }
	inline int32_t IDX(void) const { return _IDX ; }
	inline const std::string & FileName(void) const { return _FileName ; }
	inline const std::string & Type(void) const { return _Type ; }
	inline void SetWS(Workspace *WS) { _Workspace = WS ; }
	inline void SetProblem(ARP *P) { _Problem = P ; }
	inline void SetIDX(int32_t IDX) { _IDX = IDX ; }
	virtual void SetType(const std::string & Type) { _Type = Type ; }
	inline void SetFileName(const std::string & FileName) { _FileName = FileName ; }
	inline bool IsQueryIrrelevant(void) const { return _IsQueryIrrelevant ; }
	inline void MarkAsQueryIrrelevant(void) { _IsQueryIrrelevant = true ; }
	inline void MarkAsQueryRelevant(void) { _IsQueryIrrelevant = false ; }
	inline int32_t BayesianCPTChildVariable(void) const { return _BayesianCPTChildVariable ; }

protected :
	int32_t _nArgs ; // number of variables in the function
	int32_t *_Arguments ; // a list of variables; size of the array is N. in case of Bayesian CPTs, the child variable is last.
	int32_t *_SortedArgumentsList ; // in non-decresasing order; created on-demand.
	int32_t *_ArgumentsPermutationList ; // this is a list of indeces of the arguments of this fn wrt some other ordering of variables. it is allocated when _Arguments is allocated. this array is used when values of the variables come not from a global array of size [0,N), but from a local array.
	int32_t _OneHotArgVectorLength ;
public :
	inline int32_t N(void) const { return _nArgs ; }
	inline int32_t Argument(int32_t IDX) const { return _Arguments[IDX] ; }
	inline const int32_t *Arguments(void) const { return _Arguments ; }
	inline int32_t *ArgumentsPermutationList(void) { return _ArgumentsPermutationList ; }
	inline int32_t SortedArgument(int32_t IDX) const { return _SortedArgumentsList[IDX] ; }
	inline void SetArgumentsPermutationListValue(int32_t idxA, int32_t v) { _ArgumentsPermutationList[idxA] = v ; }
	inline int32_t *SortedArgumentsList(bool CreateIfNULL)
	{
		if (nullptr != _SortedArgumentsList) 
			return _SortedArgumentsList ;
		if (_nArgs <= 0 || ! CreateIfNULL) 
			return nullptr ;
		_SortedArgumentsList = new int32_t[_nArgs] ;
		if (nullptr == _SortedArgumentsList) 
			return nullptr ;
		for (int32_t i = 0 ; i < _nArgs ; i++) _SortedArgumentsList[i] = _Arguments[i] ;
		if (_nArgs > 1) {
			int32_t left[32], right[32] ;
			QuickSortLong2((int32_t*) _SortedArgumentsList, _nArgs, left, right) ;
			}
		return _SortedArgumentsList ;
	}
	int32_t SetArguments(int32_t n, const int32_t *Arguments)
	{
		if (n < 0) 
			return 1 ;
		Destroy() ;
		if (n < 1) 
			return 0 ;
		_Arguments = new int32_t[2*n] ;
		if (NULL == _Arguments) 
			return 1 ;
		for (int32_t i = 0 ; i < n ; i++) 
			_Arguments[i] = Arguments[i] ;
		_nArgs = n ;
		_ArgumentsPermutationList = _Arguments + _nArgs ;
//		for (int32_t i = 0 ; i < _nArgs ; i++) 
//			_ArgumentsPermutationList[i] = i ;
		if (ARE_Function_Type_BayesianCPT == _Type) 
			_BayesianCPTChildVariable = _Arguments[n-1] ;
		if (NULL != _SortedArgumentsList) { delete [] _SortedArgumentsList ; _SortedArgumentsList = NULL ; }
		ComputeTableSize() ;
		return 0 ;
	}
	int32_t SetArguments(int32_t N, const int32_t *Arguments, int32_t ExcludeThisVar) ;
	void ComputeArgumentsPermutationList(int32_t n, const int32_t *vars)
	{
		if (NULL == _Arguments || _nArgs < 0) 
			return ;
		for (int32_t i = 0 ; i < _nArgs ; i++) {
			_ArgumentsPermutationList[i] = -1 ;
			for (int32_t j = 0 ; j < n ; j++) {
				if (_Arguments[i] == vars[j]) 
					{ _ArgumentsPermutationList[i] = j ; break ; }
				}
			}
	}
	inline bool ContainsVariable(const int32_t V) const
	{
		if (_nArgs <= 0) 
			return false ;
		if (_nArgs <= 3) { // check short arrays directly
			for (int32_t i = 0 ; i < _nArgs ; i++) 
				{ if (V == _Arguments[i]) return true ; }
			}
		else if (NULL != _SortedArgumentsList) {
			// can do binary search
			int32_t v, lIDX = 0, rIDX = _nArgs ; // if V is in sorted argslist, then it is withint range [lIDX,rIDX).
			while (lIDX < rIDX) {
				int32_t mIDX = (lIDX + rIDX) >> 1 ; // midpoint between l and r.
				v = _SortedArgumentsList[mIDX] ;
				if (V == v) 
					return true ;
				if (V < v) { // V must be to the left of v; exclude mIDX
					rIDX = mIDX ;
					}
				else { // V must be to the right of v; exclude mIDX
					lIDX = mIDX+1 ;
					}
				}
			}
		else {
			// check the entire args array
			for (int32_t i = 0 ; i < _nArgs ; i++) 
				{ if (V == _Arguments[i]) return true ; }
			}
		return false ;
	}
	inline int32_t ContainsVariables(std::vector<int32_t> & Vars) 
	// compute how many of the given variables _Vars are in this function
	// ASSUME _Vars is sorted
	{
		if (_nArgs <= 0 || 0 == Vars.size()) 
			return 0 ;
		int32_t n = 0 ;
		int32_t *sal = SortedArgumentsList(true) ;
		if (nullptr == sal) 
			return -1 ;
		int32_t i = 0, j = 0 ; // i is for this fn, j is for Vars; meaning of i,j is "next argument to check"
		while (i < _nArgs && j < Vars.size()) {
			if (sal[i] == Vars[j]) {
				++n ; ++i ; ++j ; continue ; }
			if (sal[i] < Vars[j]) 
				++i ;
			else 
				++j ;
			}
		return n ;
	}
	void RemoveVariableFromArguments(int32_t V)
	{
		for (int32_t i = 0 ; i < _nArgs ; i++) {
			if (V == _Arguments[i]) {
				--_nArgs ;
				for (int32_t j = i ; j < _nArgs ; j++) 
					_Arguments[j] = _Arguments[j+1] ;
				break ;
				}
			}
		if (NULL != _SortedArgumentsList) { delete [] _SortedArgumentsList ; _SortedArgumentsList = NULL ; }
	}
	inline int32_t GetHighestOrderedVariable(const int32_t *Var2PosMap) const
	{
		if (_nArgs <= 0) 
			return -1 ;
		int32_t i, v = _Arguments[0] ;
		for (i = 1 ; i < _nArgs ; i++) {
			if (Var2PosMap[_Arguments[i]] > Var2PosMap[v]) 
				v = _Arguments[i] ;
			}
		return v ;
	}

/*
	// This function is used when a bucket elimination computes the outgoing (from the bucket) function, by enumerating over all its argument value combinations. 
	// _Arguments[_nArgs-1] is the variable being eliminated in the bucket, and thus is not among the argument of the outgoing function; 
	// all other variables in _Arguments[] are also arguments of the outgoing function.
	// Also, the order of variables in _Arguments[] and argumentsof(outgoingfunction) are the same (BE should have 
	// reordered arguments of this function to agree with the bucket function arguments). 
	// Given a bucket-function argument values combination, this function computes the adr, wrt the table of this function, 
	// of the corresponding table cell in this table. Obviously, given an adr (argument value combination) of the 
	// bucket function, corresponding adr in the table of this function is different, since these two functions have different scopes.
	// since VarSuperSet does not specify the value of _Arguments[_nArgs-1], this function returns adr wrt this fn corresponding to argument value combination where _Arguments[_nArgs-1]=0.
	inline int64_t ComputeFnTableAdrEx(int NSuperSet, const int *VarSuperSet, const int *ValSuperSet, const int *DomainSizes)
	{
		// _Arguments and VarSuperSet have the same order of variables, except _Arguments[_nArgs-1] is not in VarSuperSet.
		// Therefore _nArgs <= NSuperSet, with the exception of the last var in _Arguments[].
		if (_nArgs < 1) 
			return -1 ;
		if (NSuperSet < 1) 
			return 0 ;
		int i = NSuperSet - 1 ;
		int k = _nArgs - 2 ;
		int64_t adr = 0 ;
		int64_t j = 1 ;
		for (; i >= 0 ; i--) {
			if (VarSuperSet[i] == _Arguments[k]) {
				j *= DomainSizes[_Arguments[k+1]] ;
				adr += j*ValSuperSet[i] ;
				--k ;
				}
			}
		return adr ;
	}*/
/*
	// This function is used when a super bucket elimination computes the outgoing (from the super bucket) function, by enumerating over all its argument value combinations. 
	// The order of variables in _Arguments[] and argumentsof(outgoingfunction) are the same (BE should have reordered arguments of this function to agree with the bucket function arguments). 
	// Given a bucket-function arguments values combination, this function computes the adr, wrt the table of this function, 
	// of the corresponding table cell in this table. Obviously, given an adr (argument value combination) of the 
	// bucket function, corresponding adr in the table of this function is different, since these two functions have different scopes.
	// We assume that VarSuperSet and _Arguments are in the same order.
	inline int64_t ComputeFnTableAdrExN(int NSuperSet, const int *VarSuperSet, const int *ValSuperSet, const int *DomainSizes)
	{
		if (_nArgs < 1) 
			return -1 ;
		if (NSuperSet < 1) 
			return 0 ;
		int i = NSuperSet - 1 ;
		int64_t adr = -1 ;
		int k = _nArgs - 1 ;
		for (; i >= 0 ; i--) {
			if (VarSuperSet[i] == _Arguments[k]) {
				adr = ValSuperSet[i] ;
				--k ;
				--i ;
				break ;
				}
			}
		int64_t j = 1 ;
		for (; i >= 0 ; i--) {
			if (VarSuperSet[i] == _Arguments[k]) {
				j *= DomainSizes[_Arguments[k+1]] ;
				adr += j*ValSuperSet[i] ;
				--k ;
				}
			}
		return adr ;
	}*/
	// This function given adr in this fn table, given context assignment.
	// FullContextAssignment is the assignment (to ancestor variable of the bucket) to variables; variables of the fn need to be instantiated.
	inline int64_t ComputeFnTableAdr(const int32_t *FullContextAssignment, const int32_t *DomainSizes) const
	{
		if (_nArgs < 1) 
			return -1 ;
		int32_t i = _nArgs - 1 ;
		int64_t adr = FullContextAssignment[_Arguments[i]] ; // value of the last variable
		int64_t j = 1 ;
		for (--i ; i >= 0 ; i--) {
			j *= DomainSizes[_Arguments[i+1]] ;
			adr += j*FullContextAssignment[_Arguments[i]] ; // value of variable i
			}
		return adr ;
	}
/*	// compute adr in the table when ContextAssignment is given as a list of var/value assignments.
	inline int64_t ComputeFnTableAdr(const std::vector<VarValAssignment> & FullContextAssignment, const int32_t *DomainSizes) const
	{
		if (_nArgs < 1) 
			return -1 ;
		int32_t i = _nArgs - 1 ;
		int64_t adr = FindVarValueAssignment(_Arguments[i], FullContextAssignment) ; // FullContextAssignment[_Arguments[i]] ; // value of the last variable
		int64_t j = 1 ;
		for (--i ; i >= 0 ; i--) {
			j *= DomainSizes[_Arguments[i+1]] ;
			adr += j*FindVarValueAssignment(_Arguments[i], FullContextAssignment) ; // FullContextAssignment[_Arguments[i]] ; // value of variable i
			}
		return adr ;
	}*/
	// this function computes fn table adr, wrt some local context, passed in as 'LocalContextAssignment'.
	// e.g. LocalContextAssignment could be the path assignment from root bucket to the current bucket (where this fn belongs).
	// the particular argument index is its distance to the root; i.e. argument[0] value is in LocalContextAssignment[0], etc.
	// we assume size of ValSuperSet[] is at least _nArgs
	inline int64_t ComputeFnTableAdr_wrtLocalPermutation(const int32_t *LocalContextAssignment, const int32_t *DomainSizes) const
	{
		if (_nArgs < 1) 
			return -1 ;
		int32_t i = _nArgs - 1 ;
		int64_t adr = LocalContextAssignment[_ArgumentsPermutationList[i]] ;
		int64_t j = 1 ;
		for (--i ; i >= 0 ; i--) {
			j *= DomainSizes[_Arguments[i+1]] ;
			adr += j*LocalContextAssignment[_ArgumentsPermutationList[i]] ;
			}
		return adr ;
	}

	inline int64_t ComputeFnTableAdr_wrtSignatureAssignment(const int32_t *SignatureAssignment, const int32_t *DomainSizes) const
	{
		if (_nArgs < 1) 
			return -1 ;
		int32_t i = _nArgs - 1 ;
		int64_t adr = SignatureAssignment[i] ; // value of the last variable
		int64_t j = 1 ;
		for (--i ; i >= 0 ; i--) {
			j *= DomainSizes[_Arguments[i+1]] ;
			adr += j*SignatureAssignment[i] ; // value of variable i
			}
		return adr ;
	}

	inline int32_t FillInOneHotNNinput_wrtNativeAssignemnt(at::Tensor& input, const int32_t* NativeAssignment, const int32_t* DomainSizes)
	{
		if (_nArgs <= 0)
			return 0;
		int l = input.numel();
		if (_OneHotArgVectorLength <= 0 || l != _OneHotArgVectorLength) {
			//			input = torch::zeros({ 1, _nArgs }) ;
			//			l = input.numel();
			//			if (l != _nArgs)
			return 1;
		}
		else {
			void* ptr = input.data_ptr();
			float* e = (float*)ptr;
			for (int32_t i = 0; i < l; ++i)
				e[i] = (float)0.0;
		}
		int32_t j = 0; // j points to the beginning on the variable's one-hot encoding bits...
		// print out input to see type...
		//std::cout << "input = " << input << "\n";
		void* ptr = input.data_ptr();
		float* e = (float*)ptr;
		for (int32_t i = 0; i < _nArgs; ++i) {
			int32_t var = _Arguments[i];
			int32_t k = DomainSizes[var];
			int32_t val = NativeAssignment[var]; //  NativeAssignment[var];
			if (val < 0 || val >= k) {
				int bug = 1;
			}
			if (val > 0) {
				int32_t adr = j + (val - 1);
				if (adr < 0 || adr >= _OneHotArgVectorLength) {
					printf("\n1Hot adr out of bounds adr=%d _OneHotArgVectorLength=%d wrtNativeAssignemnt", adr, _OneHotArgVectorLength);
					printf("\n");
					exit(153);
				}
				e[adr] = (float)1.0;
			}
			j += k - 1;
		}
		return 0;
	}

	inline int32_t FillInOneHotNNinput_wrtPermutationList(at::Tensor & input, const int32_t *BEPathAssignment, const int32_t *DomainSizes)
	{
		if (_nArgs <= 0)
			return 0;
		int l = input.numel() ;
		if (_OneHotArgVectorLength <= 0 || l != _OneHotArgVectorLength) {
//			input = torch::zeros({ 1, _nArgs }) ;
//			l = input.numel();
//			if (l != _nArgs)
				return 1 ;
			}
		else {
			void* ptr = input.data_ptr();
			float* e = (float*)ptr;
			for (int32_t i = 0 ; i < l ; ++i)
				e[i] = (float)0.0;
			}
		int32_t j = 0 ; // j points to the beginning on the variable's one-hot encoding bits...
// print out input to see type...
//std::cout << "input = " << input << "\n";
		void *ptr = input.data_ptr() ;
		float *e = (float *) ptr ;
		for (int32_t i = 0 ; i < _nArgs ; ++i) {
			int32_t var = _Arguments[i] ;
			int32_t k = DomainSizes[var] ;
			int32_t val = BEPathAssignment[_ArgumentsPermutationList[i]] ; //  NativeAssignment[var];
			if (val < 0 || val >= k) {
				int bug = 1;
				}
			if (val > 0) {
				int32_t adr = j + (val - 1) ;
				if (adr < 0 || adr >= _OneHotArgVectorLength) {
					printf("\n1Hot adr out of bounds adr=%d _OneHotArgVectorLength=%d wrtPermutationList", adr, _OneHotArgVectorLength);
					printf("\n");
					exit(153);
					}
				e[adr] = (float)1.0;
				}
			j += k-1 ;
			}
		return 0;
	}

	// assume the value of variable i is at NativeAssignment[i]
	inline int64_t ComputeFnTableAdr_Native(const int32_t *NativeAssignment, const int32_t *DomainSizes) const
	{
		if (_nArgs < 1) 
			return -1 ;
		int32_t i = _nArgs - 1 ;
		int64_t adr = NativeAssignment[_Arguments[i]] ;
		int64_t j = 1 ;
		for (--i ; i >= 0 ; i--) {
			j *= DomainSizes[_Arguments[i+1]] ;
			adr += j*NativeAssignment[_Arguments[i]] ;
			}
		return adr ;
	}

	// **************************************************************************************************
	// when running Mini-Bucket Elimination, functions belong to a bucket.
	// specifically, this is the bucket where the function is assigned to (Original/Augmented).
	// note that this does not mean the fn belongs to that bucket; it belongs to originating bucket.
	// **************************************************************************************************

protected :
	BucketElimination::Bucket *_Bucket ;
public :
	inline BucketElimination::Bucket *Bucket(void) const { return _Bucket ; }
	inline void SetBucket(BucketElimination::Bucket *B) { _Bucket = B ; }

	// _OriginatingBucket is used when this function is generated during Bucket Elimination
protected :
	BucketElimination::MiniBucket *_OriginatingMiniBucket ;
	BucketElimination::Bucket *_OriginatingBucket ;
public :
	inline BucketElimination::MiniBucket *OriginatingMiniBucket(void) const { return _OriginatingMiniBucket ; }
	inline void SetOriginatingMiniBucket(BucketElimination::MiniBucket *MB) { _OriginatingMiniBucket = MB ; }
	inline BucketElimination::Bucket *OriginatingBucket(void) const { return _OriginatingBucket ; }
	inline void SetOriginatingBucket(BucketElimination::Bucket *B) { _OriginatingBucket = B ; }

	// *************************************************************************************************************************
	// function current value is the value of the function corresponding to current context (argument) values.
	// typically we compute/set this value during search/sampling/etc when traversing bucket(pseudo) tree, 
	// to improve performance. e.g. we compute current value of augmented functions (when in the bucket of the var/functions), 
	// but when processing them in another bucket as intermediate functions, we don't need to recompute them.
	// *************************************************************************************************************************

protected :

	ARE_Function_TableType _CurrentValue ;

public :

	inline ARE_Function_TableType & CurrentValue(void) { return _CurrentValue ; }

protected :

	// size of the function table, as a number of elements; this is a function of domain size of arguments.
	int64_t _TableSize ;

	// when fn is represented as a table, its contents are here.
	ARE_Function_TableType *_TableData ;

public :

	inline int64_t TableSize(void) const { return _TableSize ; }
	inline ARE_Function_TableType TableEntry(int64_t IDX) const { return _TableData[IDX] ; }
	inline ARE_Function_TableType TableEntry(int32_t *BEPathAssignment, const int32_t *K) const 
	{
		if (_nArgs <= 0) 
			return _ConstValue ;
		__int64 IDX = ComputeFnTableAdr_wrtLocalPermutation(BEPathAssignment, K) ;
		return _TableData[IDX] ;
	}
	virtual ARE_Function_TableType TableEntryExNativeAssignment(int32_t *NativeAssignment, const int32_t *K) 
	{
		if (_nArgs <= 0) 
			return _ConstValue ;
		int64_t adr = ComputeFnTableAdr_Native(NativeAssignment, K) ;
		return _TableData[adr] ;
	}
	virtual ARE_Function_TableType TableEntryEx(int32_t *BEPathAssignment, const int32_t *K)  
	/*
		desc = return fn value corresponding to given input configuration...
		BEPathAssignment = assignment to all variables on the path from the bucket to the root of the bucket tree...
	    this fn is a wrapper for these two lines... when functions are table-based...
		adr = fn->ComputeFnTableAdr_wrtLocalPermutation(BEPathAssignment, K) ;
		double v = fn->TableEntry(adr) ;
		which is 
		return fn->TableEntry(fn->ComputeFnTableAdr_wrtLocalPermutation(BEPathAssignment, K)) ;
		OVERWRITE in a NN-based fn...
	*/
	{
		return _TableData[ComputeFnTableAdr_wrtLocalPermutation(BEPathAssignment, K)] ;
	}
	inline ARE_Function_TableType *TableData(void) { return _TableData ; }
	inline bool HasTableData(void) const { return NULL != _TableData ; }
	virtual void DestroyTableData(void)
	{
		if (NULL != _TableData) {
			delete [] _TableData ;
			_TableData = NULL ;
			}
	}
	virtual int32_t AllocateTableData(void)
	{
		if (_TableSize <= 0) 
			DestroyTableData() ;
		else if (NULL == _TableData) {
			try {
				_TableData = new ARE_Function_TableType[_TableSize] ;
				}
			catch (...) {
				}
			if (NULL == _TableData) 
				return 1 ;
			}
		return 0 ;
	}
	inline int32_t SetTableData(int64_t Size, ARE_Function_TableType *TableData)
	{
		//if (_TableSize < 0) - Changed Filjor
		_TableSize = ComputeTableSize() ;
		if (_TableSize != Size) 
			return 1 ;
		if (_TableSize <= 0) 
			return 0 ;
		DestroyTableData() ;
		_TableData = TableData ;
		return 0 ;
	}
	inline int32_t CheckData(void)
	{
		if (NULL != _TableData) {
			for (int64_t i = _TableSize-1 ; i >= 0 ; i--) {
//				if (0 != _isnan(_Data[i]))
				if (_TableData[i] != _TableData[i]) // tests for NaN
					// don't check if 0 == _finite(_Data[i])), since we may have -INF when the table is converted to log
					return 1 ;
				}
			}
		return 0 ;
	}
	inline int64_t CountNumberOf0s(void)
	{
		if (_TableSize <= 0 || NULL == _TableData) 
			return 0 ;
		int64_t n = 0 ;
		for (int64_t i = _TableSize-1 ; i >= 0 ; i--) 
			{ if (0.0 == _TableData[i]) n++ ; }
		return n ;
	}
	inline int32_t SumEntireData(ARE_Function_TableType & sum)
	{
		if (NULL == _TableData) 
			return 1 ;
		if (_TableSize <= 0) 
			return 2 ;
		sum = 0.0 ;
		for (int64_t i = _TableSize-1 ; i >= 0 ; i--) {
/*
#ifdef _DEBUG
			if (fabs(_Data[i]) > 1.0e+10) 
				{ int32_t bug = 1 ; }
#endif // _DEBUG
*/
			sum += _TableData[i] ;
			}
		return 0 ;
	}
	inline double Compute0Tightness(void)
	{
		if (_TableSize <= 0) 
			return -1.0 ;
		int64_t n0s = CountNumberOf0s() ;
		return ((double)n0s)/((double)_TableSize) ;
	}

	int32_t ConvertTableToLogScale(void) ;

//	// compute _ArgumentDomainFactorization
//	int32_t FactorizeArgumentDomains(void) ;
	// compute table size, based on domain sizes of arguments; this is the number of elements in the table.
	int64_t ComputeTableSize(void) ;
	// compute table size in log10 scale, based on domain sizes of arguments; this is the number of elements in the table.
	// this fn should be called when the table size can be huge.
	double GetTableSize_Log10(void) ;
	// compute table size, based on domain sizes of arguments; this is in bytes.
	int64_t ComputeTableSpace(void) ;
	// compute table size, based on domain sizes of arguments; this is in bytes.
	// this fn should be called when the table size can be huge.
	double GetTableSpace_Log10(void) ;

	// table will be stored in memory as a 1 (single) block; allocate table.
	// this fn assumes that the table is not allocated yet; i.e. no blocks of the table have ever been allocated.
	int32_t AllocateInMemoryAsSingleTableBlock(void) ;

	// *************************************************************************************************************************
	// function has const value iff it has no arguments
	// *************************************************************************************************************************

protected :

	ARE_Function_TableType _ConstValue ;

public :

	inline ARE_Function_TableType & ConstValue(void) { return _ConstValue ; }

public :

	// this function allocates and fills in random Bayesian table.
	// it assumes that the last argument is the child variable in the CPT.
	int32_t FillInRandomBayesianTable(void) ;

	// this function creates signature (i.e. variables/arguments) of this function, assuming it is a Bayesian CPT.
	// it will place the child as the last variable.
	// other variables are picked randomly, assuming their indeces are less than ChildIDX.
	int32_t GenerateRandomBayesianSignature(int32_t N, int32_t ChildIDX) ;
//	int32_t GenerateRandomBayesianSignature(int32_t N, int32_t ChildIDX, int32_t & nFVL, int32_t *FreeVariableList) ;

	// this will reorder arguments of this function, so that its arguments agree with the order of variables in AF(ront)/AB(ack).
	// we assume that the arguments of this function are in AF+AB; however, AF+AB may have variables not in this function, e.g. AF+AB is a superset of this-function->Arguments.
	// ReOrderArguments() is called by a parent bucket on the bucket-function from a child bucket, 
	// with parent bucket's keep/eliminate variables as F/B lists.
	// this function arguments will be split along F/B.
	// moreover, order of arguments of this function split along F/B will agree with F/B.
	int32_t ReOrderArguments(int32_t nAF, const int32_t *AF, int32_t nAB, const int32_t *AB) ;

	// this function will remove the given evidence variable from this function
	int32_t RemoveVariable(int32_t Variable, int32_t ValueRemaining) ;

	// this function will remove the given value of the given variable; 
	// this function is used when domains of variables are pruned.
	int32_t RemoveVariableValue(int32_t Variable, int32_t ValueRemoved) ;

	// serialize this function as XML; append it to the given string.
	virtual int32_t SaveXMLString(const char *prefixspaces, const char *tag, const std::string & Dir, std::string & S) ;

	// get the filename of the table of this function, when saved as binary
	int32_t GetTableBinaryFilename(const std::string & Dir, std::string & fn) ;

	// return non-0 iff something is wrong with the function
	int32_t CheckIntegrity(void) ;

public :

	// 2023-11-14 : we assume we have a table ... save into given file ...
	virtual int32_t SaveToFile(std::string & FileName) ;

	virtual int32_t GenerateSamplesXmlFilename(
		const char* sSuffix, std::string& fnSamples, std::string& fnNetwork, std::string& fnFNsignalling, std::string& sPrefix, std::string& sPostFix,
		int32_t nSamples, double samples_min_value, double samples_max_value, double samples_sum) ;

	virtual void Initialize(Workspace *WS, ARP *Problem, int32_t IDX)
	{
		// note : _OriginatingBucket and _OriginatingMiniBucket should be set and can be used here...
		Destroy() ;
		_Workspace = WS ;
		_Problem = Problem ;
		_IDX = IDX ;
		_CurrentValue = _ConstValue = DBL_MAX ;
	}

	virtual void Destroy(void)
	{
		DestroyTableData() ;
		if (NULL != _Arguments) {
			delete [] _Arguments ;
			_Arguments = _ArgumentsPermutationList = NULL ;
			}
		if (NULL != _SortedArgumentsList) {
			delete [] _SortedArgumentsList ;
			_SortedArgumentsList = NULL ;
			}
		_nArgs = 0 ;
		_OneHotArgVectorLength = -1 ;
		_BayesianCPTChildVariable = -1 ;
		_IsQueryIrrelevant = false ;
		_CurrentValue = _ConstValue = DBL_MAX ;
	}
	Function(void)
		:
		_Workspace(NULL), 
		_Problem(NULL), 
		_IDX(-INT_MAX), 
		_IsQueryIrrelevant(false), 
		_BayesianCPTChildVariable(-1), 
		_nArgs(0), 
		_Arguments(NULL), 
		_ArgumentsPermutationList(NULL), 
		_OneHotArgVectorLength(-1),
		_SortedArgumentsList(NULL), 
		_Bucket(NULL), 
		_OriginatingBucket(NULL), 
		_OriginatingMiniBucket(NULL), 
		_CurrentValue(DBL_MAX), 
		_TableSize(-1), 
		_TableData(NULL), 
//		_nArgumentDomainFactorization(0),
		_ConstValue(DBL_MAX)
	{
	}
	Function(Workspace *WS, ARP *Problem, int32_t IDX)
		:
		_Workspace(WS), 
		_Problem(Problem), 
		_IDX(IDX), 
		_IsQueryIrrelevant(false), 
		_BayesianCPTChildVariable(-1), 
		_nArgs(0), 
		_Arguments(NULL), 
		_ArgumentsPermutationList(NULL), 
		_OneHotArgVectorLength(-1), 
		_SortedArgumentsList(NULL), 
		_Bucket(NULL), 
		_OriginatingBucket(NULL), 
		_OriginatingMiniBucket(NULL), 
		_CurrentValue(DBL_MAX), 
		_TableSize(-1), 
		_TableData(NULL), 
//		_nArgumentDomainFactorization(0),
		_ConstValue(DBL_MAX)
	{
	}
	virtual ~Function(void)
	{
		Destroy() ;
	}
} ;

inline Function *FunctionConstructor(void)
{
	return new Function ;
}

// Variables[0 ... N), CurrentValueCombination[0 ... N), DomainSizes[0 ... problem->N)
inline int64_t ComputeFnTableAdr(int32_t nVariables, const int32_t *Variables, const int32_t *CurrentValueCombination, const int32_t *DomainSizes)
{
	if (nVariables < 1) 
		return -1 ;
	int32_t i = nVariables - 1 ;
	int64_t adr = CurrentValueCombination[i] ;
	int64_t j = 1 ;
	for (--i ; i >= 0 ; i--) {
		j *= DomainSizes[Variables[i+1]] ;
		adr += j*CurrentValueCombination[i] ;
		}
	return adr ;
}

// Variables[0 ... N), CurrentValueCombination[0 ... N), DomainSizes[0 ... problem->N)
inline void ComputeArgCombinationFromFnTableAdr(int64_t ADR, int32_t nVariables, const int32_t *Variables, int32_t *CurrentValueCombination, const int32_t *DomainSizes)
{
	int64_t adr ;
	for (int32_t i = nVariables-1 ; i >= 0 ; i--) {
		adr = ADR/DomainSizes[Variables[i]] ;
		CurrentValueCombination[i] = ADR - adr*DomainSizes[Variables[i]] ;
		ADR = adr ;
		}
}

// Arguments[0 ... N), CurrentValueCombination[0 ... N), DomainSizes[0 ... problem->N)
inline void EnumerateNextArgumentsValueCombination(int32_t nArguments, const int32_t *Arguments, int32_t *CurrentValueCombination, const int32_t *DomainSizes)
{
	for (int32_t i = nArguments - 1 ; i >= 0 ; i--) {
		if (++CurrentValueCombination[i] < DomainSizes[Arguments[i]]) return ;
		CurrentValueCombination[i] = 0 ;
		}
}
inline void EnumerateNextArgumentsValueCombinationEx(int32_t nArguments, const int32_t *Arguments, int32_t *CurrentValueCombination, const int32_t *DomainSizes)
{
	for (int32_t i = nArguments - 1 ; i >= 0 ; i--) {
		int32_t v = Arguments[i] ;
		if (++CurrentValueCombination[v] < DomainSizes[v]) return ;
		CurrentValueCombination[v] = 0 ;
		}
}

inline int32_t ComputeSignature(int32_t nFNs, ARE::Function **FNs, int32_t & nSignature, int32_t * & Signature)
{
	nSignature = -1 ;
	if (NULL != Signature) { delete [] Signature ; Signature = NULL ; }

	int32_t i, j, k, n = 0 ;
	for (i = 0 ; i < nFNs ; i++) 
		n += FNs[i]->N() ;
	// n is an upper bound on the width
	if (n <= 0) 
		{ nSignature = 0 ; return 0 ; }

	// compute width/signature
	Signature = new int32_t[n] ;
	if (NULL == Signature) 
		return 1 ;
	nSignature = 0 ;
	for (i = 0 ; i < nFNs ; i++) {
		ARE::Function *f = FNs[i] ;
		for (j = 0 ; j < f->N() ; j++) {
			int32_t v = f->Argument(j) ;
			for (k = 0 ; k < nSignature ; k++) 
				{ if (Signature[k] == v) break ; }
			if (k >= nSignature) 
				Signature[nSignature++] = v ;
			}
		}
	return 0 ;
}

// return true iff Obj1 is greater than Obj2 (i.e. Obj1 is after Obj2).
// we assume both functions have sorted argument lists!
inline int32_t FunctionComparisonByScope_Greater(void /* ARE::Function */ *Obj1, void /* ARE::Function */ *Obj2)
{
	ARE::Function *fn1 = (ARE::Function *) Obj1 ;
	ARE::Function *fn2 = (ARE::Function *) Obj2 ;
	if (fn1->N() > fn2->N()) 
		return 1 ;
	if (fn1->N() < fn2->N()) 
		return -1 ;
	// Obj1/2 have same scope size; compare pairs
	for (int32_t i = 0 ; i < fn1->N() ; i++) {
		if (fn1->SortedArgument(i) > fn2->SortedArgument(i)) 
			return 1 ;
		else if (fn1->SortedArgument(i) < fn2->SortedArgument(i)) 
			return -1 ;
		}
	return 0 ;
}

// return true iff Obj1/2 have equal scope.
// we assume both functions have sorted argument lists!
inline bool FunctionComparisonByScope_Equal(void /* ARE::Function */ *Obj1, void /* ARE::Function */ *Obj2)
{
	ARE::Function *fn1 = (ARE::Function *) Obj1 ;
	ARE::Function *fn2 = (ARE::Function *) Obj2 ;
	if (fn1->N() != fn2->N()) 
		return false ;
	// Obj1/2 have same scope size; compare pairs
	for (int32_t i = 0 ; i < fn1->N() ; i++) {
		if (fn1->Argument(i) != fn2->Argument(i)) 
			return false ;
		}
	return true ;
}

} // namespace ARE

#endif // Function_HXX_INCLUDED
