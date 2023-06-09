#ifndef BUCKET_HXX_INCLUDED
#define BUCKET_HXX_INCLUDED

#include <inttypes.h>

#include "Utils/MiscUtils.hxx"
#include "Function.hxx"
#include "MiniBucket.hxx"

typedef void (*BucketPostComputeExternalNote)(BucketElimination::Bucket & B, BucketElimination::MiniBucket & MB) ;

namespace BucketElimination
{

class MBEworkspace ;
class MiniBucket ;

class Bucket
{
protected :
	MBEworkspace *_Workspace ;
	int32_t _IDX ; // index of this bucket in the workspace
	int32_t _V ; // in case the bucket is created for a specic variable in bucket elimination, this is the var. -1 otherwise (e.g. in case of superbuckets).
	BucketPostComputeExternalNote _PostComputeExtNote ;
public :
	inline MBEworkspace *Workspace(void) const { return _Workspace ; }
	inline int32_t IDX(void) const { return _IDX ; }
	inline int32_t V(void) const { return _V ; }
	inline void SetIDX(int32_t IDX) { _IDX = IDX ; }
	inline BucketPostComputeExternalNote & PostComputeExtNote(void) { return _PostComputeExtNote ; }
	ARE::ARP *Problem(void) const ;

	// **********************************************************************************
	// width/signature of this bucket; this includes Original/Augmented functions, but not Intermediate functions.
	// when processing a bucket (bottom-up over the bucket tree) we combine original/augmented functions, ignoring intermediate functions.
	// note that when computing the heuristic, we combine augmented/intermediate functions, ignoring original functions.
	// **********************************************************************************
protected :
	int32_t _Width ; // cardinality of signature of this bucket; this includes variables eliminated in this bucket.
	int32_t *_Signature ; // a union of the scopes of all functions (original/augmented) in this bucket, including variables eliminated in this bucket.
public :
	inline int32_t Width(void) const { return _Width ; }
	inline const int32_t *Signature(void) const { return _Signature ; }
	inline void InvalidateSignature(void) { _Width = -1 ; if (NULL != _Signature) { delete [] _Signature ; _Signature = NULL ; }}
	int32_t ComputeSignature(unsigned char instructions = 0x03) ; // 1=original FNs, 2=Augmented FNs, 4=Intermediate FNs.
	int32_t AddVarsToSignature(int32_t & n, int32_t *AdditionalSignature) ;

	// **********************************************************************************
	// variable(s) of the bucket that are eliminated when a bucket output fn is computed from functions of this bucket.
	// normally this is 1 variable. in case of superbuckets, this is more.
	// we assume _Vars is sorted in increasing order...
	// **********************************************************************************
protected :
	std::vector<int32_t> _Vars ;
public :
	inline int32_t nVars(void) const { return _Vars.size() ; }
	inline int32_t Var(int32_t IDX) const { return _Vars[IDX] ; }
	inline std::vector<int32_t> & Vars(void) { return _Vars ; }
	inline int32_t *VarsArray(void) { return _Vars.data() ; }
	inline int32_t AddVar(int32_t Var)
	{
		int32_t i, n = _Vars.size() ;
		for (i = 0 ; i < n ; ++i) {
			if (_Vars[i] == Var) 
				return 0 ;
			}
		if (_Vars.capacity() == _Vars.size()) 
			_Vars.reserve(_Vars.capacity()+3) ; // reallocate by reserving a few more elements of space...
		_Vars.resize(n+1) ;
		if (_Vars.size() <= n) 
			return 1 ;
		_Vars[n] = Var ;
		return 0 ;
	}
	inline int32_t SetVars(int32_t n, int32_t *Vars)
	{
		_Vars.clear() ;
		int32_t i, j, n_ = n+1 ; // just in case 1 more, because we might need to add _V if not in Vars...
		if (n_ > _Vars.capacity()) {
			_Vars.reserve(n_) ;
			if (n_ != _Vars.capacity()) 
				return 1 ;
			}
		if (_V >= 0) 
			_Vars.push_back(_V) ;
		for (i = 0 ; i < n ; ++i) {
			int32_t v = Vars[i] ;
			for (j = 0 ; j < _Vars.size() ; ++j) {
				if (v == _Vars[j]) break ; }
			if (j >= _Vars.size()) 
				_Vars.push_back(v) ;
			}
		return 0 ;
	}
	void SortVars(void)
	{
		if (_Vars.size() <= 1) 
			return ;
		int32_t left[32], right[32] ;
		QuickSortLong2(_Vars.data(), _Vars.size(), left, right) ;
	}

	// ****************************************************************************************
	// Bucket-tree structure.
	// ****************************************************************************************
protected :

	// parent bucket is determined by _OutputFunction.
	Bucket *_ParentBucket ;
	// root of the bucket tree this bucket belongs to; note that a BE workspace may have a bucket-forest.
	Bucket *_RootBucket ;
	// distance to the root; defined as number of edges to travel to get to the root.
	int32_t _DistanceToRoot ;
	// distance to the farthest leaf node from this bucket. as number of edges to travel.
	// this is useful, e.g., when computing BEEM computation order.
	int32_t _Height ;
	// number of buckets in the subtree rooted at this bucket; including this bucket.
	int32_t _nSubtreeBuckets ;
	// maximum number of variables in any descendant bucket; not including this bucket.
	int32_t _MaxDescendantNumVars ;

	// maximum height of the bucket tree when counting branching variables; as number of branching variables on the path. include this bucket.
	int32_t _MaxTreeHeight_BranchingVars ;

	// size of the computation of all mb output functions
	int64_t _ComputationNewFunctionSize ;

	// max ComputationNewFunctionSize() for all descendants of this bucket; it does not include this bucket.
	int64_t _MaxDescendantComputationNewFunctionSize ;

	// child vars (buckets) of this bucket; child buckets are at _Buckets[_Var2BucketMapping[vChild]]
	std::vector<int32_t> _ChildVars ;

public :

	inline Bucket *ParentBucket(void) const { return _ParentBucket ; }
	inline Bucket *RootBucket(void) const { return _RootBucket ; }
	inline void SetParentBucket(Bucket *B) { _ParentBucket = B ; }
	inline void SetRootBucket(Bucket *B) { _RootBucket = B ; }
	inline void SetDistanceToRoot(int32_t D) { _DistanceToRoot = D ; }
	inline void SetHeight(int32_t H) { _Height = H ; }
	inline void SetnSubtreeBuckets(int32_t n) { _nSubtreeBuckets = n ; }
	inline void SetMaxDescendantNumVars(int32_t v) { _MaxDescendantNumVars = v ; }
	inline void SetMaxTreeHeight_BranchingVars(int32_t v) { _MaxTreeHeight_BranchingVars = v ; }
	inline void SetComputationNewFunctionSize(int64_t v) { _ComputationNewFunctionSize = v ; }
	inline void SetMaxDescendantComputationNewFunctionSize(int64_t v) { _MaxDescendantComputationNewFunctionSize = v ; }
	inline int32_t DistanceToRoot(void) const { return _DistanceToRoot ; }
	inline int32_t Height(void) const { return _Height ; }
	inline int32_t nSubtreeBuckets(void) const { return _nSubtreeBuckets ; }
	inline int32_t MaxDescendantNumVars(void) const { return _MaxDescendantNumVars ; }
	inline int32_t MaxDescendantNumVarsEx(void) const { return _MaxDescendantNumVars > _Width ? _MaxDescendantNumVars : _Width ; }
	inline int32_t MaxTreeHeight_BranchingVars(void) const { return _MaxTreeHeight_BranchingVars ; }
	inline int64_t ComputationNewFunctionSize(void) const { return _ComputationNewFunctionSize ; }
	inline int64_t MaxDescendantComputationNewFunctionSize(void) const { return _MaxDescendantComputationNewFunctionSize ; }
	inline int64_t MaxDescendantComputationNewFunctionSizeEx(void) const { return _ComputationNewFunctionSize > _MaxDescendantComputationNewFunctionSize ? _ComputationNewFunctionSize : _MaxDescendantComputationNewFunctionSize ; }

	inline int32_t nChildren(void) const { return _ChildVars.size() ; }
	inline std::vector<int32_t> & ChildVars(void) { return _ChildVars ; }
	inline int32_t ChildVar(int32_t idx) const { return _ChildVars[idx] ; }
	BucketElimination::Bucket *ChildBucket(int32_t idx) const ;
	inline int32_t SetChildren(int32_t n, int32_t *C)
	{
		_ChildVars.clear() ;
		if (n <= 0) 
			return 0 ;
		if (n > _ChildVars.capacity()) {
			_ChildVars.reserve(n) ;
			if (n != _ChildVars.capacity()) 
				return 1 ;
			}
		for (int32_t i = 0 ; i < n ; ++i) 
			_ChildVars.push_back(C[i]) ;
		return 0 ;
	}
	inline void RemoveChild(int32_t v)
	{
		int32_t j = 0, n = 0 ;
		for (int32_t i = 0 ; i < _ChildVars.size() ; ++i) {
			if (v != _ChildVars[i]) 
				_ChildVars[j++] = _ChildVars[i] ;
			else 
				++n ;
			}
		_ChildVars.resize(_ChildVars.size() - n) ;
	}

	int32_t GetChildren(std::vector<BucketElimination::Bucket*> & Children) ; // we will not clear the contents of Children at start.
	int32_t GetDescendants(int32_t nLevelsDown, std::vector<BucketElimination::Bucket*> & Descendants) ; // n levels in Breach First order.

	// get DFS traversal order of subtree rooted at this bucket; includes this bucket. this bucket is last in the list.
	int32_t GetDFSorderofsubtree(
		// OUT
		std::vector<BucketElimination::Bucket*> & dfs_reverse_chain, 
		// IN : helper arrays
		std::vector<BucketElimination::Bucket*> & dfs_stack_b, std::vector<int32_t> & dfs_stack_i) ;

	// **********************************************************************************
	// sometimes, we want to compute the MBE approximation error, min, avg, max...
	// this is a measure how well the (combination of) MBE-generated output fn(s) matches the correct (true) BE output fn...
	// this error can be 1) local (just this bucket), 2) cumulative (containing all errors from the bottom of the bucket tree on upwards, til this bucket)...
	// **********************************************************************************
public :
	// default/init/NULL value is DBL_MAX
	double _ApproxError_Local_Min, _ApproxError_Local_Avg, _ApproxError_Local_Max ;
	double _ApproxError_Cmltv_Min, _ApproxError_Cmltv_Avg, _ApproxError_Cmltv_Max ;
	double _ApproxError_Cmltv_wMax, _ApproxError_Cmltv_wMin, _ApproxError_Cmltv_wAvg ;
	double _ApproxError_Cmltv_var ;
public :
	inline double & ApproxError_Local_Min(void) { return _ApproxError_Local_Min ; }
	inline double & ApproxError_Local_Avg(void) { return _ApproxError_Local_Avg ; }
	inline double & ApproxError_Local_Max(void) { return _ApproxError_Local_Max ; }
	inline double & ApproxError_Cmltv_Min(void) { return _ApproxError_Cmltv_Min ; }
	inline double & ApproxError_Cmltv_Avg(void) { return _ApproxError_Cmltv_Avg ; }
	inline double & ApproxError_Cmltv_Max(void) { return _ApproxError_Cmltv_Max ; }

	int32_t Compute_ArroxError_Local(void) ;
	int32_t Compute_ArroxError_Cmltv(void) ;

	// **********************************************************************************
	// functions part of the original problem, assigned to this bucket
	// they functions don't belong to the bucket; normally they belong to the problem.
	// **********************************************************************************
protected :
	int32_t _nOriginalFunctions ;
	ARE::Function **_OriginalFunctions ;
	int32_t _OriginalWidth ; // cardinality of the original signature of this bucket; this includes _V; if <0, then unknown, should be computed
	// a union of the scopes of all original functions, including _V.
	// 2016-02-23 KK : these variables are in no particular order.
	int32_t *_OriginalSignature ;
public :
	inline int32_t nOriginalFunctions(void) const { return _nOriginalFunctions ; }
	inline ARE::Function *OriginalFunction(int32_t IDX) const { return _OriginalFunctions[IDX] ; }
	inline ARE::Function **OriginalFunctionsArray(void) { return _OriginalFunctions ; }
	inline int32_t OriginalWidth(void) const { return _OriginalWidth ; }
	int32_t SetOriginalFunctions(int32_t N, ARE::Function *FNs[], bool InvalidateSignature = true) ;
	int32_t AddOriginalFunctions(int32_t N, ARE::Function *FNs[], bool InvalidateSignature = true) ;

	int32_t ContainsOriginalFunction(ARE::Function & F)
	{
		for (int32_t i = 0 ; i < _nOriginalFunctions ; ++i) {
			if (&F == _OriginalFunctions[i]) 
				return i ;
			}
		return -1 ;
	}

	// **********************************************************************************
	// functions generated by other buckets (higher in the ordering; i.e. below this bucket in the bucket-tree), assigned to this bucket.
	// i.e. functions generated during (M)BE that contain _V.
	// these functions don't belong to this bucket; they belong to the (mini) bucket that generated them (i.e. e.g. this bucket should not delete them).
	// **********************************************************************************
protected :
	int32_t _nAugmentedFunctions ;
	ARE::Function **_AugmentedFunctions ;
	int32_t _AugmentedFunctionsArraySize ;
public :
	inline int32_t nAugmentedFunctions(void) const { return _nAugmentedFunctions ; }
	inline void ResetnAugmentedFunctions(void) { _nAugmentedFunctions = 0 ; }
	inline ARE::Function *AugmentedFunction(int32_t IDX) const { return _AugmentedFunctions[IDX] ; }
	inline ARE::Function **AugmentedFunctionsArray(void) { return _AugmentedFunctions ; }
	int32_t AddAugmentedFunction(ARE::Function & F) ;
	int32_t AddAugmentedFunctions(int32_t N, ARE::Function *FNs[], bool InvalidateSignature = true) ;
	int32_t RemoveAugmentedFunction(ARE::Function & F, bool InvalidateSignature) ;
	
	int32_t DestroyAugmentedFunctions(void)
	{
		for (int32_t i = _nAugmentedFunctions - 1 ; i >= 0 ; --i) {
			ARE::Function *fn = AugmentedFunction(i) ;
			delete fn ;
			}
		_nAugmentedFunctions = 0 ;
		InvalidateSignature() ;
		return 0 ;
	}

	int32_t ContainsAugmentedFunction(ARE::Function & F)
	{
		for (int32_t i = 0 ; i < _nAugmentedFunctions ; ++i) {
			if (&F == _AugmentedFunctions[i]) 
				return i ;
			}
		return -1 ;
	}

	// check that each augmented functions contains the buckets variable
	bool VerifyAugmentedFunctions(void) ;

	// **********************************************************************************
	// functions generated by other buckets (higher in the ordering; i.e. below this bucket in the bucket-tree), assigned to an ancestor of this bucket.
	// i.e. functions generated during (M)BE that do not contain _V, but that come from below and contain a variable that is ancestor of _V (in the bucket tree).
	// these functions don't belong to this bucket; they belong to the bucket that generated them (i.e. e.g. this bucket should not delete them).
	// **********************************************************************************
protected :
	int32_t _nIntermediateFunctions ;
	ARE::Function **_IntermediateFunctions ;
	int32_t _IntermediateFunctionsArraySize ;
public :
	inline int32_t nIntermediateFunctions(void) const { return _nIntermediateFunctions ; }
	inline void ResetnIntermediateFunctions(void) { _nIntermediateFunctions = 0 ; }
	inline ARE::Function *IntermediateFunction(int32_t IDX) const { return _IntermediateFunctions[IDX] ; }
	int32_t AddIntermediateFunction(ARE::Function & F) ;
	int32_t RemoveIntermediateFunction(ARE::Function & F) ;

	int32_t ContainsIntermediateFunction(ARE::Function & F)
	{
		for (int32_t i = 0 ; i < _nIntermediateFunctions ; ++i) {
			if (&F == _IntermediateFunctions[i]) 
				return i ;
			}
		return -1 ;
	}

	// check that none of the intermediate functions contain the buckets variable
	bool VerifyIntermediateFunctions(void)
	{
		for (int32_t i = 0 ; i < _nIntermediateFunctions ; ++i) {
			ARE::Function *f = _IntermediateFunctions[i] ;
			for (int32_t j = 0 ; j < f->N() ; ++j) {
				if (_V == f->Argument(j)) 
					return 1 ;
				}
			}
		return 0 ;
	}

	// **********************************************************************************
	// actual bucket function computation is done using mini-buckets...
	// typically a bucket ('s functions) are partitioned (divided) into 1 or more mini-buckets...
	// even if there is no partitioning, everything is places into a single mini-bucket, which is then (later) processed...
	// **********************************************************************************
protected :
	std::vector<MiniBucket *> _MiniBuckets ;
public :
	inline int32_t nMiniBuckets(void) const { return _MiniBuckets.size() ; }
	inline std::vector<MiniBucket *> & MiniBuckets(void) { return _MiniBuckets ; }

	int32_t DestroyMBPartitioning(void)
	{
		for (MiniBucket *mb : _MiniBuckets) {
			mb->Destroy() ;
			delete mb ;
			}
		_MiniBuckets.clear() ;
		return 0 ;
	}

public :

	// note : 
	// 1) scope(bucketfuncion) = _Signature - _V.
	// 2) when this function is called, we assume that scope of _OutputFunction is already ordered wrt the parent-bucket, 
	// since _OutputFunction belongs to the parent-bucket, and it is supposed to be already sorted.
	// this function will reorder the scopes of all functions in this bucket so that 
	// 1) _V is the last variable, 
	// 2) order of other variables agrees with the order of scope(bucketfuncion).
// 2016-02-20 KK : commented out; not using this fn right now.
//	int32_t ReorderFunctionScopesWrtParentBucket(bool IncludeOriginalFunctions, bool IncludeNewFunctions) ;

	// this creates the output function of this bucket and its scope; it does not fill in the table.
	// arguments of the buckets are sorted in the increasing order of distance to root.
	int32_t ComputeOutputFunctionWithScopeWithoutTable(
		// IN 
		int32_t * & TempSpaceForArglist, 
		int32_t TempSpaceForArglistSize,
		// OUT
		ARE::Function * & FN, 
		int32_t & fnMaxVar) ;

	// output function has been computed.
	// to stuff, e.g. cleanup (release FTBs of all child buckets).
	int32_t NoteOutputFunctionComputationCompletion(void) ;

	// processing complexity is the size of output table
	int64_t ComputeProcessingComplexity(bool ForMiniBuckets = true) ;
	int64_t ComputeEliminationComplexity(void) ;

	// do mini-bucket partitioning for this bucket.
	// in each MB, the number of non-elimination variables is iBound-1
	int32_t CreateMBPartitioning(
		// IN : if iBound<=0, then each function will in its own minibucket
		int32_t iBound, 
		// IN
		bool CreateTables, bool doMomentMatching, 
		// IN : Bound=-1 means lower bound, 0=Bound means approximation, Bound=1 upper
		signed char Bound, 
		// IN : if true, create only 1 MB
		bool noPartitioning, 
		// IN-OUT
		bool & AbandonIfMoreThan1MB, 
		// IN : helper arrays; passed in so that they can be created by a calling fn and reused for all buckets
		std::vector<int32_t> & key, std::vector<int64_t> & data, std::vector<int32_t> & helperArray) ;

	// compute output functions of all minibuckets
	// Bound=-1 means lower bound, 0=Bound means approximation, Bound=1 upper
	int32_t ComputeOutputFunctions(bool DoMomentMatching, signed char Bound, int64_t & TotalSumOutputFunctionsNumEntries) ;

	// bucket label is a combination of all original functions; we assume context assingment has a value for all variables in the bucket (including the variable(s) of this bucket).
	int32_t ComputeCost(int32_t *FullAssignment, ARE_Function_TableType & LabelValue) ;
	int32_t ComputeCostEx(int32_t *PathAssignment, ARE_Function_TableType & LabelValue) ;

	// bucket label is a combination of all original functions; we assume context assingment has a value for all variables in the bucket (including the variable(s) of this bucket).
	// also, set the CurrentValue of all agumented functions to the newly computed value.
	int32_t ComputeHeuristic(int32_t *FullAssignment, ARE_Function_TableType & HeuristicValue) ;
	int32_t ComputeHeuristicEx(int32_t *PathAssignment, ARE_Function_TableType & HeuristicValue) ;

	// compute value of Original, Augments, Intermediate functions... Hints&1 = Original, Hints&2 = Augments, Hints&4 = Intermediate
	int32_t ComputeValue(int32_t *FullAssignment, int32_t Hints, ARE_Function_TableType & Value) ;

	// bucket WeightedMiniBucket q as defined by WMB IS.
	int32_t ComputeWMBEq(
		// IN
		int32_t *ContextAssignment, 
		double w_IS, // IS weight
		double g_parent, // g of parent node
		double & fInt, // combination of all intermediate functions in this bucket
		// OUT
		double & q) ;

	// compute distribution on the first variable; this is used when a marginal distribution is required.
	// this eliminates all variables of this bucket, except for the given variable, combining original/augmented/intermediate functions.
	int32_t ComputeFirstVariableDistribution(
		// OUT
		ARE_Function_TableType *dist) ;

	// assuming this bucket has 1 variable, distribution on this variable.
	// values of all variables, except for the variable(s) of this bucket, are given in ContextValues.
	// this fn is used, e.g., when computing a cost-to-go for each configuration of this bucket's variable(s).
	int32_t ComputeFirstVariableDistributionEx(
		// IN : values to all context variables (= ancestors of this variable in the bucket tree)
		int32_t *ContextValues, 
		// OUT
		ARE_Function_TableType *dist) ;

	// delete all minibuckets
	int32_t DestroyPartitioning(void) ;

protected :
	// this var is used temporarily when BEworkspace is creating a total order of buckets (e.g. computing stats, comp order)
	Bucket *_NextInOrder ;
public :
	inline Bucket * & NextInOrder(void) { return _NextInOrder ; }

public :

	void Destroy(void) ;
	Bucket(void) ;
	Bucket(MBEworkspace & WS, int32_t IDX, int32_t V) ;
	virtual ~Bucket(void) ;
} ;

} // namespace BucketElimination

#endif // BUCKET_HXX_INCLUDED
