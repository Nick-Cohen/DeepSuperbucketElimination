#ifndef MiniBUCKET_HXX_INCLUDED
#define MiniBUCKET_HXX_INCLUDED

#include <vector>
#include <memory>

#include "Function.hxx"
#include "Problem.hxx"
#include "Utils/MiscUtils.hxx"

typedef enum
{
	MB_outputfn_type_table = 0, 
	MB_outputfn_type_NN = 1 
} MB_outputfn_type ;

namespace BucketElimination
{

class MBEworkspace ;
class Bucket ;
class MiniBucket ;

//typedef ARE::FnConstructor (*mboutputfncnstrctr)(MiniBucket & MB) ;
//extern mboutputfncnstrctr MBoutputfncnstrctr ;
//extern mboutputfncnstrctr NNoutputfncnstrctr ;

class MiniBucket
{
protected :
	MBEworkspace *_Workspace ;
	Bucket *_Bucket ;
	int32_t _IDX ; // index of this minibucket wrt the bucket
	int32_t _V ; // in case the bucket is created for a specic variable in bucket elimination, this is the var. -1 otherwise (e.g. in case of superbuckets).
public :
	inline MBEworkspace *Workspace(void) const { return _Workspace ; }
	ARE::ARP *Problem(void) const ;
	inline BucketElimination::Bucket *GetBucket(void) const { return _Bucket ; }
	inline int32_t IDX(void) const { return _IDX ; }
	inline int32_t V(void) const { return _V ; }
	inline void SetIDX(int32_t IDX) { _IDX = IDX ; }
	inline void SetWorkspace(MBEworkspace *ws) { _Workspace = ws ; }
	inline void SetBucket(BucketElimination::Bucket *b) { _Bucket = b ; }

	// variable(s) of the bucket that are eliminated when a bucket output fn is computed from functions of this bucket...
	// this may include bucket variables that are actually not in this MB...
	// we assume _Vars is sorted in increasing order...
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

	// signature of this bucket is the set of all functions arguments. width is the size of signature.
protected :
	std::vector<int32_t> _SortedSignature ; // signature sorted in non-decreasing order; computed to make AllowsFunction() check run faster.
	std::vector<int32_t> _SortedOutputFnScope ;
public :
	inline int32_t Width(void) const { return _SortedSignature.size() ; }
	inline const int32_t *Signature(void) const { return _SortedSignature.data() ; }
	inline const int32_t *SortedSignature(void) const { return _SortedSignature.data() ; }
	inline std::vector<int32_t> & SortedOutputFnScope(void) { return _SortedOutputFnScope ; }
	int32_t ComputeSignature(void) ;

	inline int32_t ComputeOutputFnVars(std::vector<int32_t> *OutputFnScope)
	{
		_SortedOutputFnScope.clear() ;
		if (_SortedOutputFnScope.capacity() < _SortedSignature.size()) {
			int32_t nNew = _SortedSignature.size() + 5 ; // a few more
			_SortedOutputFnScope.reserve(nNew) ;
			if (_SortedOutputFnScope.capacity() < nNew) 
				return -1 ;
			}
		int32_t n = 0, i = 0, j = 0 ;
		while (i < _Vars.size() && j < _SortedSignature.size()) {
			if (_Vars[i] == _SortedSignature[j]) {
				++i ; ++j ; } // common variable -> will be eliminated...
			else if (_Vars[i] < _SortedSignature[j]) 
				++i ; // variable only in _Vars -> no effect...
			else {
				_SortedOutputFnScope.push_back(_SortedSignature[j]) ;
				++n ; ++j ; // variable only in _SortedSignature -> will be in output fn...
				}
			}
		for (; j < _SortedSignature.size() ; ++j) { // remaining _SortedSignature variables will go in output fn...
			_SortedOutputFnScope.push_back(_SortedSignature[j]) ; ++n ; }
		if (nullptr != OutputFnScope) *OutputFnScope = _SortedOutputFnScope ;
		return n ;
	}
	inline int32_t ComputeElimVars(std::vector<int32_t> *ElimVars)
	{
		if (nullptr != ElimVars) {
			ElimVars->clear() ;
			if (ElimVars->capacity() < _Vars.size()) {
				ElimVars->reserve(_Vars.size()) ;
				if (ElimVars->capacity() < _Vars.size()) 
					return -1 ;
				}
			}
		int32_t n = 0, i = 0, j = 0 ;
		while (i < _Vars.size() && j < _SortedSignature.size()) {
			if (_Vars[i] == _SortedSignature[j]) {
				if (nullptr != ElimVars) ElimVars->push_back(_Vars[i]) ;
				++i ; ++j ; } // common variable -> will be eliminated...
			else if (_Vars[i] < _SortedSignature[j]) {
				++n ; ++i ; } // variable only in _Vars -> no effect...
			else 
				++j ; // variable only in _SortedSignature -> will be in output fn...
			}
		// remaining _Vars variables will have no effect...
		return n ;
	}

	int64_t ComputeTableSize(void) ;

protected :
	int32_t _nFunctions ;
	ARE::Function **_Functions ;
	int32_t _FunctionsArraySize ;
public :
	inline int32_t nFunctions(void) const { return _nFunctions ; }
	inline ARE::Function *Function(int32_t IDX) const { return _Functions[IDX] ; }
	inline ARE::Function **FunctionsArray(void) { return _Functions ; }
	int32_t AddFunction(ARE::Function & F, std::vector<int32_t> & HelperArray) ;
	int32_t RemoveFunction(ARE::Function & F) ;

	// compute number of variables in (the scope of) F that are not in this MB...
	// if ExcludeElimVars=1 -> MissingVars = vars not in _SortedSignature and not in _Vars... (i.e. non-elim variables in F that are not in this MB)
	// if ExcludeElimVars=0 -> MissingVars = vars not in _SortedSignature...
	int32_t ComputeVariablesNotPresent(ARE::Function & F, bool ExcludeElimVars, std::vector<int32_t> & MissingVars) ;

	// check if the given fn fits in this MB without breaking the limits.
	// 1=yes, 0=no, -1=function fails.
	// ASSUME VarsToIgnore is sorted...
	int32_t AllowsFunction(ARE::Function & F, int32_t maxNumVarsInMB, int32_t maxOutputFnScopeSize, std::vector<int32_t> & HelperArray) ;

	// ****************************************************************************************
	// Function generated by this bucket; its scope is _Signature-_V.
	// This function is generated by this bucket and belongs to this bucket.
	// However, after it is generated, it is placed in the parent-bucket.
	// Note that bucket function (as any other funcion), may have its table broken into pieces (blocks).
	// If the table is small, it may be kept in memory in its entirety (as 1 block); if it is large it may be broken into pieces (blocks) 
	// and each block stored as a table on the disk.
	// ****************************************************************************************

protected :

	ARE::Function *_OutputFunction ;

	// new function size (in number of elements) required to compute this bucket; this is the sum of child-function size + output fn size of this bucket.
	int64_t _ComputationNewFunctionSize ;

	// weighted mini-bucket weight of this minibucket
	double _WMBE_weight ;

public :

	inline ARE::Function & OutputFunction(void) { return *_OutputFunction ; }
	inline int64_t ComputationNewFunctionSize(void) const { return _ComputationNewFunctionSize ; }
	inline double & WMBE_weight(void) { return _WMBE_weight ; }

	int32_t CreateOutputFunction(ARE::FnConstructor Cnstrctr) ;

	// this creates the output function of this bucket and its scope; it does not fill in the table.
	// it sets the outputfn::Bucket member variable, but does not add it to the Bucket.
	int32_t ComputeOutputFunctionWithScopeWithoutTable(void) ;

	// processing complexity is the size of output table X numElimCombinations
	int64_t ComputeProcessingComplexity(void) ;

	// sample from the output fn; we will not compute the output fn explicitly...
	int32_t SampleOutputFunction(
		// IN
		int32_t varElimOperator, int64_t nSamples, 
		uint32_t RNGseed,  // when running on multiple threads, need a new seed each time, so that different calls don't repeat random number sequence
		// OUT
		int32_t & nFeaturesPerSample, 
		std::unique_ptr<int16_t[]> & Samples_signature, // len = nSamples * signature_size
		std::unique_ptr<float[]> & Samples_values, // len = nSamples
		float & min_value, 
		float & max_value, 
		float & sample_sum
		) ;

public :

	// when sampling from the output fn table, samples xml file name...
	// Prefix/Postfix are before/after the actual samples...
	int32_t GenerateSamplesXmlFilename(
		// IN : filename extra suffix...
		const char *sSuffix, 
		// OUT
		std::string & fnSamples, 
		std::string & fnNetwork, 
		std::string & sFNsignalling, 
		// OUT
		std::string & sFilePrefix, std::string & sFilePostFix,
		// IN : samples info/data...
		int32_t nSamples, double samples_min_value, double samples_max_value, double samples_sum
		) ;

	// output function has been computed.
	// to stuff, e.g. cleanup (release FTBs of all child buckets).
	int32_t NoteOutputFunctionComputationCompletion(void) ;

	// compute output function from scratch completely.
	// this fn is usually used when regular bucket elimination is used.
	// we build a MB table, by combining all MB functions. 
	// extras are : 
	//		-) adding avgMaxMarginal and subtract MaxMarginal.
	//		-) do power-sum with WMBEweight
	// the var(s) being eliminated are in the scope of avgMaxMarginal/MaxMarginal.
	virtual int32_t ComputeOutputFunction(
		// IN
		int32_t varElimOperator, 
		// IN : if result goes to file...
		bool ResultToFile, // not in memory
		bool SaveTableToFile, // table is memory, save to file
		// IN : temp, used when doing MomentMatching (product/max) (FU=fMaxMarginal fU=fAvgMaxMarginal) or CostShifting (product/sum)
		ARE::Function *FU, ARE::Function *fU, 
		// IN : used when running WMBE
		double WMBEweight
		) ;
	// Neural Network based version of ComputeOutputFunction()...
	// i.e. the output function is of type FunctionNN..
	virtual int32_t ComputeOutputFunction_NN(int32_t varElimOperator, 
		ARE::Function *FU, ARE::Function *fU, // used when doing MomentMatching (product/max) (FU=fMaxMarginal fU=fAvgMaxMarginal) or CostShifting (product/sum)
		double WMBEweight // used when running WMBE
		) ;
	// basic/general worker fn for eliminating a set of vars; we assume all ElimVars are in the minibucket signature. OutputFN.scope = MB.variables - ElimVars 
	virtual int32_t ComputeOutputFunction(
		// IN
		int32_t varElimOperator, 
		// IN : if result goes to file...
		bool ResultToFile, 
		// OUT
		ARE::Function & OutputFN, 
		// MISC
		const int32_t *ElimVars, int32_t nElimVars, int32_t *TempSpaceForVars, 
		double WMBEweight // used when running WMBE
		) ;
	// eliminate all vars; output is const fn.
	virtual int32_t ComputeOutputFunction_EliminateAllVars(int32_t varElimOperator) ;

	// manage the configuration of MBE processing...
	// we assume MB.signature values are in CurrentValueSet[] array; first OutputFN.scope, then ElimVars
	// ret : +1=failure, 0=ok, -1=all elimination combinations are enumerated...
	inline int32_t ManageEliminationConfiguration(
		int32_t nFNvars, const int32_t *FNvars, 
		int32_t nELIMvars, const int32_t *ELIMvars, 
		int64_t idxRun, // 0,1,2,... run index run=initial run
		std::unique_ptr<int32_t[]> & CurrentValueSet) // we assume in CurrentValueSet, first come all FNvars, then ELIMvars...
	{
		int32_t i, k, n = nFNvars + nELIMvars, v ;
		if (0 == idxRun) {
			for (i = nFNvars ; i < n ; ++i) CurrentValueSet[i] = 0 ; return 0 ; }
		// start from end; advance backwards
		ARE::ARP *p = Problem() ;
		if (nullptr == p) return 1 ;
		for (i = n-1 ; i >= nFNvars ; --i) {
			v = ELIMvars[i-nFNvars] ;
			k = p->K(v) ;
			if (++CurrentValueSet[i] < k) 
				return 0 ;
			CurrentValueSet[i] = 0 ;
			}
		return -1 ;
	}

	// manage the configuration of MBE processing...
	// we assume MB.signature values are in CurrentValueSet[] array that contains all problem variables; e.g. variable 527 value is at CurrentValueSet[527]
	// ret : +1=failure, 0=ok, -1=all elimination combinations are enumerated...
	inline int32_t ManageEliminationConfigurationEx(
		int32_t nELIMvars, const int32_t *ELIMvars, 
		int64_t idxRun, // 0,1,2,... run index run=initial run
		std::unique_ptr<int32_t[]> & CurrentValueSet)
		{
		int32_t i, k, v ;
		if (0 == idxRun) {
			for (i = 0 ; i < nELIMvars ; ++i) CurrentValueSet[ELIMvars[i]] = 0 ; return 0 ; }
		// start from end; advance backwards
		ARE::ARP *p = Problem() ;
		if (nullptr == p) return 1 ;
		for (i = nELIMvars-1 ; i >= 0 ; --i) {
			v = ELIMvars[i] ;
			k = p->K(v) ;
			if (++CurrentValueSet[v] < k) 
				return 0 ;
			CurrentValueSet[v] = 0 ;
			}
		return -1 ;
		}

	inline void ApplyVarEliminationOperator(int32_t varElimOperator, bool ValueAreLogSpace, ARE_Function_TableType & V, ARE_Function_TableType v)
	{
		if (VAR_ELIMINATION_TYPE_SUM == varElimOperator) {
			if (ValueAreLogSpace) 
				LOG_OF_SUM_OF_TWO_NUMBERS_GIVEN_AS_LOGS(V, V, v)
			else 
				V += v ;
			}
		else if (VAR_ELIMINATION_TYPE_MAX == varElimOperator) 
			{ if (v > V) V = v ; }
		else if (VAR_ELIMINATION_TYPE_MIN == varElimOperator) 
			{ if (v < V) V = v ; }
	}

public :

	void Initalize(BucketElimination::Bucket & B, int32_t IDX) ;

	void Destroy(void) ;
	MiniBucket(void) ;
	MiniBucket(MBEworkspace & WS, int32_t IDX, int32_t V, ARE::FnConstructor Cnstrctr = ARE::FunctionConstructor) ;
	virtual ~MiniBucket(void) ;
} ;

} // namespace BucketElimination

#endif // MiniBUCKET_HXX_INCLUDED
