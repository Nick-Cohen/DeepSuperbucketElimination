#include <stdlib.h>
#include <memory.h>

#include <Function.hxx>
#include <Function-NN.hxx>
#include <Bucket.hxx>
#include <MBEworkspace.hxx>
#include <Bucket.hxx>
#include <MiniBucket.hxx>
#include <Sort.hxx>
#include "Utils/MersenneTwister.h"

/*
ARE::FnConstructor default_mboutputfncnstrctr(BucketElimination::MiniBucket & MB)
	{ return ARE::FunctionConstructor ; }
ARE::FnConstructor default_nnoutputfncnstrctr(BucketElimination::MiniBucket & MB)
	{ return ARE::FunctionNNConstructor ; }
BucketElimination::mboutputfncnstrctr BucketElimination::MBoutputfncnstrctr = default_mboutputfncnstrctr ;
BucketElimination::mboutputfncnstrctr BucketElimination::NNoutputfncnstrctr = default_nnoutputfncnstrctr ;
*/

BucketElimination::MiniBucket::MiniBucket(void)
	:
	_Workspace(NULL),
	_IDX(-1), 
	_nFunctions(0),
	_Functions(NULL), 
	_FunctionsArraySize(0), 
	_OutputFunction(NULL), 
	_ComputationNewFunctionSize(-1), 
	_WMBE_weight(DBL_MAX)
{
}


BucketElimination::MiniBucket::~MiniBucket(void)
{
	Destroy() ;
}


BucketElimination::MiniBucket::MiniBucket(MBEworkspace & WS, int32_t IDX, int32_t V, ARE::FnConstructor Cnstrctr)
	:
	_Workspace(&WS),
	_IDX(IDX), 
	_V(V), 
	_nFunctions(0), 
	_Functions(NULL), 
	_FunctionsArraySize(0), 
	_OutputFunction(NULL), 
	_ComputationNewFunctionSize(-1), 
	_WMBE_weight(DBL_MAX)
{
	BucketElimination::Bucket *b = WS.getBucket(V) ;
	_Bucket = b ;
	// if Var is given, add it
	if (V >= 0) 
		AddVar(V) ;
	// fix up output function of the bucket
	if (NULL != Cnstrctr) 
		CreateOutputFunction(Cnstrctr) ;
}


void BucketElimination::MiniBucket::Destroy(void)
{
	if (NULL != _Functions) {
		delete [] _Functions ;
		_Functions = NULL ;
		_FunctionsArraySize = 0 ;
		}
	_SortedSignature.clear() ;
	_SortedOutputFnScope.clear() ;
	_nFunctions = 0 ;
	_Vars.clear() ;
	_ComputationNewFunctionSize = -1 ;
	_WMBE_weight = DBL_MAX ;
}


void BucketElimination::MiniBucket::Initalize(BucketElimination::Bucket & B, int32_t IDX)
{
	_Bucket = &B ;
	_Workspace = B.Workspace() ;
	_IDX = IDX ;
	_V = B.Var(0) ;
	if (NULL != _OutputFunction) {
		_OutputFunction->SetProblem(_Workspace->Problem()) ;
		_OutputFunction->SetWS(B.Workspace()) ;
		_OutputFunction->SetIDX(-(_V+1)) ;
		}
}


ARE::ARP *BucketElimination::MiniBucket::Problem(void) const 
	{ return nullptr != _Workspace ? _Workspace->Problem() : nullptr ; }

int32_t BucketElimination::MiniBucket::CreateOutputFunction(ARE::FnConstructor Cnstrctr)
{
	if (NULL == Cnstrctr) 
		return 1 ;
	if (NULL != _OutputFunction) 
		{ _OutputFunction->Destroy() ; delete _OutputFunction ; _OutputFunction = NULL ; }
/*	 if(_V == 10 ){
         printf("THIS IS VERY BAD THINF THAT I AM DOING\n");
         printf("THIS is V %d", _V);
         _OutputFunction = dynamic_cast<ARE::Function*>(ARE::FunctionNNConstructor());
         printf("SUCCESSFULLY created a functionNN \n");
	 }
	 else */
	 {
         _OutputFunction = (*Cnstrctr)() ;
	 }
	_OutputFunction->SetOriginatingBucket(_Bucket) ;
	_OutputFunction->SetOriginatingMiniBucket(this) ;
	_OutputFunction->Initialize(_Workspace, _Workspace->Problem(), -1) ;
	return 0 ;
}


int32_t BucketElimination::MiniBucket::ComputeVariablesNotPresent(ARE::Function & F, bool ExcludeElimVars, std::vector<int32_t> & MissingVars)
{
	if (F.N() <= 0) 
		return 0 ;

	int32_t i, j, n ;
	int32_t *fSortedArgs = F.SortedArgumentsList(true) ;
	if (nullptr == fSortedArgs) {
		return 1 ;
/*		for (i = F.N()-1 ; i >= 0 ; --i) {
			int32_t v = F.Argument(i) ;
			for (j = _Width-1 ; j >= 0 ; --j) {
				if (_Signature[j] == v) 
					break ;
				}
			if (j >= 0) continue ;
			if (ExcludeElimVars) {
				for (j = _Vars.size()-1 ; j >= 0 ; --j) {
					if (_Vars[j] == v) 
						break ;
					}
				if (j >= 0) continue ;
				}
			MissingVars.push_back(v) ;
			}*/
		}

	MissingVars.clear() ;
	if (MissingVars.capacity() < F.N()) {
		MissingVars.reserve(F.N()) ;
		if (MissingVars.capacity() < F.N()) 
			return 1 ;
		}

	i = j = 0 ;
	while (i < _SortedSignature.size() && j < F.N()) {
		if (_SortedSignature[i] == fSortedArgs[j]) 
			{ ++i ; ++j ; } // common variable
		else if (_SortedSignature[i] < fSortedArgs[j]) 
			{ ++i ; } // variable only in _SortedSignature
		else 
			{ MissingVars.push_back(fSortedArgs[j]) ; ++j ; } // variable only in fSortedArgs
		}
	for (; j < F.N() ; ++j) // add the rest of variables in fSortedArgs
		MissingVars.push_back(fSortedArgs[j]) ;
	if (ExcludeElimVars) {
		// check against _Vars; there may be variables in _Vars and F that are not in _SortedSignature
		i = j = n = 0 ;
		while (i < MissingVars.size() && j < _Vars.size()) {
			if (MissingVars[i] == _Vars[j]) 
				{ ++i ; ++j ; } // common variable -> eliminate it
			else if (MissingVars[i] < _Vars[j]) 
				{ MissingVars[n] = MissingVars[i] ; ++n ; ++i ; } // variable only in MissingVars -> keep it
			else 
				{ ++j ; } // variable only in _Vars -> ignore it
			}
		MissingVars.resize(n) ;
		}

	return 0 ;
}


int32_t BucketElimination::MiniBucket::AllowsFunction(ARE::Function & F, int32_t maxNumVarsInMB, int32_t maxOutputFnScopeSize, std::vector<int32_t> & HelperArray)
{
	if (0 == F.N() || 0 == _nFunctions) 
		return 1 ; // const fn can placed in any minibucket; a fn can always be added to an empty MB...

//	int32_t iBound = maxOutputFnScopeSize + 1 ;

	// must have current width of the minibucket.
	if (_nFunctions > 0) {
		if (0 == _SortedSignature.size()) {
			if (0 != ComputeSignature()) 
				return -1 ;
			}
		}
	else _SortedSignature.clear() ;

	if (F.N() > maxNumVarsInMB) 
		return 0 ; // this minibucket is not empty (_nFunctions>0), and since this F signature is larger than maxNumVarsInMB, so we know we will go over limit.
	if ((_SortedSignature.size() + F.N()) <= maxNumVarsInMB) 
		return 1 ; // _Width + F.N() is upper bound on the new width if we added this fn.

	if (HelperArray.capacity() < F.N()) {
		HelperArray.reserve(F.N()) ;
		if (HelperArray.capacity() < F.N()) 
			return -1 ;
		}
	HelperArray.clear() ;
	if (0 != ComputeVariablesNotPresent(F, true, HelperArray)) 
		return -1 ;
	int32_t nMissing = HelperArray.size() ;
	if (0 == nMissing) 
		return 1 ; // all arguments of the function are already in the mini-bucket
	return (_SortedOutputFnScope.size() + nMissing) <= maxOutputFnScopeSize ? 1 : 0 ;
}


int64_t BucketElimination::MiniBucket::ComputeTableSize(void)
{
	if (0 == _SortedSignature.size()) 
		return 0 ;
	int64_t ts = 1 ;
	ARE::ARP *p = _Workspace->Problem() ;
	for (int32_t i = _SortedSignature.size()-1 ; i >= 0 ; --i) {
		int32_t v = _SortedSignature[i] ;
		if (v == _V) continue ;
		ts *= p->K(v) ;
		if (ts < 0) {
			// overflow ?
			ts = -1 ;
			return ts ;
			}
		}
	return ts ;
}

int32_t BucketElimination::MiniBucket::AddFunction(ARE::Function & F, std::vector<int32_t> & HelperArray)
{
	if (_nFunctions > 0) {
		if (0 == _SortedSignature.size()) {
			if (0 != ComputeSignature()) 
				return 1 ;
			}
		}
	else {
		_SortedOutputFnScope.clear() ;
		_SortedSignature.clear() ;
		}

	int32_t *fSortedArgs = F.SortedArgumentsList(true) ;
	if (nullptr == fSortedArgs) 
		return 1 ;

	// check if we have enough space
	if (_nFunctions+1 > _FunctionsArraySize) {
		int32_t newsize = _FunctionsArraySize + 8 ;
		ARE::Function **newspace = new ARE::Function*[newsize] ;
		if (NULL == newspace) 
			return 1 ;
		if (_nFunctions > 0) 
			memcpy(newspace, _Functions, sizeof(ARE::Function *)*_nFunctions) ;
		if (NULL == _Functions) 
			delete [] _Functions ;
		_Functions = newspace ;
		_FunctionsArraySize = newsize ;
		}

	_Functions[_nFunctions++] = &F ;

	// fix up SortedSignature, etc.
	int32_t nNew = F.N() + _SortedSignature.size() ;
	HelperArray.clear() ;
	if (HelperArray.capacity() < nNew) {
		HelperArray.reserve(nNew) ;
		if (HelperArray.capacity() < nNew) 
			return -1 ;
		}

	// compute union of SortedSignature & F.Signature
	int32_t i = 0, j = 0 ;
	while (i < _SortedSignature.size() && j < F.N()) {
		if (_SortedSignature[i] == fSortedArgs[j]) {
			HelperArray.push_back(_SortedSignature[i]) ; ++i ; ++j ; } // common variable
		else if (_SortedSignature[i] < fSortedArgs[j]) {
			HelperArray.push_back(_SortedSignature[i]) ; ++i ; } // variable only in _SortedSignature
		else {
			HelperArray.push_back(fSortedArgs[j]) ; ++j ; } // variable only in fSortedArgs
		}
	// add the rest of variables
	// note : either i or j is out of bounds of its array...
	for (; i < _SortedSignature.size() ; ++i) {
		HelperArray.push_back(_SortedSignature[i]) ; }
	for (; j < F.N() ; ++j) {
		HelperArray.push_back(fSortedArgs[j]) ; }
	_SortedSignature = HelperArray ;

	// compute _SortedOutputFnScope as _SortedSignature - _Vars
	i = j = 0 ;
	_SortedOutputFnScope.clear() ;
	while (i < _SortedSignature.size() && j < _Vars.size()) {
		if (_SortedSignature[i] == _Vars[j]) {
			++i ; ++j ; } // common variable
		else if (_SortedSignature[i] < _Vars[j]) {
			_SortedOutputFnScope.push_back(_SortedSignature[i]) ; ++i ; } // variable only in _SortedSignature
		else {
			++j ; } // variable only in _Vars
		}
	for (; i < _SortedSignature.size() ; ++i) { // add the rest of variables in _SortedSignature
		_SortedOutputFnScope.push_back(_SortedSignature[i]) ; }

#ifdef _DEBUG
	std::vector<int32_t> varsOF, varsEL ;
	ComputeOutputFnVars(&varsOF) ;
	ComputeElimVars(&varsEL) ;
	if (varsOF.size() + varsEL.size() != _SortedSignature.size()) {
		int32_t error = 1 ;
		}
#endif

	return 0 ;
}


int32_t BucketElimination::MiniBucket::RemoveFunction(ARE::Function & F)
{
	int32_t i, n = 0 ;
	for (i = _nFunctions - 1 ; i >= 0 ; i--) {
		if (&F == _Functions[i]) {
			_Functions[i] = _Functions[--_nFunctions] ;
			++n ;
			}
		}
	// if any removals, update signature
	if (n > 0) 
		ComputeSignature() ;
	return 0 ;
}


int32_t BucketElimination::MiniBucket::ComputeSignature(void)
{
// DEBUGGGGG
/*if (NULL != ARE::fpLOG) {
	fprintf(ARE::fpLOG, "\n         ComputeSignature var=%d", (int32_t) V()) ;
	fflush(ARE::fpLOG) ;
	}*/
	_SortedSignature.clear() ;

	// compute approx width
	if (0 == _nFunctions) 
		return 0 ;
	int32_t i, n = 0 ;
	for (i = 0 ; i < _nFunctions ; i++) 
		n += _Functions[i]->N() ;
	// n is an upper bound on the width
	if (n <= 0) 
		return 0 ;

	// OriginalSignature is part of Signature
	if (_SortedSignature.capacity() < n+5) 
		{ _SortedSignature.reserve(n+5) ; if (_SortedSignature.capacity() < n+5) goto failed ; }

	// add scopes of bucketfunctions to the signature
	int32_t j, k ;
	for (i = 0 ; i < _nFunctions ; i++) {
		ARE::Function *f = _Functions[i] ;
		for (j = 0 ; j < f->N() ; j++) {
			int32_t v = f->Argument(j) ;
			for (k = _SortedSignature.size()-1 ; k >= 0 ; --k) 
				{ if (_SortedSignature[k] == v) break ; }
			if (k >= 0) 
				continue ;
			_SortedSignature.push_back(v) ;
			}
		}

	if (_SortedSignature.size() > 1) {
		int32_t left[32], right[32] ;
		QuickSortLong2((int32_t*) _SortedSignature.data(), _SortedSignature.size(), left, right) ;
		}

	return 0 ;
failed :
	return 1 ;
}


int64_t BucketElimination::MiniBucket::ComputeProcessingComplexity(void)
{
	int64_t n = 1 ;
	for (int32_t i = _SortedSignature.size()-1 ; i >= 0 ; --i) {
		n *= _Workspace->Problem()->K(_SortedSignature[i]) ;
		}
	return n ;
}


int32_t BucketElimination::MiniBucket::ComputeOutputFunctionWithScopeWithoutTable(void)
{
	if (NULL == _Bucket || NULL == _OutputFunction) 
		return 1 ;

// DEBUGGGGG
/*if (NULL != ARE::fpLOG) {
	fprintf(ARE::fpLOG, "\n         ComputeOutputFunctionWithScopeWithoutTable var=%d", (int32_t) V()) ;
	fflush(ARE::fpLOG) ;
	}*/

	// dump current bucket fn; cleanup.
	_OutputFunction->Destroy() ;

	if (_nFunctions > 0) {
		if (0 == _SortedSignature.size()) {
			if (0 != ComputeSignature()) 
				return 1 ;
			}
		}
	else 
		_SortedSignature.clear() ;

	std::vector<int32_t> scope ;
	if (ComputeOutputFnVars(&scope) < 0) 
		return 1 ;

	if (0 == scope.size()) {
		// _Width=0 can only be when this bucket has no functions
		// _Width=_nVars means all functions in this bucket have only _Vars in their scope; this bucket has a function with const-value; in this case
		// we will still compute the const-value, but the output function does not go anywhere.
		_OutputFunction->SetOriginatingBucket(_Bucket) ;
		_OutputFunction->SetOriginatingMiniBucket(this) ;
		return 0 ;
		}

#ifdef _DEBUG
	for (int32_t jjj = 0 ; jjj < _SortedSignature.size() ; jjj++) {
		if (_SortedSignature[jjj] < 0 || _SortedSignature[jjj] >= _Bucket->Workspace()->Problem()->N()) {
			int32_t error = 1 ;
			}
		}
#endif // _DEBUG

	double newFNsize;
	double temp_sum;
	double of_log;
	double log_max;

	// create and initialize bucket function
	if (0 != _OutputFunction->SetArguments(scope.size(), scope.data())) 
		goto failed ;
	of_log = _OutputFunction->GetTableSize_Log10() ;
	log_max = log10((double) _I64_MAX) ;
	if (of_log < log_max) 
		_OutputFunction->ComputeTableSize() ;
	else {
		// output fn size too large; continue since we can still compute the output fn scope, which is what this fn is supposed to do.
		}
	if (_nFunctions > 0) 
		_OutputFunction->SetType(_Functions[0]->Type()) ;

	// find appropriate parent bucket and assign the bucket function to it
	{
	const int32_t *varpos = _Workspace->VarPos() ;
	int32_t v = _OutputFunction->GetHighestOrderedVariable(varpos) ;
	BucketElimination::Bucket *parentbucket = _Workspace->MapVar2Bucket(v) ;
	if (NULL == parentbucket) 
		// this is not supposed to happen
		goto failed ;
	if (parentbucket->IDX() >= _Bucket->IDX()) 
		// this is not supposed to happen
		goto failed ;
	_OutputFunction->SetBucket(parentbucket) ;
	_OutputFunction->SetOriginatingBucket(_Bucket) ;
	_OutputFunction->SetOriginatingMiniBucket(this) ;
	}

	// compute new fn size of computing this bucket; we assume child buckets are set up.
	// compute in log space since individual fn sizes may be huge.
	/*
		from http://en.wikipedia.org/wiki/List_of_logarithmic_identities
		log10(SUM a_i) = log10(a_0) + log10(1 + SUM (a_i/a_0)) = log10(a_0) + log10(1 + SUM 10^(log10(a_i) - log10(a_0)))
	*/
	newFNsize = _OutputFunction->GetTableSize_Log10() ;
	temp_sum = 1.0 ;
	if (newFNsize >= 0.0) {
		for (int32_t i = 0 ; i < _nFunctions ; i++) {
			ARE::Function *f = _Functions[i] ;
			if (NULL == f->OriginatingMiniBucket()) 
				continue ; // this is original fn, not MBE generated function.
			double fnsize = f->GetTableSize_Log10() ;
			if (fnsize < 0.0) 
				{ newFNsize = -1.0 ; break ; }
			temp_sum += pow(10.0, fnsize - newFNsize) ;
			}
		}
	_ComputationNewFunctionSize = 0 ;
	if (newFNsize >= 0.0) {
		newFNsize += log10(temp_sum) ;
		// if size overflows int64_t, set it to max.
		double log_max = log10((double) _I64_MAX) ;
		if (newFNsize < log_max) 
			_ComputationNewFunctionSize = pow(10.0, newFNsize) ;
		else 
			_ComputationNewFunctionSize = _I64_MAX ;
		}
// DEBUGGGGG
/*if (NULL != ARE::fpLOG) {
	int64_t tNOW = ARE::GetTimeInMilliseconds() ;
	fprintf(ARE::fpLOG, "\n         ComputeOutputFunctionWithScopeWithoutTable var=%d newfncompsize=%lld", (int32_t) V(), _ComputationNewFunctionSize) ;
	fflush(ARE::fpLOG) ;
	}*/

	return 0 ;
failed :
	_OutputFunction->Destroy() ;
	return 1 ;
}


int32_t BucketElimination::MiniBucket::ComputeOutputFunction_EliminateAllVars(int32_t varElimOperator)
{
	int32_t j, k, ret = 0 ;

	MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
	if (NULL == bews) 
		return ERRORCODE_generic ;
	ARE::ARP *problem = bews->Problem() ;
	if (NULL == problem) 
		return ERRORCODE_generic ;
	ARE_Function_TableType nv = bews->FnCombinationNeutralValue() ;
	ARE::Function & f = OutputFunction() ;
	ARE_Function_TableType & V = f.ConstValue() ;
	const int32_t w = Width() ;
	if (w < 0 || _nFunctions < 1) {
		V = nv ;
		return 0 ;
		}
	const int32_t *signature = Signature() ;

	if (w > MAX_NUM_VARIABLES_PER_BUCKET) 
		return ERRORCODE_too_many_variables ;
	if (_nFunctions > MAX_NUM_FUNCTIONS_PER_BUCKET) 
		return ERRORCODE_too_many_functions ;

	int32_t values[MAX_NUM_VARIABLES_PER_BUCKET] ; // this is the current value combination of the arguments of this function (table block).
	ARE::Function *flist[MAX_NUM_FUNCTIONS_PER_BUCKET] ; // this is a list of input functions

	ARE_Function_TableType const_factor = bews->FnCombinationNeutralValue() ;
	int32_t nFNs = 0 ;
	for (j = 0 ; j < _nFunctions ; j++) {
		ARE::Function *f = _Functions[j] ;
		if (NULL == f) return ERRORCODE_fn_ptr_is_NULL ;
		if (0 == f->N()) bews->ApplyFnCombinationOperator(const_factor, f->ConstValue()) ;
		else { flist[nFNs++] = f ; f->ComputeArgumentsPermutationList(w, signature) ; }
		}

	int64_t ElimSize = 1 ;
	for (j = 0 ; j < w ; j++) 
		ElimSize *= problem->K(signature[j]) ;

	ARE::Function *MissingFunction = NULL ;
	int64_t MissingBlockIDX = -1 ;

	V = bews->VarEliminationDefaultValue() ;
	for (j = 0 ; j < w ; j++) 
		values[j] = 0 ;
	for (int64_t ElimIDX = 0 ; ElimIDX < ElimSize ; ElimIDX++) {
		ARE_Function_TableType value = nv ;
		for (j = 0 ; j < nFNs ; j++) { // note : it should be that 0 != flist[j]->N(). note : it is assumed that flist[j] has 1 block. note : it is assumed that order of flist[j] arguments is the same as signature.
			int64_t adr = flist[j]->ComputeFnTableAdr_wrtLocalPermutation(values, problem->K()) ;
			bews->ApplyFnCombinationOperator(value, flist[j]->TableEntry(adr)) ;
			}
		ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), V, value) ;
		// go to next argument value combination
		ARE::EnumerateNextArgumentsValueCombination(w, signature, values, problem->K()) ;
		}
	bews->ApplyFnCombinationOperator(V, const_factor) ;
done :
	return ret ;
}


int32_t BucketElimination::MiniBucket::SampleOutputFunction(int32_t varElimOperator, int64_t nSamples, uint32_t RNGseed, int32_t & nFeaturesPerSample, std::unique_ptr<int16_t[]> & Samples_signature, std::unique_ptr<float[]> & Samples_values, float & min_value, float & max_value, float & sample_sum)
{
	Samples_signature.reset() ;
	Samples_values.reset() ;
	min_value =  std::numeric_limits<float>::infinity() ; // std::numeric_limits<float>::max() ;
	max_value = -std::numeric_limits<float>::infinity() ; // std::numeric_limits<float>::min() ;

	MTRand RNG(RNGseed) ; // need a new seed each time, so that different calls don't repeat random number sequence

	if (NULL == _OutputFunction)
        return ERRORCODE_generic ;
    MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
    if (NULL == bews)
        return ERRORCODE_generic ;
    ARE::ARP *problem = bews->Problem() ;
    if (NULL == problem)
        return ERRORCODE_generic ;
    if (_SortedSignature.size() < 0) {
		// assume signature is computed...
		return ERRORCODE_generic ;
	    }

	std::vector<int32_t> elimVars ;
	if (ComputeElimVars(&elimVars) < 0) 
		return ERRORCODE_generic ;

    if (elimVars.size() < 1)
        // nothing to do; should not happen; however, allow this to pass as ok; calling fn should be able to handle this.
        return 0 ;
	const int32_t w = Width() ;
	if (w < 0 || _nFunctions < 1) 
		return 0 ;
	const int32_t *signature = Signature() ;
	int32_t i ;

    // generate some number of random samples...
//	std::unique_ptr<int32_t[]> vars(new int32_t[w]) ;
	std::unique_ptr<int32_t[]> vals(new int32_t[problem->N()]) ; // could use w?
	if (nullptr == vals) 
		return 1 ;
//	for (i = 0 ; i < _OutputFunction->N() ; ++i) 
//		vars[i] = _OutputFunction->Argument(i) ;
//	vars[i] = _V ;
	nFeaturesPerSample = _OutputFunction->N() ;
	int32_t sample_sig_size = nSamples * _OutputFunction->N() ;
	std::unique_ptr<int16_t[]> samples_signature(new int16_t[sample_sig_size]) ;
	if (nullptr == samples_signature) return 1 ;
	std::unique_ptr<float[]> samples_values(new float[nSamples]) ;
	if (nullptr == samples_values) return 1 ;
/*  samples_signiture = new int32_t *[nSamples] ;
    for (int nn=0; nn<nSamples; nn++)
        samples_signiture[nn] = new int32_t[_OutputFunction->N()];
    float *sample_values = new float[nSamples] ; */

	ARE_Function_TableType const_factor = bews->FnCombinationNeutralValue() ;
	int32_t nFNs = 0 ;
	std::vector<ARE::Function *> flist ; flist.reserve(_nFunctions) ; if (flist.capacity() != _nFunctions) return 1 ;
	for (int32_t j = 0 ; j < _nFunctions ; j++) {
		ARE::Function *f = _Functions[j] ;
		if (NULL == f) return ERRORCODE_fn_ptr_is_NULL ;
		if (0 == f->N()) bews->ApplyFnCombinationOperator(const_factor, f->ConstValue()) ;
		else { flist.push_back(f) ; } // f->ComputeArgumentsPermutationList(w, vars.get()) ; }
		}

	sample_sum = bews->VarEliminationDefaultValue() ;
	int64_t ElimIdx ;
	for (i = 0 ; i < nSamples ; ++i) {
		int16_t *sample_signature = samples_signature.get() + i*_OutputFunction->N() ;
		float *sample_value = samples_values.get() + i ;

        // generate assignment to _OutputFunction arguments
        for (int32_t j = 0 ; j < _OutputFunction->N() ; ++j) {
            int32_t v = _OutputFunction->Argument(j) ;
            int32_t domain_size_of_v = problem->K(v) ;
            int32_t value = RNG.randInt(domain_size_of_v-1) ;
			vals[v] = value ;
            sample_signature[j] = value ;
			}

		ARE_Function_TableType V = bews->VarEliminationDefaultValue(), v ;
		ElimIdx = -1 ;
do_next_elim_config :
		int32_t res = ManageEliminationConfigurationEx(elimVars.size(), elimVars.data(), ++ElimIdx, vals) ;
		if (res > 0) 
			return res ;
		if (res < 0) 
			goto done_with_this_elim_config ; // all elimination combinations have been enumerated...
		v = bews->FnCombinationNeutralValue() ;
		// compute value for this configuration : fNN argument assignment + _V=j
		for (int32_t l = 0 ; l < _nFunctions ; ++l) {
			ARE::Function *f = _Functions[l];
			if (NULL == f) continue ;
			// double fn_v = f->TableEntryEx(vals.get(), problem->K());
			double fn_v = f->TableEntryExNativeAssignment(vals.get(), problem->K());
			bews->ApplyFnCombinationOperator(v, fn_v) ;
			}
		ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), V, v) ;
		goto do_next_elim_config ;

done_with_this_elim_config :
		bews->ApplyFnCombinationOperator(V, const_factor) ;
		*sample_value = V ;

		if (V > max_value) max_value = V ;
		if (V < min_value) min_value = V ;
		LOG_OF_SUM_OF_TWO_NUMBERS_GIVEN_AS_LOGS(sample_sum, sample_sum, V)
		}

	// transfer samples to the output parameters...
	Samples_signature = std::move(samples_signature) ;
	Samples_values = std::move(samples_values) ;

	return 0 ;
}


int32_t BucketElimination::MiniBucket::GenerateSamplesXmlFilename(
	const char *sSuffix, std::string & fnSamples, std::string & fnNetwork, std::string& fnFNsignalling, std::string & sPrefix, std::string & sPostFix,
	int32_t nSamples, double samples_min_value, double samples_max_value, double samples_sum)
{
	MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
	if (NULL == bews)
		return ERRORCODE_generic ;
	ARE::ARP *problem = bews->Problem() ;
	if (NULL == problem)
		return ERRORCODE_generic ;

	std::unique_ptr<char[]> sBUF(new char[1024]) ;
	if (nullptr == sBUF) 
		return 1 ;
	char *buf = sBUF.get() ;
	std::string s ;

	bool data_is_log_space = problem->FunctionsAreConvertedToLogScale() ;

	fnSamples = "samples-"; fnNetwork = "nn-"; fnFNsignalling = "nnready-";
	// file name = list of vars being eliminated...
	for (int32_t i = 0 ; i < _Vars.size() ; ++i) {
		if (i > 0) { fnSamples += ';'; fnNetwork += ';'; fnFNsignalling += ';'; }
		sprintf(buf, "%d", (int) _Vars[i]) ;
		fnSamples += buf; fnNetwork += buf; fnFNsignalling += buf;
		}
	if (nullptr != sSuffix) {
		fnSamples += sSuffix; fnNetwork += sSuffix; fnFNsignalling += sSuffix; }
	fnSamples += ".xml" ;
	fnNetwork += ".jit" ;
	fnFNsignalling += ".txt";

	// generate scope and domain size lists
	if (nullptr != _OutputFunction) {
		s = " outputfnscope=\"" ;
		for (int32_t i = 0 ; i < _OutputFunction->N() ; ++i) {
			if (i > 0) s += ';' ;
			sprintf(buf, "%d", (int) _OutputFunction->Argument(i)) ;
			s += buf ;
			}
		s += "\" outputfnvariabledomainsizes=\"" ;
		for (int32_t i = 0 ; i < _OutputFunction->N() ; ++i) {
			if (i > 0) s += ';' ;
			sprintf(buf, "%d", (int) problem->K(_OutputFunction->Argument(i))) ;
			s += buf ;
			}
		s += '\"' ;
		}
	sPrefix = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" ;
	int32_t nFeaturesPerSample = _OutputFunction->N() ;
	sprintf(buf, "\n<samples n=\"%d\" nFeaturesPerSample=\"%d\"%s datainlogspace=\"%c\"", (int) nSamples, (int) nFeaturesPerSample, s.c_str(), (char) data_is_log_space ? 'Y' : 'N') ;
	sPrefix += buf ;
	if (samples_min_value < 1.0e+128) {
		sprintf(buf, " min=\"%g\"", (double) samples_min_value) ; sPrefix += buf ; }
	if (samples_max_value < 1.0e+128) {
		sprintf(buf, " max=\"%g\"", (double) samples_max_value) ; sPrefix += buf ; }
	if (samples_sum < 1.0e+128) {
		sprintf(buf, " sum=\"%g\"", (double) samples_sum) ; sPrefix += buf ; }
	if (fnNetwork.length() > 0) {
		sPrefix += " fnNetwork=\"" ;
		sPrefix += fnNetwork ;
		sPrefix += '\"' ;
		}
	sPrefix += '>' ;

	// sPostFix
	sPostFix = "\n</samples>" ;

	return 0 ;
}


int32_t BucketElimination::MiniBucket::ComputeOutputFunction(int32_t varElimOperator, bool ResultToFile, ARE::Function *FU, ARE::Function *fU, double WMBEweight)
{
	int32_t i, j, k ;

	// TODO : if _OutputFunction is of type FunctionNN, then do .....
	ARE::FunctionNN *fNN = dynamic_cast<ARE::FunctionNN *>(_OutputFunction) ;
	
	if (nullptr != fNN) {
		return ComputeOutputFunction_NN(varElimOperator, FU, fU, WMBEweight) ;
		}

	// if (NULL != fNN) {
// #ifdef INCLUDE_TORCH
//         printf("&&&&&&&&&&&&& IS THIS ONE NN &&&&&&&&&\n");
// //        return ComputeOutputFunction_NN(varElimOperator, FU, fU, WMBEweight);
// 		return 1 ; // todo fix previous line
// #else
// 		return 1 ; // TORCH/NN stuff not included...
// #endif // INCLUDE_TORCH
		// }

	MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
	if (NULL == bews) 
		return ERRORCODE_generic ;
	ARE::ARP *problem = bews->Problem() ;
	if (NULL == problem) 
		return ERRORCODE_generic ;
	if (_nFunctions > 0) {
		if (0 == _SortedSignature.size()) {
			if (0 != ComputeSignature()) 
				return ERRORCODE_generic ;
			}
		}
	else _SortedSignature.clear() ;
	std::vector<int32_t> elimVars ;
	if (ComputeElimVars(&elimVars) < 0) 
		return ERRORCODE_generic ;
	if (0 == elimVars.size()) 
		// nothing to do; should not happen; however, allow this to pass as ok; calling fn should be able to handle this.
		return 0 ;
	ARE::Function & OutputFN = OutputFunction() ;
	ARE_Function_TableType & f_const_value = OutputFN.ConstValue() ;
	if (_nFunctions < 1) {
		if (FN_COBINATION_TYPE_PROD == problem->FnCombinationType()) {
			// we are just summing over variables, each sum counts as 1.
			f_const_value = 1.0 ;
			for (int32_t ii = 0 ; ii < elimVars.size() ; ++ii) 
				f_const_value *= problem->K(elimVars[ii]) ;
			if (problem->FunctionsAreConvertedToLogScale()) 
				f_const_value = log10(f_const_value) ;
			}
		return 0 ;
		}

	std::unique_ptr<char[]> sBUF(new char[1024]) ;
	if (nullptr == sBUF) 
		return 1 ;
	char *buf = sBUF.get() ;

	OutputFN.ComputeTableSize() ;
	int32_t nA = OutputFN.N() ;
//	if (0 == nA) 
//		return ComputeOutputFunction_EliminateAllVars(varElimOperator) ;
	if (_SortedSignature.size() != nA + elimVars.size()) 
		return ERRORCODE_generic ;

	if (nA > 0 && ! ResultToFile) { // if result goes to file, don't need table...
		if (0 != OutputFN.AllocateTableData())
			return ERRORCODE_memory_allocation_failure ;
		}

#ifdef _DEBUG
	if (_SortedSignature.size() != nA + elimVars.size()) {
		int32_t error = 1 ;
		}
#endif

	int32_t vars[MAX_NUM_VARIABLES_PER_BUCKET] ; // this is the list of variables : vars2Keep + vars2Eliminate
	const int32_t *refFNarguments = OutputFN.Arguments() ;
	for (j = 0 ; j < nA ; ++j) 
		vars[j] = refFNarguments[j] ;
	for (; j < _SortedSignature.size() ; ++j) 
		vars[j] = elimVars[j - nA] ;

	ARE_Function_TableType const_factor = bews->FnCombinationNeutralValue() ;
	int32_t nFNs = 0 ;
	ARE::Function *flist[MAX_NUM_FUNCTIONS_PER_BUCKET] ; // this is a list of input functions
	for (j = 0 ; j < _nFunctions ; j++) {
		ARE::Function *f = _Functions[j] ;
		if (NULL == f) return ERRORCODE_fn_ptr_is_NULL ;
		if (0 == f->N()) bews->ApplyFnCombinationOperator(const_factor, f->ConstValue()) ;
		else  { flist[nFNs++] = f ; f->ComputeArgumentsPermutationList(_SortedSignature.size(), vars) ; }
		}
	if (NULL != FU && NULL != fU) {
		FU->ComputeArgumentsPermutationList(_SortedSignature.size(), vars) ;
		fU->ComputeArgumentsPermutationList(_SortedSignature.size(), vars) ;
		}

	int64_t ElimIdx, ElimSize = 1 ;
	for (j = 0 ; j < elimVars.size() ; j++) 
		ElimSize *= problem->K(elimVars[j]) ;

	ARE_Function_TableType *data = OutputFN.TableData(), vTableEntry, value ;
	double one_over_WMBEweight = WMBEweight < 1.0e+32 ? 1.0/WMBEweight : DBL_MAX ;
	int64_t tablesize = 0 == nA ? 1 : OutputFN.TableSize(), adr ;
	std::unique_ptr<int32_t[]> currentValueSet(new int32_t[_SortedSignature.size()]) ;
	if (nullptr == currentValueSet) 
		return 1 ;
	for (i = 0 ; i < nA ; ++i) currentValueSet[i] = 0 ;

	// sometimes the result should be written to file...
	FILE *fp = nullptr ;
	std::string sFN, sFNnn, sFNsignalling, sPrefix, sPostFix, sSample ;
	if (ResultToFile) {
		int64_t nSamples = tablesize ;
		float samples_min_value = 1.0e+129, samples_max_value = 1.0e+129, samples_sum = 1.0e+129 ;
		GenerateSamplesXmlFilename("-full", sFN, sFNnn, sFNsignalling, sPrefix, sPostFix, nSamples, samples_min_value, samples_max_value, samples_sum) ;
		fp = fopen(sFN.c_str(), "w") ;
		if (nullptr != fp) 
			fwrite(sPrefix.c_str(), 1, sPrefix.length(), fp) ;
		}

	for (int64_t KeepIDX = 0 ; KeepIDX < tablesize ; KeepIDX++) {
		ARE_Function_TableType & v = ResultToFile ? vTableEntry : (nullptr == data ? f_const_value : data[KeepIDX]) ;
		v = bews->VarEliminationDefaultValue() ;
		ElimIdx = -1 ;
do_next_elim_config :
		int32_t res = ManageEliminationConfiguration(nA, refFNarguments, elimVars.size(), elimVars.data(), ++ElimIdx, currentValueSet) ;
		if (res > 0) 
			return res ;
		if (res < 0) 
			goto done_with_this_elim_config ; // all elimination combinations have been enumerated...
		value = bews->FnCombinationNeutralValue() ;
		for (j = 0 ; j < nFNs ; j++) {
			double fn_v = 0.0 ;
			try {
				fn_v = flist[j]->TableEntryEx(currentValueSet.get(), problem->K()); // NEW
				}
			catch (...) {
				printf("\nEXCEPTION : flist[j]->TableEntryEx(...) in BucketElimination::MiniBucket::ComputeOutputFunction(...)") ;
				printf("\n");
				exit(151) ;
				}
//			adr = flist[j]->ComputeFnTableAdr_wrtLocalPermutation(values, problem->K()) ;
//			bews->ApplyFnCombinationOperator(value, flist[j]->TableEntry(adr)) ;
			bews->ApplyFnCombinationOperator(value, fn_v) ;
			}
		// if value would not change elim result (e.g. the problem is product-sum and value is 0 so far), continue.
		if (value == bews->VarEliminationDefaultValue()) 
			goto do_next_elim_config ;
		// apply MomentMatching/CostShifting, if given
		if (NULL != FU && NULL != fU) {
			ARE_Function_TableType valueMM ;
			adr = FU->ComputeFnTableAdr_wrtLocalPermutation(currentValueSet.get(), problem->K()) ;
			valueMM = FU->TableEntry(adr) ;
//			bews->ApplyFnCombinationOperator(value, FU->TableEntry(adr)) ;
			adr = fU->ComputeFnTableAdr_wrtLocalPermutation(currentValueSet.get(), problem->K()) ;
			bews->ApplyFnDivisionOperator(valueMM, fU->TableEntry(adr)) ;
			// apply pow(value, WMBEweight)
			if (WMBEweight < 1.0e+32) 
				{ if (problem->FunctionsAreConvertedToLogScale()) valueMM *= WMBEweight ; else valueMM = pow(valueMM, WMBEweight) ; }
			bews->ApplyFnCombinationOperator(value, valueMM) ;
			}
		// apply MB weight, if given
		if (WMBEweight < 1.0e+32) {
			if (problem->FunctionsAreConvertedToLogScale()) 
				value /= WMBEweight ;
			else 
				value = pow(value, one_over_WMBEweight) ;
			}
		ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), v, value) ;
		goto do_next_elim_config ;

done_with_this_elim_config :
		if (WMBEweight < 1.0e+32) {
			if (problem->FunctionsAreConvertedToLogScale()) 
				v *= WMBEweight ;
			else 
				v = pow(v, WMBEweight) ;
			}
		bews->ApplyFnCombinationOperator(v, const_factor) ;
goto_next_keep_value_combination :
		// if needed, add to file
		if (ResultToFile && nullptr != fp) {
			sSample = "\n   <sample signature=\"" ;
			for (int32_t iSig = 0 ; iSig < nA ; ++iSig) {
				if (iSig > 0) sSample += ';' ;
				sprintf(buf, "%d", (int) currentValueSet[iSig]) ; sSample += buf ;
				}
			sprintf(buf, "\" value=\"%g\"/>", (double) v) ; sSample += buf ;
			fwrite(sSample.c_str(), 1, sSample.length(), fp) ;
			}
		// go to next argument value combination
		ARE::EnumerateNextArgumentsValueCombination(nA, vars, currentValueSet.get(), problem->K()) ;
		}

	if (nullptr != fp) {
		fwrite(sPostFix.c_str(), 1, sPostFix.length(), fp) ;
		fclose(fp) ;
		}

	static bool saveToFile = false ;

	
	if (saveToFile && nullptr != _OutputFunction) {
		// std::string fn("bucket-output-fn"+std::to_string(_OutputFunction->IDX())+".xml") ;
		std::string fn("bucket-output-fn") ;
		for (int32_t i = 0 ; i < _Vars.size() ; ++i) {
			if (i > 0) { fn += ';';}
			fn += std::to_string(_Vars[i]) ;
			}
		fn += ".xml" ;
		_OutputFunction->SaveToFile(fn) ;
		}

	return 0 ;
}


int32_t BucketElimination::MiniBucket::ComputeOutputFunction(int32_t varElimOperator, bool ResultToFile, ARE::Function & OutputFN, const int32_t *ElimVars, int32_t nElimVars, int32_t *TempSpaceForVars, double WMBEweight)
{
	int32_t i, j, k ;

	MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
	if (NULL == bews) 
		return ERRORCODE_generic ;
	ARE::ARP *problem = bews->Problem() ;
	if (NULL == problem) 
		return ERRORCODE_generic ;
	if (nElimVars < 1) 
		// nothing to do; should not happen; however, allow this to pass as ok; calling fn should be able to handle this.
		return 0 ;
	if (_nFunctions > 0) {
		if (0 == _SortedSignature.size()) {
			if (0 != ComputeSignature()) 
				return ERRORCODE_generic ;
			}
		}
	else _SortedSignature.clear() ;
	ARE_Function_TableType & f_const_value = OutputFN.ConstValue() ;
	if (_nFunctions < 1) {
		if (FN_COBINATION_TYPE_PROD == problem->FnCombinationType()) {
			// we are just summing over variables, each sum counts as 1.
			f_const_value = 1.0 ;
			for (int32_t ii = 0 ; ii < nElimVars ; ++ii) 
				f_const_value *= problem->K(ElimVars[ii]) ;
			if (problem->FunctionsAreConvertedToLogScale()) 
				f_const_value = log10(f_const_value) ;
			}
		return 0 ;
		}

	std::unique_ptr<char[]> sBUF(new char[1024]) ;
	if (nullptr == sBUF) 
		return 1 ;
	char *buf = sBUF.get() ;

	// compute output function signature
	if (nElimVars < _SortedSignature.size()) {
		for (i = 0 ; i < _SortedSignature.size() ; i++) TempSpaceForVars[i] = _SortedSignature[i] ; j = _SortedSignature.size() ;
		ARE::SetMinus(TempSpaceForVars, j, ElimVars, nElimVars) ;
		if (0 != OutputFN.SetArguments(j, TempSpaceForVars)) 
			return ERRORCODE_generic ;
		}

	// set up output fn table
	OutputFN.ComputeTableSize() ;
	int32_t nA = OutputFN.N() ;
	if (nA > 0 && ! ResultToFile) { // if result goes to file, don't need table...
		if (0 != OutputFN.AllocateTableData()) 
			return ERRORCODE_generic ;
		}
	if (_SortedSignature.size() != nA + nElimVars) 
		return ERRORCODE_generic ;
	const int32_t *refFNarguments = OutputFN.Arguments() ;
	const int32_t *signature = Signature() ;

#ifdef _DEBUG
	if (_SortedSignature.size() != nA + nElimVars) {
		int32_t error = 1 ;
		}
#endif

	int32_t vars[MAX_NUM_VARIABLES_PER_BUCKET] ; // this is the list of variables : vars2Keep + vars2Eliminate
	for (j = 0 ; j < nA ; j++) {
		vars[j] = refFNarguments[j] ;
		}
	for (; j < _SortedSignature.size() ; j++) 
		vars[j] = ElimVars[j - nA] ;

	ARE_Function_TableType const_factor = bews->FnCombinationNeutralValue() ;
	int32_t nFNs = 0 ;
	ARE::Function *flist[MAX_NUM_FUNCTIONS_PER_BUCKET] ; // this is a list of input functions
	for (j = 0 ; j < _nFunctions ; j++) {
		ARE::Function *f = _Functions[j] ;
		if (NULL == f) return ERRORCODE_fn_ptr_is_NULL ;
		if (0 == f->N()) bews->ApplyFnCombinationOperator(const_factor, f->ConstValue()) ;
		else  { flist[nFNs++] = f ; f->ComputeArgumentsPermutationList(_SortedSignature.size(), vars) ; }
		}

	int64_t ElimIdx, ElimSize = 1 ;
	for (j = 0 ; j < nElimVars ; j++) 
		ElimSize *= problem->K(ElimVars[j]) ;

	ARE_Function_TableType *data = OutputFN.TableData(), vTableEntry, value ;
	double one_over_WMBEweight = WMBEweight < 1.0e+32 ? 1.0 / WMBEweight : DBL_MAX ;
	int64_t tablesize = 0 == nA ? 1 : OutputFN.TableSize(), adr ;
	std::unique_ptr<int32_t[]> currentValueSet(new int32_t[_SortedSignature.size()]) ;
	if (nullptr == currentValueSet) 
		return 1 ;
	for (i = 0 ; i < nA ; ++i) currentValueSet[i] = 0 ;

	// sometimes the result should be written to file...
	FILE *fp = nullptr ;
	std::string sFN, sFNnn, sFNsignalling, sPrefix, sPostFix, sSample ;
	if (ResultToFile) {
		int64_t nSamples = tablesize ;
		float samples_min_value = 1.0e+129, samples_max_value = 1.0e+129, samples_sum = 1.0e+129 ;
		GenerateSamplesXmlFilename("-full", sFN, sFNnn, sFNsignalling, sPrefix, sPostFix, nSamples, samples_min_value, samples_max_value, samples_sum) ;
		fp = fopen(sFN.c_str(), "w") ;
		if (nullptr != fp) 
			fwrite(sPrefix.c_str(), 1, sPrefix.length(), fp) ;
		}

	for (int64_t KeepIDX = 0; KeepIDX < tablesize; KeepIDX++) {
		ARE_Function_TableType & v = ResultToFile ? vTableEntry : (nullptr == data ? f_const_value : data[KeepIDX]) ;
		v = bews->VarEliminationDefaultValue() ;
		ElimIdx = -1 ;
do_next_elim_config:
		int32_t res = ManageEliminationConfiguration(nA, refFNarguments, nElimVars, ElimVars, ++ElimIdx, currentValueSet);
		if (res > 0)
			return res ;
		if (res < 0)
			goto done_with_this_elim_config ; // all elimination combinations have been enumerated...
		value = bews->FnCombinationNeutralValue() ;
		for (j = 0; j < nFNs; j++) {
			double fn_v = flist[j]->TableEntryEx(currentValueSet.get(), problem->K()) ; // NEW
//			adr = flist[j]->ComputeFnTableAdr_wrtLocalPermutation(values, problem->K()) ;
//			bews->ApplyFnCombinationOperator(value, flist[j]->TableEntry(adr)) ;
			bews->ApplyFnCombinationOperator(value, fn_v) ;
			}
		// if value would not change elim result (e.g. the problem is product-sum and value is 0 so far), continue.
		if (value == bews->VarEliminationDefaultValue())
			goto do_next_elim_config ;
		// apply MB weight, if given
		if (WMBEweight < 1.0e+32) {
			if (problem->FunctionsAreConvertedToLogScale())
				value /= WMBEweight ;
			else
				value = pow(value, one_over_WMBEweight) ;
			}
		ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), v, value) ;
		goto do_next_elim_config ;

done_with_this_elim_config :
		if (WMBEweight < 1.0e+32) {
			if (problem->FunctionsAreConvertedToLogScale())
				v *= WMBEweight ;
			else
				v = pow(v, WMBEweight) ;
			}
		bews->ApplyFnCombinationOperator(v, const_factor) ;
goto_next_keep_value_combination :
		// if needed, add to file
		if (ResultToFile && nullptr != fp) {
			sSample = "\n   <sample signature=\"" ;
			for (int32_t iSig = 0 ; iSig < nA ; ++iSig) {
				if (iSig > 0) sSample += ';' ;
				sprintf(buf, "%d", (int) currentValueSet[iSig]) ; sSample += buf ;
				}
			sprintf(buf, "\" value=\"%g\"/>", (double) v) ; sSample += buf ;
			fwrite(sSample.c_str(), 1, sSample.length(), fp) ;
			}
		// go to next argument value combination
		ARE::EnumerateNextArgumentsValueCombination(nA, vars, currentValueSet.get(), problem->K()) ;
		}

/* old code
	ARE_Function_TableType *data = OutputFN.TableData() ;
	int64_t tablesize = OutputFN.TableSize() ;
	double one_over_WMBEweight = WMBEweight < 1.0e+32 ? 1.0/WMBEweight : DBL_MAX ;
	for (int64_t KeepIDX = 0 ; KeepIDX < tablesize ; KeepIDX++) {
		for (j = nA ; j < _Width ; j++) 
			values[j] = 0 ;
		data[KeepIDX] = bews->VarEliminationDefaultValue() ;
		for (ElimIDX = 0 ; ElimIDX < ElimSize ; ElimIDX++) {
			ARE_Function_TableType value = bews->FnCombinationNeutralValue() ;
			for (j = 0 ; j < nFNs ; j++) {
				int64_t adr = flist[j]->ComputeFnTableAdr_wrtLocalPermutation(values, problem->K()) ;
				bews->ApplyFnCombinationOperator(value, flist[j]->TableEntry(adr)) ;
				}
			if (WMBEweight < 1.0e+32) {
				if (problem->FunctionsAreConvertedToLogScale()) 
					value /= WMBEweight ;
				else 
					value = pow(value, one_over_WMBEweight) ;
				}
			ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), data[KeepIDX], value) ;
			// go to next argument value combination
			ARE::EnumerateNextArgumentsValueCombination(_Width, vars, values, problem->K()) ;
			}
		bews->ApplyFnCombinationOperator(data[KeepIDX], const_factor) ;
		}
*/

	if (nullptr != fp) {
		fwrite(sPostFix.c_str(), 1, sPostFix.length(), fp) ;
		fclose(fp) ;
		}

	return 0 ;
}


int32_t BucketElimination::MiniBucket::NoteOutputFunctionComputationCompletion(void)
{
	int32_t i ;

	MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
	bool deleteMBfunctions = NULL != bews ? bews->DeleteUsedTables() : false ;

	if (deleteMBfunctions) {
		for (i = 0 ; i < _nFunctions ; i++) {
			ARE::Function *f = _Functions[i] ;
			if (NULL == f->OriginatingMiniBucket()) 
				continue ; // this is original fn, not MBE generated function.
			f->DestroyTableData() ;
			}
		}

	return 0 ;
}

