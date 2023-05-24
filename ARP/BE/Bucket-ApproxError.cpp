#include <stdlib.h>
#include <limits.h>
#include <float.h>
#include <memory.h>

#include <Function.hxx>
#include <Bucket.hxx>
#include <MiniBucket.hxx>
#include <MBEworkspace.hxx>
#include <MersenneTwister.h>

static MTRand RNG ;

int32_t BucketElimination::Bucket::Compute_ArroxError_Local(void)
{
	if (nullptr == _Workspace) return 1 ;
	ARE::ARP *P = _Workspace->Problem() ;
	if (nullptr == P) return 1 ;
    int32_t nMBs = _MiniBuckets.size() ;
	if (nMBs <= 0) return 0 ; // nothing computed yet, cannot do it
	if (_Width <= 0) return 0 ; // bucket has no variables
	int32_t nOutputVars = _Width - _Vars.size() ;
	// note it could be nOutputVars==0, when all variables of the bucket are eliminated; output fn is a constant...

	// 2021-08-08 : for now, assume _nVars==1
	if (1 != _Vars.size()) 
		return 2 ;
	int32_t k = P->K(_V) ;

	// collect signature of the output fn
	std::vector<int32_t> output_fn_signature ;
	// we allocate space for _Width not nOutputVars, since we need space for bucket vars too...
	output_fn_signature.reserve(_Width) ;
	int32_t *vars = output_fn_signature.data() ;
	if (nullptr == vars) 
		return 1 ;
	int32_t nOutputVars_ = 0 ;
	for (int32_t i = 0 ; i < _Width ; ++i) {
		int32_t v = _Signature[i], j ;
		for (j = 0 ; j < _Vars.size() ; ++j) {
			if (_Vars[j] == v) break ; }
		if (j >= _Vars.size()) 
			vars[nOutputVars_++] = v ;
		}
	if (nOutputVars != nOutputVars_) 
		return 1 ; // something is wrong; signature incorrect...
	// if output_fn_signature[] (or vars[]), the first nOutputVars elements are output fn signature; after that bucket vars
	for (int32_t i = 0 ; i < _Vars.size() ; ++i) 
		vars[nOutputVars_++] = _Vars[i] ;

	// random assignment (of output fn signature) stored here
	std::vector<int32_t> assignment ;
	assignment.reserve(P->N()) ; // note size is P->N()! value of variable v is at vals[v]
	int32_t *vals = assignment.data() ;
	if (nullptr == vals) 
		return 1 ;

	// collect all original/augmented functions of the bucket; they define the output fn of the bucket...
	int32_t nFNs = _nOriginalFunctions + _nAugmentedFunctions ;
	std::vector<ARE::Function*> bucket_functions ;
	bucket_functions.reserve(nFNs) ;
	if (nullptr == bucket_functions.data()) 
		return 1 ;
	for (int32_t i = 0 ; i < _nOriginalFunctions ; i++) {
		if (nullptr != _OriginalFunctions[i]) bucket_functions.push_back(_OriginalFunctions[i]) ; }
	for (int32_t i = 0 ; i < _nAugmentedFunctions ; i++) {
		if (nullptr != _AugmentedFunctions[i]) bucket_functions.push_back(_AugmentedFunctions[i]) ; }
	nFNs = bucket_functions.size() ;

	// compute bucket output fn table size
	int64_t output_fn_table_size = 0 ;
	if (nOutputVars > 0) {
		output_fn_table_size = 1 ;
		for (int32_t i = 0 ; i < nOutputVars ; ++i) {
			int32_t v = vars[i] ;
			output_fn_table_size *= P->K(v) ;
			}
		}

	int32_t nSamples = 1000 ; // set this to desired/appropriate number...
	for (int32_t i = 0 ; i < nSamples ; ++i) {
		// generate random assignment to output fn signature...
        for (int32_t j = 0 ; j < nOutputVars ; ++j) {
            int32_t v = output_fn_signature[j] ;
            int32_t domain_size_of_v = P->K(v) ;
            int32_t value = RNG.randInt(domain_size_of_v-1) ;
            vals[v] = value ;
	        }

		// ****************************************************
		// compute BE output fn value for the assignment
		// ****************************************************
        // enumerate all current variable values; compute bucket value for each configuration and combine them using elimination operator...
        ARE_Function_TableType V = _Workspace->VarEliminationDefaultValue() ;
        for (int32_t j = 0 ; j < k ; ++j) {
            vals[_V] = j ;
            ARE_Function_TableType v = _Workspace->FnCombinationNeutralValue() ;
            // compute value for this configuration : fNN argument assignment + _V=j
            for (int32_t l = 0 ; l < nFNs ; ++l) {
                ARE::Function *f = bucket_functions[l] ;
                double fn_v = f->TableEntryExNativeAssignment(vals, P->K()) ;
                _Workspace->ApplyFnCombinationOperator(v, fn_v) ;
				}
            _Workspace->ApplyVarEliminationOperator(V, v) ;
			}
		double be_value = V ;

		// ****************************************************
		// compute MBE output fn value for the assignment
		// ****************************************************
		double mbe_value = _Workspace->FnCombinationNeutralValue() ;
        for (int32_t j = 0 ; j < nMBs ; ++j) {
			BucketElimination::MiniBucket *mb = _MiniBuckets[j] ;
			if (nullptr == mb) continue ;
			ARE::Function & fn = mb->OutputFunction() ;
            double fn_v = fn.TableEntryExNativeAssignment(vals, P->K()) ;
            _Workspace->ApplyFnCombinationOperator(mbe_value, fn_v) ;
			}

		// ****************************************************
		// DONE! difference between be_value and mbe_value is the approx value for this assignment...
		// ****************************************************

		// TODO : process the approx value....
		}

	return 0 ;
}

int32_t BucketElimination::Bucket::Compute_ArroxError_Cmltv(void)
{
	// TODO : implement

	return 0 ;
}

