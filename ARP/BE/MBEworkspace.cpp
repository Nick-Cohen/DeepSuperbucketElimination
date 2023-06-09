#include <stdlib.h>
#include "time.h"
#include "math.h"

#if defined WINDOWS || _WINDOWS
#include "process.h"    /* _beginthread, _endthread */
#endif // WINDOWS

//#define SAMPLE_BUCKET_TEST

#include "Utils/Sort.hxx"

#include "Globals.hxx"
#include "Utils/MiscUtils.hxx"
#include "Utils/ThreadPool.hxx"
#include "Function.hxx"
#include "Bucket.hxx"
#include "MiniBucket.hxx"
#include "MBEworkspace.hxx"

#ifdef LINUX
pthread_mutex_t BucketElimination::MBEworkspace::stopSignalMutex = PTHREAD_MUTEX_INITIALIZER ;
#endif

#if defined WINDOWS || _WINDOWS
typedef unsigned int (*pBEWSThreadFn)(void *X) ;
static unsigned int __stdcall BEWSThreadFn(void *X) 
#elif defined (LINUX)
typedef void *(*pBEWSThreadFn)(void *X) ;
static void *BEWSThreadFn(void *X)
#endif 
{
	BucketElimination::MBEworkspace *bews = (BucketElimination::MBEworkspace *)(X) ;

	bews->SetCompleteEliminationResult(DBL_MAX, 0) ;
	bews->MarginalSingleVariableDistribution().clear() ;
	bews->MarginalSingleVariableDistributionVar() = -1 ;

	int32_t stop_signalled = 0 ;
	int32_t n = bews->lenComputationOrder() - 1 ;
	signed char approx_bound = 1 ; // 1=max
	bews->tStart() = ARE::GetTimeInMilliseconds() ;
	while (bews->IsValid()) {
#if defined WINDOWS || _WINDOWS
		stop_signalled = InterlockedCompareExchange(&(bews->_StopAndExit), 1, 1) ;
#else
    pthread_mutex_lock(&BucketElimination::MBEworkspace::stopSignalMutex);
    stop_signalled = bews->_StopAndExit;
    pthread_mutex_unlock(&BucketElimination::MBEworkspace::stopSignalMutex);
#endif
		if (0 != stop_signalled) {
			break ;
			}
		int32_t idxToCompute = bews->BucketOrderToCompute(n) ;
		if (idxToCompute < 0) {
			if (NULL != bews->logFile()) {
				fprintf(bews->logFile(), "\n   BE elimination nBucketsComputed=%d, WARNING : idxToCompute<0", (int32_t) n) ;
				}
			continue ;
			}
		BucketElimination::Bucket *b = bews->getBucket(idxToCompute) ;
		if (NULL == b) {
			if (NULL != bews->logFile()) {
				fprintf(bews->logFile(), "\n   BE elimination nBucketsComputed=%d, WARNING : bucket==NULL", (int32_t) n) ;
				}
			continue ;
			}
		int64_t totalSumOutputFunctionsNumEntries = 0 ;
		int32_t res = b->ComputeOutputFunctions(true, approx_bound, totalSumOutputFunctionsNumEntries) ;
		bool do_break = false ;
/*		if (NULL != bews->logFile()) {
			ARE::Function & f = b->OutputFunction() ;
			if (0 == f.N()) 
				fprintf(bews->logFile(), "\n   BE elimination var=%d, const-output=%g", b->Var(0), f.ConstValue()) ;
			else if (1 == f.N()) 
				fprintf(bews->logFile(), "\n   BE elimination var=%d, output=%g,%g", b->Var(0), f.Table()->Data()[0], f.Table()->Data()[1]) ;
			else if (2 == f.N()) 
				fprintf(bews->logFile(), "\n   BE elimination var=%d, output=%g,%g,%g,%g", b->Var(0), f.Table()->Data()[0], f.Table()->Data()[1], f.Table()->Data()[2], f.Table()->Data()[3]) ;
			}*/
		if (0 != res) {
			// failed; abandon.
			if (NULL != bews->logFile()) {
				fprintf(bews->logFile(), "\n   MBE elimination n=%d v=%d, ERROR : ComputeOutputFunctions() returned error=%d; will quit ...", (int32_t) n, (int32_t) b->Var(0), (int32_t) res) ;
				}
			do_break = true ; goto done_with_this_b ;
			}
		for (BucketElimination::MiniBucket *mb : b->MiniBuckets()) {
			ARE::Function & output_fn = mb->OutputFunction() ;
			double table_size_log10 = output_fn.GetTableSize_Log10() ;
			if (bews->TotalNewFunctionSizeComputed_Log10() < -1.0) 
				bews->TotalNewFunctionSizeComputed_Log10() = table_size_log10 ;
			else 
				bews->TotalNewFunctionSizeComputed_Log10() += log10(1.0 + pow(10.0, table_size_log10 - bews->TotalNewFunctionSizeComputed_Log10())) ;
			}
		if (--n < 0) {// all computed
			bews->PostComputationProcessing() ;
			if (bews->Problem()->IsOptimizationProblem()) {
				bews->BuildSolution() ;
				}
			do_break = true ;
			}
done_with_this_b :
		b->NoteOutputFunctionComputationCompletion() ;
		if (do_break) 
			break ;
		}

done :
	bews->tEnd() = ARE::GetTimeInMilliseconds() ;
	bews->RunTimeInMilliseconds() = bews->tEnd() - bews->tStart() ;
	bews->_ThreadHandle = 0 ;
#if defined WINDOWS || _WINDOWS
	_endthreadex(0) ;
	return 0  ;
#else
	return NULL ;
#endif
}


BucketElimination::MBEworkspace::MBEworkspace(const char *BEEMDiskSpaceDirectory)
	:
	ARE::Workspace(BEEMDiskSpaceDirectory), 
	_VarList(NULL), 
	_VarPos(NULL), 
	_Var2BucketMapping(NULL), 
//	_BTchildlistStorage(NULL), 
	_DeleteUsedTables(false), 
	_EClimit(-1), 
	_iBound(1000000), 
	_VarOrdering_MaxCliqueSize(-1), 
	_PseudoWidth(-1), 
	_nBucketsWithPartitioning(-1), 
	_maxDepthPartitionedBucket(-1), 
	_MaxNumMiniBucketsPerBucket(-1), 
	_MaxTreeHeight_BranchingVars_Limit(INT_MAX), 
	_CurrentComputationBound(0), 
	_MaxSpaceAllowed_Log10(_I64_MAX), 
	_fpLOG(NULL), 
	_MBoutputFnTypeWhenOverIBound(MB_outputfn_type_table), 
	_AnswerFactor(1.0), 
	_CompleteEliminationResult(DBL_MAX), 
	_CompleteEliminationResult_ub(DBL_MAX), 
	_CompleteEliminationResult_lb(DBL_MAX), 
	_tStart(0), 
	_tEnd(0), 
	_tToStop(0), 
	_RunTimeInMilliseconds(-1), 
	_StopAndExit(0), 
	_ThreadHandle(0), 
	_FnCombinationType(FN_COBINATION_TYPE_NONE), 
	_VarEliminationType(VAR_ELIMINATION_TYPE_NONE), 
	_nBuckets(0), 
	_Buckets(NULL), 
	_MaxNumChildren(-1), 
	_MaxNumVarsElimInBucket(-1), 
	_nLeafBucketsWithNoFunctions(-1), 
	_MaxNumVarsInBucket(-1),
	_MaxElimComplexityInBucket(-1), 
	_MaxTreeHeight(-1), 
	_MaxTreeHeight_BranchingVars(-1), 
	_nBucketsWithSingleChild_initial(-1), 
	_nBucketsWithSingleChild_final(-1), 
	_nBuckets_initial(-1), 
	_nBuckets_final(-1), 
	_nVarsWithoutBucket(-1), 
	_nConstValueFunctions(-1), 
	_MaxBucketFunctionWidth(-1), 
	_nBucketsWithSingleChild(-1), 
	_nBucketsWithNoChildren(-1), 
	_nRoots(-1), 
	_nSBmerges(-1), 
	_nOriginalFunctions(-1), 
	_nAugmentedFunctions(-1), 
	_ForORtreetype_use_DFS_order(false),
	_nEvidenceVars(-1), 
	_TotalOriginalFunctionSize(-1), 
	_TotalOriginalFunctionSpace(-1), 
	_TotalNewFunctionSize_Log10(-1.0), 
	_TotalNewFunctionSpace_Log10(-1.0), 
	_TotalNewFunctionComputationComplexity_Log10(-1.0), 
	_MaxSimultaneousNewFunctionSize_Log10(-1.0), 
	_MaxSimultaneousNewFunctionSpace_Log10(-1.0), 
	_MaxSimultaneousTotalFunctionSize_Log10(-1.0), 
	_MaxSimultaneousTotalFunctionSpace_Log10(-1.0), 
	_TotalNewFunctionSizeComputed_Log10(-1.0), 
	_lenComputationOrder(0), 
	_BucketOrderToCompute(NULL) 
{
	if (! _IsValid) 
		return ;

	_MarginalSingleVariableDistribution.clear() ;
	_MarginalSingleVariableDistributionVar = -1 ;

	_IsValid = true ;
}


int32_t BucketElimination::MBEworkspace::Destroy(void)
{
	StopThread() ;

	DestroyBucketPartitioning() ;

	if (NULL != _VarList) {
		delete [] _VarList ;
		_VarList = NULL ;
		}
	if (NULL != _VarPos) {
		delete [] _VarPos ;
		_VarPos = NULL ;
		}
	if (NULL != _Var2BucketMapping) {
		delete [] _Var2BucketMapping ;
		_Var2BucketMapping = NULL ;
		}
/*	if (NULL != _BTchildlistStorage) {
		delete [] _BTchildlistStorage ;
		_BTchildlistStorage = NULL ; 
		}*/
	if (NULL != _BucketOrderToCompute) {
		delete [] _BucketOrderToCompute ;
		_BucketOrderToCompute = NULL ;
		}
	_lenComputationOrder = 0 ;
	_nVars = 0 ;
	if (NULL != _Buckets) {
		for (int32_t i = 0 ; i < _nBuckets ; i++) {
			if (NULL != _Buckets[i]) {
				delete _Buckets[i] ;
				_Buckets[i] = NULL ;
				}
			}
		delete [] _Buckets ;
		_Buckets = NULL ;
		}
	_nBuckets = 0 ;
	_nVarsWithoutBucket = -1 ;

	Workspace::Destroy() ;

	_iBound = 1000000 ;
	_VarOrdering_MaxCliqueSize = -1 ;
	_PseudoWidth = -1 ;
	_nBucketsWithPartitioning = -1 ;
	_maxDepthPartitionedBucket = -1 ;
	_MaxNumMiniBucketsPerBucket = -1 ;
	_MaxSpaceAllowed_Log10 = _I64_MAX ;

	_tStart = _tEnd = _tToStop = 0 ;
	_RunTimeInMilliseconds = -1 ;

	_CompleteEliminationResult = DBL_MAX ;
	_CompleteEliminationResult_ub = DBL_MAX ;
	_CompleteEliminationResult_lb = DBL_MAX ;

	_MarginalSingleVariableDistribution.clear() ;
	_MarginalSingleVariableDistributionVar = -1 ;

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::DestroyBucketPartitioning(void)
{
	if (NULL == _Buckets || _nVars <= 0) 
		return 0 ;
	for (int32_t i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (NULL == b) continue ;
		b->DestroyPartitioning() ;
		}
	return 0 ;
}


int32_t BucketElimination::MBEworkspace::CreateThread(void)
{
#if defined WINDOWS || _WINDOWS
	_ThreadHandle = _beginthreadex(NULL, 0, BEWSThreadFn, this, 0, NULL) ;
#else
	pthread_create(&_ThreadHandle, NULL, BEWSThreadFn, this) ; // TODO third argument
#endif
	return 0 != _ThreadHandle ? 0 : 1 ;
}


int32_t BucketElimination::MBEworkspace::StopThread(void)
{
	if (0 == _ThreadHandle) {
		if (NULL != ARE::fpLOG) {
			int64_t tNowLog = ARE::GetTimeInMilliseconds() ;
			fprintf(ARE::fpLOG, "\n%lld   BEWS_th : stop computation; already stopped ...", tNowLog) ;
			fflush(ARE::fpLOG) ;
			}
		return 0 ;
		}
#if defined WINDOWS || _WINDOWS
	InterlockedCompareExchange(&_StopAndExit, 1, 0) ;
#else
	pthread_mutex_lock(&BucketElimination::MBEworkspace::stopSignalMutex);
	if (_StopAndExit = 0) {
		_StopAndExit = 1;
		}
	pthread_mutex_unlock(&BucketElimination::MBEworkspace::stopSignalMutex);
#endif
	_tToStop = ARE::GetTimeInMilliseconds() ;
	if (NULL != ARE::fpLOG) {
		fprintf(ARE::fpLOG, "\n%lld   BEWS_th : stop variable order computation; stop signalled, will wait ...", _tToStop) ;
		fflush(ARE::fpLOG) ;
		}
	while (true) {
		SLEEP(50) ;
		if (0 == _ThreadHandle) 
			break ;
		int64_t tNow = ARE::GetTimeInMilliseconds() ;
		int64_t dt = tNow - _tToStop ;
		if (dt > 10000) {
			// we asked the thread to stop and waited for it to stop, but it won't stop, so kill the thread.
#if defined WINDOWS || _WINDOWS
			TerminateThread((HANDLE) _ThreadHandle, 0) ;
			CloseHandle((HANDLE) _ThreadHandle) ;
			_ThreadHandle = 0 ;
#else
			// TODO : handle linux
#endif
			if (NULL != ARE::fpLOG) {
				int64_t tNowLog = ARE::GetTimeInMilliseconds() ;
				fprintf(ARE::fpLOG, "\n%lld   BEWS_th : stop variable order computation, hard kill ...", tNowLog) ;
				fflush(ARE::fpLOG) ;
				}
			break ;
			}
		}
	return 0 ;
}


int32_t BucketElimination::MBEworkspace::Initialize(ARE::ARP & Problem, bool UseLogScale, const int32_t *VarOrderingAsVarList, int32_t DeleteUsedTables)
{
	int32_t i ;

	Destroy() ;

	if (NULL != ARE::fpLOG) {
		int64_t tNOW = ARE::GetTimeInMilliseconds() ;
		fprintf(ARE::fpLOG, "\n%lld      BEWS : Initialize; UseLogScale=%c ...", tNOW, UseLogScale ? 'Y' : 'N') ;
		fflush(ARE::fpLOG) ;
		}

	if (NULL == VarOrderingAsVarList && Problem.N() > 0) {
		VarOrderingAsVarList = Problem.VarOrdering_VarList() ;
		}

	if (! _IsValid || 0 != ARE::Workspace::Initialize(Problem)) 
		{ _IsValid = false ; return 1 ; }
	if (NULL == _Problem) 
		{ _IsValid = false ; return 2 ; }

	if (UseLogScale && ! Problem.FunctionsAreConvertedToLogScale()) {
		Problem.ConvertFunctionsToLogScale() ;
		}

	_FnCombinationType = Problem.FnCombinationType() ;
	_VarEliminationType = Problem.VarEliminationType() ;

	_AnswerFactor = FnCombinationNeutralValue() ;

	if (0 == DeleteUsedTables) 
		SetDeleteUsedTables(false) ;
	else if (DeleteUsedTables > 0) 
		SetDeleteUsedTables(true) ;

	_nVars = _Problem->N() ;
	if (_nVars <= 0) 
		return 0 ;

	// allocate space; initialize

	_VarList = new int32_t[_nVars] ;
	_VarPos = new int32_t[_nVars] ;
	_Var2BucketMapping = new BucketElimination::Bucket*[_nVars] ;
	_BucketOrderToCompute = new int32_t[_nVars] ;
	if (NULL == _VarList || NULL == _VarPos || NULL == _Var2BucketMapping || NULL == _BucketOrderToCompute) {
		if (NULL != ARE::fpLOG) {
			int64_t tNOW = ARE::GetTimeInMilliseconds() ;
			fprintf(ARE::fpLOG, "\n%lld      BEWS : Initialize; allocating basic memory failed ...", tNOW) ;
			fflush(ARE::fpLOG) ;
			}
		Destroy() ;
		_IsValid = false ;
		return 1 ;
		}
	_lenComputationOrder = 0 ;
	_VarOrdering_MaxCliqueSize = Problem.VarOrdering_InducedWidth() ;
	if (_VarOrdering_MaxCliqueSize >= 0) _VarOrdering_MaxCliqueSize++ ;
	_PseudoWidth = -1 ;
	for (i = 0 ; i < _nVars ; i++) 
		_VarList[i] = _VarPos[i] = _BucketOrderToCompute[i] = -1 ;
	if (NULL != VarOrderingAsVarList) {
		for (i = 0 ; i < _nVars ; i++) {
			_VarList[i] = VarOrderingAsVarList[i] ;
			_VarPos[_VarList[i]] = i ;
			_Var2BucketMapping[i] = NULL ;
			}
		}
	for (i = 0 ; i < _nVars ; i++) {
		if (_VarList[i] < 0 || _VarList[i] >= _nVars) {
			Destroy() ;
			_IsValid = false ;
			return 1 ;
			}
		if (_VarPos[i] < 0 || _VarPos[i] >= _nVars) {
			Destroy() ;
			_IsValid = false ;
			return 1 ;
			}
		}

	ComputeTotalOriginalFunctionSizeAndSpace() ;

/*
	// create buckets
	if (0 != CreateBuckets(false)) {
		_IsValid = false ;
		return 1 ;
		}

	// create computation order
	CreateComputationOrder(1) ;

#ifdef _DEBUG
	// verify integrity of functions
	for (i = 0 ; i < _Problem->nFunctions() ; i++) {
		ARE::Function *f = _Problem->getFunction(i) ;
		if (0 != f->CheckIntegrity()) 
			{ _IsValid = false ; return 1 ; }
		}
	for (i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		ARE::Function & f = b->OutputFunction() ;
		if (0 != f.CheckIntegrity()) 
			{ _IsValid = false ; return 1 ; }
		}
#endif // _DEBUG
*/

	_TotalNewFunctionSizeComputed_Log10 = -1.0 ;

	if (NULL != ARE::fpLOG) {
		int64_t tNOW = ARE::GetTimeInMilliseconds() ;
		fprintf(ARE::fpLOG, "\n%lld      BEWS : Initialize; done ...", tNOW) ;
		fflush(ARE::fpLOG) ;
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::CreateBuckets(bool ANDORtree, bool KeepBTsignature, bool SimplifyBTstructure)
{
	if (_nVars <= 0) 
		return 0 ;

//	if (NULL != _BTchildlistStorage) 
//		{ delete [] _BTchildlistStorage ; _BTchildlistStorage = NULL ; }

	// when creating bt functions, we need temp space to store signatures of functions
	int32_t *TempSpaceForArglist = NULL, TempSpaceForArglistSize = 0 ;

	if (NULL != ARE::fpLOG) {
		int64_t tNOW = ARE::GetTimeInMilliseconds() ;
		fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets() ...", tNOW) ;
		fflush(ARE::fpLOG) ;
		}

	int32_t i, j, n, ret = 1 ;

	// ***************************************************************************************************
	// create initial bucket tree structure
	// ***************************************************************************************************

	_Buckets = new BucketElimination::Bucket*[_nVars] ;
	if (NULL == _Buckets) {
		Destroy() ;
		return 1 ;
		}
// DEBUGGGG
/*if (NULL != ARE::fpLOG) {
	int64_t tNOW = ARE::GetTimeInMilliseconds() ;
	fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets _nVars=%d", tNOW, (int32_t) _nVars) ;
	fflush(ARE::fpLOG) ;
	}*/
	memset(_Buckets, 0, _nVars*sizeof(BucketElimination::Bucket*)) ;
	for (i = 0 ; i < _nVars ; i++) {
		int32_t v = _VarList[i] ;
		BucketElimination::Bucket *b = _Buckets[i] = new BucketElimination::Bucket(*this, i, v) ;
// DEBUGGGG
//fprintf(ARE::fpLOG, "\n				BEWS : CreateBuckets v=%d order=%d", (int32_t) i, (int32_t) _VarList[i]) ;
		if (NULL == b) {
			Destroy() ;
			return 1 ;
			}
		if (1 != b->nVars()) {
			Destroy() ;
			return 1 ;
			}
		_Var2BucketMapping[v] = b ;
		}
	_nBuckets = _nVars ;

	// ***************************************************************************************************
	// create initial function assignment
	// ***************************************************************************************************

	int32_t NF = _Problem->nFunctions() ;
// DEBUGGGG
/*if (NULL != ARE::fpLOG) {
	int64_t tNOW = ARE::GetTimeInMilliseconds() ;
	fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets NF=%d", tNOW, (int32_t) NF) ;
	fflush(ARE::fpLOG) ;
	}*/
	_nConstValueFunctions = 0 ;
	for (i = 0 ; i < NF ; i++) {
		ARE::Function *f = _Problem->getFunction(i) ;
		if (NULL == f) {
			if (NULL != ARE::fpLOG) {
				int64_t tNOW = ARE::GetTimeInMilliseconds() ;
				fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets function %d (of %d) is NULL ...", tNOW, (int32_t) i, (int32_t) NF) ;
				fflush(ARE::fpLOG) ;
				}
			continue ;
			}
		f->SetBucket(NULL) ;
		f->SetOriginatingBucket(NULL) ;
		f->SetOriginatingMiniBucket(NULL) ;
		// collect all const-functions; add as factor to the workspace
		if (0 == f->N()) {
			AddAnswerFactor(f->ConstValue()) ;
			++_nConstValueFunctions ;
			}
		}

	ARE::Function **fl = NULL ;
	if (NF > 0) {
		fl = new ARE::Function*[2*NF] ;
		if (NULL == fl) {
			Destroy() ;
			return 1 ;
			}
		}
	// for debugging purposes, we want to check which functions get assigned to a bucket
	ARE::Function **fl_assigned = fl + NF ;

	// process buckets, from last to first
	int32_t nfPlaced = 0 ;
	int32_t nfQirrelevant  = 0 ;
	for (i = 0 ; i < NF ; i++) {
		ARE::Function *f = _Problem->getFunction(i) ;
		if (0 == f->N()) {
			fl_assigned[i] = NULL ;
			}
		else if (f->IsQueryIrrelevant()) {
			++nfQirrelevant ;
			fl_assigned[i] = NULL ;
			}
		else 
			fl_assigned[i] = f ;
		}
	for (i = _nVars - 1 ; i >= 0 ; i--) {
		int32_t v = _VarList[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL == b) 
			continue ;
		// find functions whose highest indexed variable is variable of the bucket (b->Var(0))
		n = 0 ;
		for (j = 0 ; j < _Problem->nAdjFunctions(v) ; j++) {
			ARE::Function *f = _Problem->AdjFunction(v, j) ;
			if (NULL == f) 
				continue ;
			int32_t idxFN = f->IDX() ;
			if (NULL == fl_assigned[idxFN]) 
				continue ; // normally this should not happen; this means this fn is already assigned.
			if (f->IsQueryIrrelevant()) 
				continue ;
			if (NULL != f->Bucket()) 
				// this means f was placed in a later bucket; i.e. v is not the highest-ordered variable in f.
				continue ;
			fl[n++] = f ;
			}
		nfPlaced += n ;
		b->SetOriginalFunctions(n, fl) ;
		// mark off the assigned functions
		for (j = 0 ; j < n ; j++) {
			ARE::Function *f = fl[j] ;
			int32_t idxFN = f->IDX() ;
			if (idxFN >= 0 && idxFN < NF) {
				if (NULL == fl_assigned[idxFN]) {
					int32_t error_FN_assigned_more_than_once = 1 ;
					}
				else 
					fl_assigned[idxFN] = NULL ;
				}
			}
		}

	// DEBUGGGG
/*	if (NULL != ARE::fpLOG) {
		int64_t tNOW = ARE::GetTimeInMilliseconds() ;
		fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets _FnCombinationType=%d _VarElimType=%d AnswerFactor=%g", tNOW, (int32_t) _FnCombinationType, (int32_t) _VarEliminationType, (double) _AnswerFactor) ;
		fflush(ARE::fpLOG) ;
		}
*/

	// test all functions are processed
	if ((nfPlaced + nfQirrelevant + _nConstValueFunctions) != NF) {
		for (i = 0 ; i < NF ; i++) {
			ARE::Function *f = _Problem->getFunction(i) ;
			int32_t break_point_here = 1 ;
			}
		goto failed ;
		}

// DEBUGGGGG
/*if (NULL != ARE::fpLOG) {
	int64_t tNOW = ARE::GetTimeInMilliseconds() ;
	fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets 6 _nBuckets=%d ...", tNOW, _nBuckets) ;
	fflush(ARE::fpLOG) ;
	}*/

	// ***************************************************************************************************
	// create a bucket-function for each bucket, from last to first (bottom up on the bucket tree).
	// then compute children of each bucket using this bucket-tree.
	// ***************************************************************************************************

//	_BTchildlistStorage = new int32_t[_nBuckets] ;
//	int32_t nBTchildlistStorageUsed ;
//	if (NULL == _BTchildlistStorage) 
//		goto failed ;
	if (_VarOrdering_MaxCliqueSize > 0) {
		TempSpaceForArglist = new int32_t[_VarOrdering_MaxCliqueSize] ;
		if (NULL == TempSpaceForArglist) 
			goto failed ;
		TempSpaceForArglistSize = _VarOrdering_MaxCliqueSize ;
		}
	// process along the order, from last to first; generate bucket output functions and place in appropriate buckets.
	for (i = _nVars - 1 ; i >= 0 ; i--) {
		int32_t v = _VarList[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL == b) continue ;
		ARE::Function *fn = NULL ;
		int32_t fnMaxVar = -1 ;
		int32_t res = b->ComputeOutputFunctionWithScopeWithoutTable(TempSpaceForArglist, TempSpaceForArglistSize, fn, fnMaxVar) ;
		if (0 != res) {
			if (NULL != ARE::fpLOG) {
				int64_t tNOW = ARE::GetTimeInMilliseconds() ;
				fprintf(ARE::fpLOG, "\n%lld      BEWS : ComputeOutputFunctionWithScopeWithoutTable failed; var=%d ...", tNOW, (int32_t) b->Var(0)) ;
				::fflush(ARE::fpLOG) ;
				}
			if (NULL != fn) delete fn ;
			goto failed ;
			}
		if (NULL != fn ? fn->N() > 0 : false) {
			if (fnMaxVar < 0 || fnMaxVar >= _nBuckets) 
				{ delete fn ; goto failed ; }
			// add to target bucket
			BucketElimination::Bucket *b = _Var2BucketMapping[fnMaxVar] ;
			if (NULL == b) 
				{ delete fn ; goto failed ; }
			if (0 != b->AddAugmentedFunction(*fn)) 
				{ delete fn ; goto failed ; }
			}
		}
	if (NULL != TempSpaceForArglist) 
		{ delete [] TempSpaceForArglist ; TempSpaceForArglist = NULL ; TempSpaceForArglistSize = 0 ; }
	// create bucket tree by connecting parents/children
	{
	std::vector<int32_t> temp_children_storage ; temp_children_storage.reserve(1000) ;
	// compute children for each bucket, by taking the largestidx var of all augmented fns of each bucket
//	nBTchildlistStorageUsed = 0 ;
	for (i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (NULL == b) 
			continue ;
//		int32_t nchildren = 0 ;
		temp_children_storage.clear() ;
		for (j = 0 ; j < b->nAugmentedFunctions() ; j++) {
			ARE::Function *fn = b->AugmentedFunction(j) ;
			if (NULL == fn) continue ;
			int32_t childvar = -fn->IDX() - 1 ;
			if (childvar < 0 || childvar >= _nVars) 
				goto failed ;
			BucketElimination::Bucket *cb_ = _Var2BucketMapping[childvar] ;
			BucketElimination::Bucket *cb = fn->OriginatingBucket() ;
			if (cb_ != cb) 
				goto failed ;
			temp_children_storage.push_back(childvar) ;
//			_BTchildlistStorage[nBTchildlistStorageUsed + nchildren++] = childvar ;
			cb->SetParentBucket(b) ;
			}
		if (0 != b->SetChildren(temp_children_storage.size(), temp_children_storage.data())) 
			goto failed ;
//		nBTchildlistStorageUsed += nchildren ;
		}

	// eliminate leaf buckets that have no functions in them; note - the parent bucket should be NULL for them.
	// HERE WE ASSUME _Buckets[] and _VarList[] are consistent.
	_nVarsWithoutBucket = 0 ;
	_nLeafBucketsWithNoFunctions = 0 ;
	if (SimplifyBTstructure) {
		for (i = _Problem->N()-1 ; i >= 0 ; i--) {
			int32_t v = _VarList[i] ;
			BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
			if (nullptr == b) continue ;
			if (0 == b->nOriginalFunctions() && 0 == b->nChildren()) {
				if (nullptr != b->ParentBucket()) {
					int32_t q_is_this_error = 1 ;
					// this is not necessarily an error; if AND-OR tree, this should not happen.
					// but when an OR tree, this could happen, since we create a chain of buckets, 
					// regardless of what the functions are.
					// also, this var could have originally had functions that it participated in, 
					// but perhaps its domain size was 1 and hence it was removed from all its functions.
					b->ParentBucket()->RemoveChild(b->V()) ;
					b->SetParentBucket(nullptr) ;
					}
				for (j = 0 ; j < b->nVars() ; j++) {
					_Var2BucketMapping[b->Var(j)] = nullptr ;
					++_nVarsWithoutBucket ;
					}
				_Buckets[b->IDX()] = nullptr ;
				delete b ;
				++_nLeafBucketsWithNoFunctions ;
				}
			}
		// compress the list of buckets
		TidyBucketsArray() ;
		}

	// construct BT vriable partial order
	_BTpartialorder.SetReallocationSize(_nBuckets*_nBuckets) ;
	_BTpartialorder.Empty() ;
	for (i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (NULL == b) continue ;
		int32_t vC = b->V() ;
		for (BucketElimination::Bucket *b_ = b->ParentBucket() ; NULL != b_ ; b_ = b_->ParentBucket()) {
			int32_t vA = b_->V() ;
			int64_t key = vA ; key <<= 32 ; key += vC ;
			_BTpartialorder.Insert(key, NULL) ;
			}
		}

	// if OR chain required, do it here
	if (! ANDORtree) {
		if (_ForORtreetype_use_DFS_order) {
			// need OR tree; do DFS and DFS traversal chain
			std::vector<BucketElimination::Bucket*> dfs_reverse_chain ; dfs_reverse_chain.reserve(_nBuckets) ;
			std::vector<BucketElimination::Bucket*> dfs_stack_b ; dfs_stack_b.reserve(_nBuckets) ;
			std::vector<int32_t> dfs_stack_i ; dfs_stack_i.reserve(_nBuckets) ;
			std::vector<BucketElimination::Bucket*> roots ;
			// collect roots
			for (i = 0 ; i < _nBuckets ; i++) 
				{ BucketElimination::Bucket *b = _Buckets[i] ; if (NULL == b) continue ; if (NULL == b->ParentBucket()) roots.push_back(b) ; }
			// generate DFS traversal of each tree
			for (i = 0 ; i < roots.size() ; ++i) {
				int32_t res = roots[i]->GetDFSorderofsubtree(dfs_reverse_chain, dfs_stack_b, dfs_stack_i) ;
				}
			// generate OR chain
			for (i = 1 ; i < dfs_reverse_chain.size() ; ++i) {
				BucketElimination::Bucket *B = dfs_reverse_chain[i] ;
				BucketElimination::Bucket *b = dfs_reverse_chain[i-1] ;
				int32_t v = b->V() ;
				b->SetParentBucket(B) ;
				B->SetChildren(1, &v) ;
				}
			dfs_reverse_chain[0]->SetChildren(0, NULL) ;
			dfs_reverse_chain[dfs_reverse_chain.size()-1]->SetParentBucket(NULL) ;
			}
		else {
			// create a chain of buckets
			BucketElimination::Bucket *bLast = NULL ;
	//		nBTchildlistStorageUsed = 0 ;
			for (i = _nVars - 1 ; i >= 0 ; i--) {
				int32_t v = _VarList[i] ;
				BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
				if (NULL == b) continue ;
				temp_children_storage.clear() ;
				if (NULL != bLast) {
	//				_BTchildlistStorage[nBTchildlistStorageUsed] = bLast->V() ;
					temp_children_storage.push_back(bLast->V()) ;
					if (0 != b->SetChildren(temp_children_storage.size(), temp_children_storage.data())) 
						goto failed ;
	//				++nBTchildlistStorageUsed ;
					bLast->SetParentBucket(b) ;
					}
				else 
					b->SetChildren(0, NULL) ;
				bLast = b ;
				}
			}
		}
	}

	// compute num of functions/variables in each bucket; do this before bt functions are deleted.
	ComputeMaxNumVarsInBucket(false) ;
	// Compute num of roots
	ComputeNumRoots() ;
	// delete bt functions
	if (! KeepBTsignature) {
		for (i = _nBuckets-1 ; i > 0 ; --i) {
			BucketElimination::Bucket *b = _Buckets[i] ;
			if (NULL == b) continue ;
			if (b->nAugmentedFunctions() <= 0) 
				continue ;
			b->DestroyAugmentedFunctions() ;
/*			for (j = b->nAugmentedFunctions() - 1 ; j >= 0 ; j--) {
				ARE::Function *fn = b->AugmentedFunction(j) ;
				if (NULL != fn) {
					b->RemoveAugmentedFunction(*fn, ! KeepBTsignature) ;
					delete fn ;
					}
				}
			b->ResetnAugmentedFunctions() ;
			b->InvalidateSignature() ;*/
			}
		}

// DEBUGGGGG
/*if (NULL != ARE::fpLOG) {
	int64_t tNOW = ARE::GetTimeInMilliseconds() ;
	fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateBuckets 8 ...", tNOW) ;
	fflush(ARE::fpLOG) ;
	}*/

#ifdef _DEBUG
	if (0 != CheckBucketTreeIntegrity()) 
		goto failed ;
#endif // _DEBUG

	_nBuckets_initial = _nBuckets ;
	_nBucketsWithSingleChild_initial = ComputeNBucketsWithSingleChild() ;

	// ***************************************************************************************************
	// check the bucket tree
	// ***************************************************************************************************

#ifdef _DEBUG
	if (0 != CheckBucketTreeIntegrity()) 
		goto failed ;
#endif // _DEBUG

	// ***************************************************************************************************
	// compute stats
	// ***************************************************************************************************

	ComputeBucketTreeStatistics() ;

	_nBuckets_final = _nBuckets ;

	ret = 0 ;

failed :
	if (NULL != fl) 
		delete [] fl ;
	if (NULL != TempSpaceForArglist) 
		delete [] TempSpaceForArglist ;
	return ret ;
}


int32_t BucketElimination::MBEworkspace::CreateSuperBuckets(void)
{
	_nSBmerges = -1 ;
	if (_EClimit <= 0 || _Problem->N() <= 1) 
		return 0 ;

	// ***************************************************************************************************
	// traverse the bucket-tree bottom up; at each step, check if buckets can be merged.
	// at each step, check if current bucket can be merged with its parent :
	//     if the nV_this + nV_parent <= vBound, then they can be merged...
	// ***************************************************************************************************

	/* 
		first, collect all buckets in order, from root down to the leaves...
		we will then process it in reverse order...
	*/

	std::unique_ptr<BucketElimination::Bucket*[]> bfsBuckets(new BucketElimination::Bucket*[_Problem->N()]) ;
	if (nullptr == bfsBuckets) 
		return 1 ;

	// keep track of buckets that changed; we need to recompute them at the end...
	// changedBuckets[i] >0 means bucket of this var has changed...
	_nSBmerges = 0 ;
	std::unique_ptr<BucketElimination::Bucket*[]> changedBuckets(new BucketElimination::Bucket*[_Problem->N()]) ;
	if (nullptr == changedBuckets) 
		return 1 ;
	int32_t nChangedTrees = 0 ;
	std::unique_ptr<BucketElimination::Bucket*[]> changedRootBuckets(new BucketElimination::Bucket*[_Problem->N()]) ;
	if (nullptr == changedRootBuckets) 
		return 1 ;

	// collect all roots
	int32_t n = 0, nRoots = 0, i, j, k ;
	for (i = 0 ; i < _nBuckets ; ++i) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (nullptr == b) continue ;
		if (nullptr == b->ParentBucket() && b->nChildren() > 0) bfsBuckets[n++] = b ;
		}
	if (0 == n) return 0 ;
	nRoots = n ;
	// move down bfsBuckets[] and add children to bfsBuckets...
	for (i = 0 ; i < n ; ++i) {
		BucketElimination::Bucket *b = bfsBuckets[i] ;
		for (j = 0 ; j < b->nChildren() ; ++j) bfsBuckets[n++] = b->ChildBucket(j) ;
		}

	// debug stuff
	int32_t maxVars = 0 ;
	int64_t maxEC = 0 ;

	// process bfsBuckets from leaves up to the root (i.e. in reverse order)
	std::vector<int32_t> tempVlist ;
	for (i = n-1 ; i > 0 ; --i) {
		BucketElimination::Bucket *bCurrent = bfsBuckets[i] ;
		BucketElimination::Bucket *bParent = bCurrent->ParentBucket() ;
		int64_t ecCurrent = bCurrent->ComputeEliminationComplexity() ;
		int32_t nOutputFnVars = bCurrent->Width() - bCurrent->nVars() ;
		if (nOutputFnVars < _iBound) 
			continue ; // output fn of this bucket is ok...
		int64_t ecParent = bParent->ComputeEliminationComplexity() ;
		int32_t ecCombined = ecCurrent*ecParent ;
		if (ecCombined > _EClimit) 
			continue ;
		int32_t nCombined = bCurrent->nVars() + bParent->nVars() ;
		if (ecCombined > maxEC) maxEC = ecCombined ;
		// merge now
		++_nSBmerges ;
		changedBuckets[bParent->V()] = bParent ;
		int32_t iC = bCurrent->IDX(), iP = bParent->IDX() ;
		// bCurrent is not a child of bParent any more; add all children of bCurrent as children of bParent...
		int32_t nNewChildren = bParent->nChildren() + bCurrent->nChildren() - 1 ;
		tempVlist.clear() ;
		if (tempVlist.capacity() < nNewChildren) {
			tempVlist.reserve(nNewChildren) ;
			if (tempVlist.capacity() < nNewChildren) 
				return 1 ;
			}
		for (j = 0 ; j < bParent->nChildren() ; ++j) {
			int32_t cP = bParent->ChildVar(j) ;
			if (cP == bCurrent->V()) continue ;
			BucketElimination::Bucket *bChild = _Var2BucketMapping[cP] ;
			if (nullptr == bChild) continue ;
			tempVlist.push_back(cP) ;
			}
		for (j = 0 ; j < bCurrent->nChildren() ; ++j) {
			int32_t cV = bCurrent->ChildVar(j) ;
			BucketElimination::Bucket *bChild = _Var2BucketMapping[cV] ;
			if (nullptr == bChild) continue ;
			tempVlist.push_back(cV) ;
			bChild->SetParentBucket(bParent) ;
			}
		bParent->SetChildren(tempVlist.size(), tempVlist.data()) ;
		bCurrent->SetChildren(0, nullptr) ;
		bCurrent->SetParentBucket(nullptr) ;
		// variables that are eliminated in bCurrent will now be eliminated in bParent...
		// set bucket[v2Elim in bCurrent] = bParent
		tempVlist.clear() ;
		if (tempVlist.capacity() < nCombined) {
			tempVlist.reserve(nCombined) ;
			if (tempVlist.capacity() < nCombined) 
				return 1 ;
			}
		tempVlist = bParent->Vars() ;
		for (j = 0 ; j < bCurrent->nVars() ; ++j) {
			int32_t v2Elim = bCurrent->Var(j) ;
			tempVlist.push_back(v2Elim) ;
			_Var2BucketMapping[v2Elim] = bParent ;
			}
		int32_t res_SetVars = bParent->SetVars(tempVlist.size(), tempVlist.data()) ;
		if (0 != res_SetVars) 
			return 1 ;
		if (tempVlist.size() > maxVars) maxVars = tempVlist.size() ;
		// add all original functions of bCurrent to bParent; invalidate bParent signature...
		if (bParent->AddOriginalFunctions(bCurrent->nOriginalFunctions(), bCurrent->OriginalFunctionsArray(), false)) 
			return 1 ;
		if (bParent->AddAugmentedFunctions(bCurrent->nAugmentedFunctions(), bCurrent->AugmentedFunctionsArray(), false)) 
			return 1 ;
//		bParent->InvalidateSignature() ;
		int32_t n = bCurrent->nVars() ;
		bParent->AddVarsToSignature(n, bCurrent->VarsArray()) ;
		// store the root bucket of bParent, so that we can later go back to it...
		BucketElimination::Bucket *bRoot = bParent->RootBucket() ;
		for (j = 0 ; j < nChangedTrees ; ++j) {
			if (bRoot == changedRootBuckets[j]) break ; }
		if (j >= nChangedTrees) changedRootBuckets[nChangedTrees++] = bRoot ;
		// we can now delete bCurrent
		_Buckets[bCurrent->IDX()] = nullptr ;
		delete bCurrent ;
		}

	// delete all intermediate functions, ... keep original/augmented functions...
	for (k = 0 ; k < nChangedTrees ; ++k) {
		BucketElimination::Bucket *bRoot = changedRootBuckets[k] ;
		n = 1 ;
		bfsBuckets[0] = bRoot ;
		for (i = 0 ; i < n ; ++i) {
			BucketElimination::Bucket *b = bfsBuckets[i] ;
//			b->ResetnAugmentedFunctions() ;
			b->ResetnIntermediateFunctions() ;
			b->DestroyMBPartitioning() ;
			for (j = 0 ; j < b->nChildren() ; ++j) {
				BucketElimination::Bucket *bChild = b->ChildBucket(j) ;
				bfsBuckets[n++] = bChild ;
				}
			}
		}

	TidyBucketsArray() ;

	// compute all stats...
	ComputeBucketTreeStatistics() ;

	// misc
	_nBucketsWithPartitioning = -1 ; // this can only be computed after MB partitioning is done...
	_nBucketsWithSingleChild_final = ComputeNBucketsWithSingleChild() ;
	_nBuckets_final = _nBuckets ;

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed(int32_t MinIBound, int32_t & Ibound, double & NewFnSpaceUsed_Log10, int32_t & nBucketsPartitioned, int32_t & maxDepthPartitionedBucket)
{
	Ibound = -1 ; NewFnSpaceUsed_Log10 = -1 ; nBucketsPartitioned = -1 ; maxDepthPartitionedBucket = -1 ;
//	int32_t original_i = _iBound ;

	// get space for i=2; if this does not work, nothing will work.
	double iMINspace = -1 ;
	int32_t iLower ;
	int32_t iUpper ;
	double iLowerspace ;
	double iUpperspace ;
	int32_t iLNP ;
	int32_t iMaxDepthPartitionedBucket ;
	_iBound = MinIBound ;
	if (0 != CreateMBPartitioning(false, false, 0, false, 0)) {
		fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; ERROR failed for i-bound=%d ...", (int32_t) _iBound) ;
		::fflush(ARE::fpLOG) ;
		goto failed ;
		}
	iMINspace = _DeleteUsedTables ? _MaxSimultaneousNewFunctionSpace_Log10 : _TotalNewFunctionSpace_Log10 ;
	if (NULL != ARE::fpLOG) {
		fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; i-bound=%d space = %g ...", (int32_t) _iBound, iMINspace) ;
		::fflush(ARE::fpLOG) ;
		}
	if (iMINspace > _MaxSpaceAllowed_Log10) {
		if (NULL != ARE::fpLOG) {
			fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; i-bound=%d space exceeds max allowed %g; will quit ...", (int32_t) _iBound, _MaxSpaceAllowed_Log10) ;
			::fflush(ARE::fpLOG) ;
			}
		goto failed ;
		}
	iLower = MinIBound ; // iLower is largest ibound that we know that does work; actual largest ibound with space_allowed may be larger
	iUpper = _VarOrdering_MaxCliqueSize > 0 ? _VarOrdering_MaxCliqueSize : _nVars ;
	iLowerspace = iMINspace ;
	iLNP = _nBucketsWithPartitioning ;
	iMaxDepthPartitionedBucket = _maxDepthPartitionedBucket ;
//goto done ;
	if (_VarOrdering_MaxCliqueSize > 0 && _VarOrdering_MaxCliqueSize <= 2) {
		// if w* is < 2, the i=2 is all we need to check, since minibucket size limit of 2 should be sufficient
		if (0 != _nBucketsWithPartitioning) {
			if (NULL != ARE::fpLOG) {
				fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; ERROR : var_order maxCliqueSize=%d w*(%d)<2 and i-bound=%d, but there is partitioning ...", (int32_t) _VarOrdering_MaxCliqueSize, (int32_t) (_VarOrdering_MaxCliqueSize-1), (int32_t) _iBound) ;
				::fflush(ARE::fpLOG) ;
				}
			goto failed ;
			}
		goto done ;
		}
	if (0 == _nBucketsWithPartitioning) 
		goto done ; // increasing i-bound will not improve; already no partitioning

	// iUpper might work; needs to be checked.
	_iBound = iUpper ;
	if (0 != CreateMBPartitioning(false, false, 0, false, 0)) 
		goto done ;
	iUpperspace = _DeleteUsedTables ? _MaxSimultaneousNewFunctionSpace_Log10 : _TotalNewFunctionSpace_Log10 ;
	if (NULL != ARE::fpLOG) {
		fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; i-bound=%d space = %g ...", (int32_t) _iBound, (double) iUpperspace) ;
		::fflush(ARE::fpLOG) ;
		}
	if (0 != _nBucketsWithPartitioning) {
		if (NULL != ARE::fpLOG) {
			fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; ERROR : ibound(%d) >= var_order maxCliqueSize(%d) [w*=%d], but there is partitioning ...", (int32_t) _iBound, (int32_t) _VarOrdering_MaxCliqueSize, (int32_t) (_VarOrdering_MaxCliqueSize-1)) ;
			::fflush(ARE::fpLOG) ;
			}
		goto failed ;
		}
	if (iUpperspace <= _MaxSpaceAllowed_Log10) 
		{ iLower = iUpper ; iLowerspace = iUpperspace ; iLNP = _nBucketsWithPartitioning ; iMaxDepthPartitionedBucket = _maxDepthPartitionedBucket ; goto done ; }

	// iLower is ok, iUpper is not ok.
	while (iLower < iUpper) {
		_iBound = (iLower + iUpper)/2 ;
		if (_iBound <= iLower) 
			goto done ;
		double space = -1 ;
		if (0 != CreateMBPartitioning(false, false, 0, false, 0)) 
			goto done ;
		space = _DeleteUsedTables ? _MaxSimultaneousNewFunctionSpace_Log10 : _TotalNewFunctionSpace_Log10 ;
		if (NULL != ARE::fpLOG) {
			fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; i-bound=%d space = %g ...", (int32_t) _iBound, (double) space) ;
			}
		if (space <= _MaxSpaceAllowed_Log10) {
			iLower = _iBound ; iLowerspace = space ; iLNP = _nBucketsWithPartitioning ; iMaxDepthPartitionedBucket = _maxDepthPartitionedBucket ;
			if (0 == _nBucketsWithPartitioning) 
				// there is no point in checking higher iBounds since we have no partitioning at current iBound.
				goto done ;
			}
		else 
			{ iUpper = _iBound ; }
		}

done :
	if (NULL != ARE::fpLOG) {
		fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::FindIBoundForSpaceAllowed; done; iLower=%d iUpper=%d iLowerspace=%g ...", (int32_t) iLower, (int32_t) iUpper, (double) iLowerspace) ;
		::fflush(ARE::fpLOG) ;
		}
//	_iBound = original_i ;
	Ibound = iLower ;
	NewFnSpaceUsed_Log10 = iLowerspace ;
	nBucketsPartitioned = iLNP ;
	maxDepthPartitionedBucket = iMaxDepthPartitionedBucket ;
	return 0 ;

failed :
//	_iBound = original_i ;
	return 1 ;
}


int32_t BucketElimination::MBEworkspace::CreateMBPartitioning(bool CreateTables, bool doMomentMatching, signed char Bound, bool noPartitioning, int32_t ComputeComputationOrder)
{
	DestroyMBPartitioning() ;

/*
// count num of functions
__int64 nF = 0 ;
for (int32_t i = _nBuckets-1 ; i >= 0 ; i--) {
	BucketElimination::Bucket *b = _Buckets[i] ;
	nF += b->nOriginalFunctions() + b->nAugmentedFunctions() ;
	}
fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::CreateMBPartitioning; i-bound=%d nF=%lld CreateTables=%c", (int32_t) _iBound, nF, CreateTables ? 'Y' : 'N') ;
::fflush(ARE::fpLOG) ;
*/

	// if computation order does not exist, compute it here
	bool we_computed_comp_order = false ;
	if (_lenComputationOrder <= 0) {
		we_computed_comp_order = true ;
		int32_t res = CreateComputationOrder(0) ;
		if (0 != res) 
			return res ;
		if (_lenComputationOrder <= 0) 
			return 0 ;
		}

	// process buckets, from last to first!
	std::vector<int32_t> key ; // keys are fn arity
	std::vector<int64_t> data ; // data a ptrs to fns
	std::vector<int32_t> helperArray ;
	int32_t left[32], right[32] ;
	_nBucketsWithPartitioning = 0 ;
	_MaxNumMiniBucketsPerBucket = 0 ;
	_maxDepthPartitionedBucket = 0 ;
	bool abandon = false ;
	for (int32_t i = _lenComputationOrder-1 ; i >= 0 ; --i) {
		int32_t idxToCompute = BucketOrderToCompute(i) ;
		if (idxToCompute < 0) 
			goto failed ;
		BucketElimination::Bucket *b = getBucket(idxToCompute) ;
		if (NULL == b) continue ;
/*	for (int32_t i = _Problem->N()-1 ; i >= 0 ; --i) {
		int32_t v = _Problem->VarOrdering_VarList()[i] ;
		BucketElimination::Bucket *b = MapVar2Bucket(v) ;
		if (NULL == b) continue ;*/
		int32_t res = b->CreateMBPartitioning(_iBound, CreateTables, doMomentMatching, Bound, noPartitioning, abandon, key, data, helperArray) ;
		if (0 != res) 
			goto failed ;
		if (b->nMiniBuckets() > _MaxNumMiniBucketsPerBucket) 
			_MaxNumMiniBucketsPerBucket = b->nMiniBuckets() ;
		if (b->nMiniBuckets() > 1) {
			_nBucketsWithPartitioning++ ;
			if (b->DistanceToRoot() > _maxDepthPartitionedBucket) 
				_maxDepthPartitionedBucket = b->DistanceToRoot() ;
			}
		}

	// compute this when MB partitioning is done
	ComputeMaxBucketFunctionWidth() ;
	ComputeTotalNewFunctionSizeAndSpace() ;
	ComputeTotalNewFunctionComputationComplexity() ;
	if (ComputeComputationOrder >= 0) {
		if (_lenComputationOrder <= 0) // no current order
			CreateComputationOrder(ComputeComputationOrder) ;
		else if (! we_computed_comp_order || ComputeComputationOrder > 0) {
			CreateComputationOrder(ComputeComputationOrder) ;
			}
		}
	SimulateComputationAndComputeMinSpace(false) ;

	// test that when i >= w*, there is no partitioning
	// 2021-12-05 KK : this above is incorrect; iBound>=maxCliqueSize should produce no partitioning...
	if (_VarOrdering_MaxCliqueSize > 0 && _iBound >= _VarOrdering_MaxCliqueSize) {
		if (_nBucketsWithPartitioning > 0) {
			if (NULL != ARE::fpLOG) {
				fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::CreateMBPartitioning ERROR : _iBound(%d) >= var_order maxCliqueSize(%d) [w*=%d] but there is partitioning ...", (int32_t) _iBound, (int32_t) _VarOrdering_MaxCliqueSize, (int32_t) _VarOrdering_MaxCliqueSize-1) ;
				::fflush(ARE::fpLOG) ;
				}
			int32_t error = 1 ;
			}
		}

	return 0 ;

failed :
	fprintf(ARE::fpLOG, "\n   BucketElimination::MBEworkspace::CreateMBPartitioning ERROR : failed label reached ...") ;
	::fflush(ARE::fpLOG) ;
	DestroyBucketPartitioning() ;
	return 1 ;
}


int32_t BucketElimination::MBEworkspace::DestroyMBPartitioning(void)
{
	_nBucketsWithPartitioning = -1 ;
	for (int32_t i = _nBuckets-1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (NULL == b) continue ;
		b->ResetnAugmentedFunctions() ;
		b->ResetnIntermediateFunctions() ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			mb->Destroy() ;
			delete mb ;
			}
		mbs.clear() ;
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::CheckBucketTreeIntegrity(void)
{
	int32_t i, j, k ;

	for (i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		BucketElimination::Bucket *B = b->ParentBucket() ;
		// check indeces are within range
		if (b->IDX() < 0 || b->IDX() >= _nBuckets) 
			goto failed ;
		if (NULL != B ? B->IDX() < 0 || B->IDX() >= _nBuckets : false) 
			goto failed ;
		// check width is computed
		if (b->Width() < 0) {
			int32_t res_sig = b->ComputeSignature() ;
			if (0 != res_sig || b->Width() < 0) 
				goto failed ;
			}
		if (0 == b->nOriginalFunctions() && 0 == b->nChildren()) {
			if (NULL != B) {
				goto failed ;
				}
			}
		// check bVar, Signature are unique
		for (j = 0 ; j < b->nVars() ; j++) {
			int32_t u = b->Var(j) ;
			for (k = j+1 ; k < b->nVars() ; k++) {
				if (u == b->Var(k)) 
					goto failed ;
				}
			}
		const int32_t *bSig = b->Signature() ;
		for (j = 0 ; j < b->Width() ; j++) {
			int32_t u = bSig[j] ;
			for (k = j+1 ; k < b->Width() ; k++) {
				if (u == bSig[k]) 
					goto failed ;
				}
			}
		// if this bucket has children, it must have augmented functions;
		// 2016-03-28 KK : this is not true when MB partitioning is created; also, MB partitioning may be missing.
//		if (b->nChildren() > 0 && 0 == b->nAugmentedFunctions()) 
//			goto failed ;
		// check minibuckets and output function(s)
		int32_t nF = b->nOriginalFunctions() + b->nAugmentedFunctions() ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		if (nF <= 1 ? mbs.size() > 1 : false) 
			goto failed ;
		if (0 == nF && b->Width() > b->nVars() && b->nChildren() <= 0) 
			goto failed ;
		for (MiniBucket *mb : mbs) {
//			if (mb.nVars() > b->Width()) 
//				goto failed ;
// 2016-11-17 KK : cannot check if mb->width>=iB, even if nFNs>1, since there may be on fn with nArgs>iB, and then can put lots of small FNs in the MB.
//			if (_iBound > 0 ? mb->Width() >= _iBound : false) 
//				goto failed ;
			for (j = 0 ; j < mb->nFunctions() ; j++) {
				ARE::Function *f = mb->Function(j) ;
				if (NULL == f) goto failed ;
				if (f->Bucket() != b) 
					goto failed ;
				if (f->IDX() >= 0 && NULL != f->OriginatingBucket()) 
					goto failed ;
				if (f->IDX() < 0 && NULL == f->OriginatingBucket()) 
					goto failed ;
				}
			ARE::Function & f = mb->OutputFunction() ;
			if (f.Bucket() != B) 
				goto failed ;
			if (f.OriginatingBucket() != B) 
				goto failed ;
			if (f.OriginatingMiniBucket() != mb) 
				goto failed ;
			if (f.N() > 0 && NULL == B) 
				goto failed ;
			if (0 == f.N() && NULL != B) 
				goto failed ;
			if (f.N() > b->Width() - b->nVars()) 
				goto failed ;
			if (1 == mbs.size()) {
				if (b->Width() != f.N() + b->nVars()) 
					goto failed ;
				}
			if (0 == nF) {
				if (0 != f.N()) 
					goto failed ;
				}
			// check f->args are unique
			const int32_t *fArgs = f.Arguments() ;
			for (j = 0 ; j < f.N() ; j++) {
				int32_t u = fArgs[j] ;
				for (k = j+1 ; k < f.N() ; k++) {
					if (u == fArgs[k]) 
						goto failed ;
					}
				}
			// check f->args and bVars are disjoint
			for (j = 0 ; j < f.N() ; j++) {
				int32_t u = fArgs[j] ;
				for (k = 0 ; k < b->nVars() ; k++) {
					if (u == b->Var(k)) 
						goto failed ;
					}
				}
			for (j = 0 ; j < b->nVars() ; j++) {
				int32_t u = b->Var(j) ;
				for (k = 0 ; k < f.N() ; k++) {
					if (u == fArgs[k]) 
						goto failed ;
					}
				}
			// check all f->args are in signature
			for (j = 0 ; j < f.N() ; j++) {
				int32_t u = fArgs[j] ;
				for (k = 0 ; k < b->Width() ; k++) {
					if (u == bSig[k]) 
						break ;
					}
				if (k >= b->Width()) 
					goto failed ;
				}
			}
		// check children
		for (j = 0 ; j < b->nChildren() ; j++) {
			int32_t childvar = b->ChildVar(j) ;
			if (childvar < 0 || childvar >= _nVars) 
				goto failed ;
			BucketElimination::Bucket *cb = _Var2BucketMapping[childvar] ;
			if (NULL == cb) 
				goto failed ;
			if (b != cb->ParentBucket()) 
				goto failed ;
			std::vector<MiniBucket *> & cmbs = b->MiniBuckets() ;
			for (MiniBucket *cmb : cmbs) {
				ARE::Function & cf = cmb->OutputFunction() ;
				// TODO : if cf has a var in v->Vars() then it must be in b->AugmentedFunctions; otherwise, in b->IntermediateFunctions.
				}
			}
		}

	return 0 ;
failed :
	if (NULL != ARE::fpLOG) {
		fprintf(ARE::fpLOG, "\nERROR : bucket tree integrity check") ;
		::fflush(ARE::fpLOG) ;
		}
	return 1 ;
}


int32_t BucketElimination::MBEworkspace::CreateComputationOrder(int32_t algorithm)
{
	int32_t i ;

	if (NULL != ARE::fpLOG) {
		int64_t tNOW = ARE::GetTimeInMilliseconds() ;
		fprintf(ARE::fpLOG, "\n%lld      BEWS : CreateComputationOrder() ...", tNOW) ;
		fflush(ARE::fpLOG) ;
		}

	_lenComputationOrder = 0 ;
	if (NULL == _BucketOrderToCompute || _nBuckets < 1) 
		return 1 ;

	if (1 == algorithm) {
		BucketElimination::Bucket *BLhead = NULL, *BLtail = NULL ;
		int64_t nAdded = 0 ;
		// collect all roots
		for (i = 0 ; i < _nBuckets ; i++) {
			BucketElimination::Bucket *b = _Buckets[i] ;
			if (NULL == b->ParentBucket()) {
				if (NULL == BLhead) 
					{ BLhead = BLtail = b ; }
				else 
					{ BLtail->NextInOrder() = b ; BLtail = b ; }
				b->NextInOrder() = NULL ;
				++nAdded ;
				}
			}
		// go through the list; for each bucket, compute all its children before buckets lates
		BucketElimination::Bucket *B = BLhead ;
		std::vector<int64_t> bucket_fn_num_vars ; bucket_fn_num_vars.reserve(1024) ;
		std::vector<BucketElimination::Bucket *> child_buckets ; child_buckets.reserve(1024) ;
		int32_t left[32], right[32] ;
		while (NULL != B) {
			_BucketOrderToCompute[_lenComputationOrder++] = B->IDX() ;
			// sort children of B
			bucket_fn_num_vars.clear() ; child_buckets.clear() ;
			for (int32_t j = 0 ; j < B->nChildren() ; j++) {
				int32_t childvar = B->ChildVar(j) ;
				BucketElimination::Bucket *orig_b = _Var2BucketMapping[childvar] ;
				if (NULL == orig_b) continue ;
				int32_t size_before = child_buckets.size() ;
				child_buckets.push_back(orig_b) ;
//				bucket_fn_num_vars[nChildren] = -f->N() ; // order from largest to smallest
//				bucket_fn_num_vars[nChildren] = child_buckets[nChildren]->MaxDescendantNumVarsEx() ; // order from smallest to largest
				bucket_fn_num_vars.push_back(orig_b->MaxDescendantComputationNewFunctionSizeEx()) ; // order from smallest to largest
				if (child_buckets.size() != bucket_fn_num_vars.size() || child_buckets.size() != size_before+1) 
					return 1 ;
				}
			int32_t nChildren = child_buckets.size() ;
			if (nChildren > 0) {
				QuickSorti64_i64(bucket_fn_num_vars.data(), nChildren, (int64_t *) child_buckets.data(), left, right) ;
				// add all children of B right after B
				BucketElimination::Bucket *Bnext = B->NextInOrder(), *b = B ;
				for (int32_t j = 0 ; j < nChildren ; j++) {
					Bucket *bChild = child_buckets[j] ;
					b->NextInOrder() = bChild ;
					b = bChild ;
					++nAdded ;
					}
				b->NextInOrder() = Bnext ;
				}
			B = B->NextInOrder() ;
			}
		if (nAdded != _nBuckets || _lenComputationOrder != _nBuckets) 
			return 1 ;
		}
	else {
		// sort by height descending
		int32_t *keys = new int32_t[_nBuckets] ;
		if (NULL == keys)
			return 1 ;
		for (i = 0; i < _nBuckets; i++) {
			BucketElimination::Bucket *b = _Buckets[i] ;
			keys[i] = b->Height() ;
			_BucketOrderToCompute[_lenComputationOrder++] = b->IDX() ;
			}
		int32_t left[32], right[32] ;
		QuickSortLong_Descending(keys, _lenComputationOrder, _BucketOrderToCompute, left, right) ;
		delete [] keys ;
		}

	return 0 ;
}


__int64 BucketElimination::MBEworkspace::ComputeTotalOriginalFunctionSizeAndSpace(void)
{
	_TotalOriginalFunctionSize = _TotalOriginalFunctionSpace = 0 ;
	if (NULL == _Problem) 
		return 0 ;
	int32_t NF = _Problem->nFunctions() ;
	for (int32_t i = 0 ; i < NF ; i++) {
		ARE::Function *f = _Problem->getFunction(i) ;
		int64_t tsize = f->ComputeTableSize() ;
		if (tsize > 0) 
			_TotalOriginalFunctionSize += tsize ;
		int64_t tspace = f->ComputeTableSpace() ;
		if (tspace > 0) 
			_TotalOriginalFunctionSpace += tspace ;
		}
	return _TotalOriginalFunctionSize ;
}


int32_t BucketElimination::MBEworkspace::ComputeMaxNumVarsInBucket(bool ForceComputeSignature)
{
	_MaxNumVarsInBucket = 0 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (nullptr == b) continue ;
		int32_t width_original = b->Width() ;
		if (ForceComputeSignature) 
			b->InvalidateSignature() ;
		if (b->Width() < 0) 
			b->ComputeSignature() ;
		if (b->Width() > _MaxNumVarsInBucket) 
			_MaxNumVarsInBucket = b->Width() ;
		}
	return 0 ;
}


BucketElimination::Bucket *BucketElimination::MBEworkspace::GetBucketWithMostVariables()
{
	int32_t MaxNumVarsInBucket = 0 ; BucketElimination::Bucket *B = NULL ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		int32_t width_original = b->Width() ;
		if (b->Width() < 0) 
			b->ComputeSignature() ;
		if (b->Width() > MaxNumVarsInBucket) {
			MaxNumVarsInBucket = b->Width() ;
			B = b ;
			}
		}
	return B ;
}


BucketElimination::Bucket *BucketElimination::MBEworkspace::GetBucketWithMostElimVariables()
{
	int32_t MaxNumVarsInBucket = 0 ; BucketElimination::Bucket *B = NULL ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		int32_t width_original = b->Width() ;
		if (b->Width() < 0) 
			b->ComputeSignature() ;
		if (b->nVars() > MaxNumVarsInBucket) {
			MaxNumVarsInBucket = b->nVars() ;
			B = b ;
			}
// temp hack
// if (MaxNumVarsInBucket > 1)
// 	return B ;
		}
	return B ;
}


int32_t BucketElimination::MBEworkspace::ComputeNBucketsWithSingleChild(void)
{
	_nBucketsWithSingleChild = _nBucketsWithNoChildren = 0 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (NULL == b) continue ;
		if (0 == b->nChildren()) 
			_nBucketsWithNoChildren++ ;
		else if (1 == b->nChildren()) 
			_nBucketsWithSingleChild++ ;
		}
	return _nBucketsWithSingleChild ;
}


int32_t BucketElimination::MBEworkspace::ComputeMaxNumChildren(void)
{
	_MaxNumChildren = 0 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (b->nChildren() > _MaxNumChildren) 
			_MaxNumChildren = b->nChildren() ;
		}
	return _MaxNumChildren ;
}


void BucketElimination::MBEworkspace::ComputeTotalNewFunctionSizeAndSpace(void)
{
	_TotalNewFunctionSize_Log10 = _TotalNewFunctionSpace_Log10 = -1.0 ;
	double temp_sum_1 = 1.0, temp_sum_2 = 1.0 ;
	// find max size/space
	int32_t maxB = -1, maxMB = -1 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		b->SetComputationNewFunctionSize(-1) ;
		b->SetMaxDescendantComputationNewFunctionSize(-1) ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		double local_max_value = -1.0 ; int32_t local_max_idx = -1 ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			double x = f.GetTableSize_Log10() ;
			double y = f.GetTableSpace_Log10() ;
			if (x > _TotalNewFunctionSize_Log10) {
				maxB = i ; maxMB = mb->IDX() ;
				_TotalNewFunctionSize_Log10 = f.GetTableSize_Log10() ;
				_TotalNewFunctionSpace_Log10 = f.GetTableSpace_Log10() ;
				}
			if (x > local_max_value) {
				local_max_idx = mb->IDX() ;
				local_max_value = f.GetTableSize_Log10() ;
				}
			}
		// compute bucket computation size
		double temp_sum = 1.0 ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			if (local_max_idx != mb->IDX()) {
				temp_sum += pow(10.0, f.GetTableSize_Log10() - local_max_value) ;
				}
			}
		local_max_value += log10(temp_sum) ;
		b->SetComputationNewFunctionSize(local_max_value) ;
		}
	if (_TotalNewFunctionSize_Log10 < 0.0) 
		return ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			if (maxB != i || maxMB != mb->IDX()) {
				temp_sum_1 += pow(10.0, f.GetTableSize_Log10() - _TotalNewFunctionSize_Log10) ;
				temp_sum_2 += pow(10.0, f.GetTableSpace_Log10() - _TotalNewFunctionSpace_Log10) ;
				}
			}
		}
	_TotalNewFunctionSize_Log10 += log10(temp_sum_1) ;
	_TotalNewFunctionSpace_Log10 += log10(temp_sum_2) ;

	// compute max descendant new fn computation size
	for (int32_t i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		BucketElimination::Bucket *B = b->ParentBucket() ;
		if (NULL != B) {
			int64_t mdnfns = b->MaxDescendantComputationNewFunctionSizeEx() ;
			if (B->MaxDescendantComputationNewFunctionSize() < mdnfns) 
				B->SetMaxDescendantComputationNewFunctionSize(mdnfns) ;
			}
		}
}


void BucketElimination::MBEworkspace::ComputeTotalNewFunctionComputationComplexity(void)
{
	_TotalNewFunctionComputationComplexity_Log10 = -1.0 ;
	double *b2c_map = new double[_nBuckets+1] ;
	if (NULL == b2c_map) 
		return ;
	// compute max; we need max since log_sum computation has to start from max, otherwise accuracy will be lost
	int32_t maxB = -1, maxMB = -1 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			const int32_t *vars = mb->Signature() ;
			if (nullptr == vars) {
				mb->ComputeSignature() ;
				vars = mb->Signature() ;
				if (nullptr == vars) {
					_TotalNewFunctionComputationComplexity_Log10 = -1.0 ;
					return ;
					}
				}
			int32_t nFunctions = mb->nFunctions() ;
			if (0 == nFunctions) 
				continue ;
			int32_t w = mb->nVars() ;
			double table_size_Log10 = 0.0 ;
			for (int32_t j = 0 ; j < w ; j++) {
				int32_t k = _Problem->K(vars[j]) ;
				if (k >= 1) 
					table_size_Log10 += log10((double) k) ;
				}
			table_size_Log10 += log10((double) nFunctions) ;
			if (_TotalNewFunctionComputationComplexity_Log10 < table_size_Log10) 
				{ maxB = i ; maxMB = mb->IDX() ; _TotalNewFunctionComputationComplexity_Log10 = table_size_Log10 ; }
			}
		}
	if (_TotalNewFunctionComputationComplexity_Log10 < 0.0) 
		return ;
	// add up everything, except max
	double temp_sum = 1.0 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			if (maxB == i && maxMB == mb->IDX()) 
				continue ;
			const int32_t *vars = mb->Signature() ;
			int32_t nFunctions = mb->nFunctions() ;
			if (0 == nFunctions) 
				continue ;
			int32_t w = mb->nVars() ;
			double table_size_Log10 = 0.0 ;
			for (int32_t j = 0 ; j < w ; j++) {
				int32_t k = _Problem->K(vars[j]) ;
				if (k >= 1) 
					table_size_Log10 += log10((double) k) ;
				}
			table_size_Log10 += log10((double) nFunctions) ;
			temp_sum += pow(10.0, table_size_Log10 - _TotalNewFunctionComputationComplexity_Log10) ;
			}
		}
	_TotalNewFunctionComputationComplexity_Log10 += log10(temp_sum) ;
}


void BucketElimination::MBEworkspace::SimulateComputationAndComputeMinSpace(bool IgnoreInMemoryTables)
{
	int32_t i ;

	// compute table sizes
	for (i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			f.ComputeTableSize() ;
			}
		}

	// assume computation order is set

	// compute maximum space for any bucket
	double size = -1.0 ;
//double size_DG = 0.0 ;
	_MaxSimultaneousNewFunctionSize_Log10 = -1.0 ;
	for (i = _lenComputationOrder - 1 ; i >= 0 ; i--) {
		int32_t idx = _BucketOrderToCompute[i] ;
		BucketElimination::Bucket *b = _Buckets[idx] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			// add the space of the output function of this bucket
			double s = f.GetTableSize_Log10() ;
			if (s >= 0.0) {
				if (size < 0.0) 
					size = s ;
				else if (size >= s) {
					size = size + log10(1.0 + pow(10.0, s - size)) ;
					}
				else {
					size = s + log10(1.0 + pow(10.0, size - s)) ;
					}
				if (size > _MaxSimultaneousNewFunctionSize_Log10) 
					_MaxSimultaneousNewFunctionSize_Log10 = size ;
				}
			}
		if (_DeleteUsedTables) {
			for (int32_t j = 0 ; j < b->nAugmentedFunctions() ; j++) {
				ARE::Function *cf = b->AugmentedFunction(j) ;
				if (NULL == cf) continue ;
				double sf = cf->GetTableSize_Log10() ;
				if (sf < 0.0) continue ;
				if (size <= sf) 
					{ size = -1.0 ; break ; }
				LOG_OF_SUB_OF_TWO_NUMBERS_GIVEN_AS_LOGS(size, size, sf) 
//				size = size + log10(1.0 - pow(10.0, sf - size)) ;
//size_DG -= cf->TableSize() ;
				}
			}
		}

	_MaxSimultaneousNewFunctionSpace_Log10 = log10(sizeof(ARE_Function_TableType)) + _MaxSimultaneousNewFunctionSize_Log10 ;
	double tofs = log10(_TotalOriginalFunctionSize) ;
	LOG_OF_SUM_OF_TWO_NUMBERS_GIVEN_AS_LOGS(_MaxSimultaneousTotalFunctionSize_Log10, _MaxSimultaneousNewFunctionSize_Log10, tofs) 
	_MaxSimultaneousTotalFunctionSpace_Log10 = log10(sizeof(ARE_Function_TableType)) + _MaxSimultaneousTotalFunctionSize_Log10 ;
}


int32_t BucketElimination::MBEworkspace::ComputeMaxBucketFunctionWidth(void)
{
	_MaxBucketFunctionWidth = 0 ;
	for (int32_t i = 0 ; i < _nBuckets ; i++) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			if (f.N() > _MaxBucketFunctionWidth) _MaxBucketFunctionWidth = f.N() ;
			}
		}
	return _MaxBucketFunctionWidth ;
}


int32_t BucketElimination::MBEworkspace::RunSimple(void)
{
	int32_t i ;

	for (i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			f.AllocateInMemoryAsSingleTableBlock() ;
			}
		}

	// compute all minibucket output functions
	for (i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<MiniBucket *> & mbs = b->MiniBuckets() ;
		int32_t varElimOperator = VarEliminationType() ;
		for (MiniBucket *mb : mbs) {
			ARE::Function & f = mb->OutputFunction() ;
			mb->ComputeOutputFunction(varElimOperator, false, NULL, NULL, DBL_MAX) ;
			// if problem is summation, sum over first mini-bucket, max (or min) over other minibuckets.
			if (VAR_ELIMINATION_TYPE_SUM == varElimOperator) 
				varElimOperator = VAR_ELIMINATION_TYPE_MAX ;
			}
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::ComputeOutputFunctions(bool DoMomentMatching, bool LogConsole, signed char approx_bound, int64_t & TotalSumOutputFunctionsNumEntries)
{
	TotalSumOutputFunctionsNumEntries = 0 ;

	int32_t i ;

#ifdef SAMPLE_BUCKET_TEST
	BucketElimination::Bucket *bMax = nullptr ;
#endif

	if (LogConsole) 
		printf("\nComputing output function, nBuckets=%d", (int32_t) _lenComputationOrder) ;
	_CurrentComputationBound = approx_bound ;
	for (int32_t i = _lenComputationOrder-1 ; i >= 0 ; --i) {
		int32_t idxToCompute = BucketOrderToCompute(i) ;
		if (idxToCompute < 0) 
			return 1 ;
		BucketElimination::Bucket *b = getBucket(idxToCompute) ;
		if (NULL == b) 
			return 1 ;
#ifdef SAMPLE_BUCKET_TEST
		if (nullptr == bMax ? true : bMax->Width() < b->Width()) 
			bMax = b ;
#endif
		int64_t size = 0 ;
//printf("\nDEBUGGG i=%d nV=%d", i, b->Width()) ;
		if (LogConsole) {
			BucketElimination::MiniBucket *mb0 = b->nMiniBuckets() > 0 ? (b->MiniBuckets())[0] : nullptr ;
			ARE::Function *f0 = nullptr != mb0 ? &(mb0->OutputFunction()) : nullptr ;
			printf("\n   i=%d idxBucket=%d bVar=%d bWidth=%d outputFNsize=%lld", 
				i, (int32_t) b->IDX(), (int32_t) b->V(), (int32_t) b->Width(), (int64_t) (nullptr != f0 ? f0->TableSize() : -1)) ;
			}
		int32_t res_mbe = b->ComputeOutputFunctions(DoMomentMatching, approx_bound, size) ;
//printf("\nDEBUGGG i=%d done res=%d", i, res_mbe) ;
		if (LogConsole) 
			printf(" res=%d", res_mbe) ;
		if (0 != res_mbe) 
			return res_mbe ;
		TotalSumOutputFunctionsNumEntries += size ;
		b->NoteOutputFunctionComputationCompletion() ;
		}

#ifdef SAMPLE_BUCKET_TEST
#if defined WINDOWS || _WINDOWS
unsigned int __stdcall BucketSamplingWorkerThreadFn(void *X) ;
#elif defined (LINUX)
void *BucketSamplingWorkerThreadFn(void *X) ;
#endif 
	if (nullptr != bMax) {
		int64_t nSamplesPerThread = 10000 ;
		std::vector<MiniBucket *> & mbs = bMax->MiniBuckets() ;
		BucketElimination::MiniBucket *mb0 = mbs[0] ;
		ARE::ThreadPool tp(ARE::maxNumProcessorThreads-1) ;
		tp._pThredFn = BucketSamplingWorkerThreadFn ;
		tp.Create() ;
		for (int32_t i = 0 ; i < tp._nThreads ; ++i) {
			tp._Tasks[i]._WorkDone = false ;
			tp._Tasks[i]._MB = mb0 ;
			tp._Tasks[i]._nSamples = nSamplesPerThread ;
			}
		// wait until all work done
		while (true) {
			int32_t nStillWorking = 0 ;
			for (int32_t i = 0 ; i < tp._nThreads ; ++i) {
				if (! tp._Tasks[i]._WorkDone) 
					++nStillWorking ;
				}
			if (0 == nStillWorking) 
				break ;
			SLEEP(25) ;
			}
		// tp._Task[] array contains the samples now...
		// E.G.
		for (int32_t i = 0 ; i < tp._nThreads ; ++i) {
			ARE::ThreadPoolThreadContext & task = tp._Tasks[i] ;
			for (int32_t j = 0 ; j < task._nSamples ; ++j) {
				int16_t *sig = task.Signature(j) ; // task._nFeaturesPerSample values sig[0...task._nFeaturesPerSample-1]
				float label = *task.Label(j) ;
				for (int32_t k = 0 ; k < task._nFeaturesPerSample ; ++k) {
					int16_t sample_i_feature_k = sig[k] ;
					}
				}
			}
		}
#endif

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::ComputeFunctionArgumentsPermutationList(void)
{
	for (int32_t i = _nVars - 1 ; i >= 0 ; --i) {
		int32_t v = _VarList[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL == b) continue ;
		int32_t nF = b->nOriginalFunctions() + b->nAugmentedFunctions() ;
		for (int32_t j = 0 ; j < nF ; ++j) {
			ARE::Function *f = j < b->nOriginalFunctions() ? b->OriginalFunction(j) : b->AugmentedFunction(j - b->nOriginalFunctions()) ;
			if (NULL == f) continue ;
			if (f->N() <= 0) continue ;
			for (int32_t k = 0 ; k < f->N() ; ++k) {
				int32_t argument = f->Argument(k) ;
				BucketElimination::Bucket *b_a = _Var2BucketMapping[argument] ;
				if (NULL == b_a) 
					return 1 ;
				f->SetArgumentsPermutationListValue(k, b_a->DistanceToRoot()) ;
				}
			}
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::DeleteMBEgeneratedTables(void)
{
	int32_t i ;

	for (i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		std::vector<BucketElimination::MiniBucket *> & MBs = b->MiniBuckets() ;
		for (BucketElimination::MiniBucket *mb : MBs) {
			ARE::Function & output_fn = mb->OutputFunction() ;
			output_fn.DestroyTableData() ;
			}
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::BuildSolution(void)
{
	if (_DeleteUsedTables) 
		return 1 ; // if tables not stored, cannot compute
	if (NULL == _Problem) 
		return 1 ;
	bool problem_is_max = true ;
	if (VAR_ELIMINATION_TYPE_MAX == _Problem->VarEliminationType()) 
		problem_is_max = true ;
	else if (VAR_ELIMINATION_TYPE_MIN == _Problem->VarEliminationType()) 
		problem_is_max = false ;
	else 
		return 1 ;

	// TODO : here we assume just one var per bucket
	int32_t *values = _Problem->ValueArray() ;
	for (int32_t i = 0 ; i < _nBuckets ; ++i) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		int32_t v = b->Var(0) ;
		if (values[v] >= 0) 
			continue ; // this var already has value; perhaps evidence value.
		int32_t K = _Problem->K(v) ;
		if (K <= 1) 
			{ values[v] = 0 ; continue ; }
		// given that context (of b) is instantiated, find best value for var(s) of this bucket.
		// don't need to include intermediate functions, since they don't contain this var, hence are invariant wrt picking value for this var.
		int32_t nF = b->nOriginalFunctions() + b->nAugmentedFunctions() ;
		int32_t best_value = -1 ; ARE_Function_TableType best_value_cost = VarEliminationDefaultValue() ;
		for (int32_t k = 0 ; k < K ; k++) {
			values[v] = k ;
			ARE_Function_TableType value = FnCombinationNeutralValue() ;
			for (int32_t j = 0 ; j < nF ; j++) {
				ARE::Function *f = j < b->nOriginalFunctions() ? b->OriginalFunction(j) : b->AugmentedFunction(j - b->nOriginalFunctions()) ;
				__int64 adr = f->ComputeFnTableAdr(values, _Problem->K()) ;
				ApplyFnCombinationOperator(value, f->TableEntry(adr)) ;
				}
			if (0 == k ? true : (problem_is_max ? value > best_value_cost : value < best_value_cost)) 
				{ best_value = k ; best_value_cost = value ; }
			}
		values[v] = best_value ;
		}

	// assign value 0 to singleton variables
	for (int32_t i = 0 ; i < _Problem->N() ; ++i) {
		if (0 != _Problem->Degree(i) || values[i] >= 0) continue ;
		values[i] = 0 ;
		}

	// compute value of the solution
	_Problem->AssignmentValue() = _AnswerFactor ;
	for (int32_t i = 0 ; i < _Problem->nFunctions() ; i++) {
		ARE::Function *f = _Problem->getFunction(i) ;
		// here we skip all const functions, since they should be part of _AnswerFactor
		if (f->N() > 0) {
			__int64 adr = f->ComputeFnTableAdr(values, _Problem->K()) ;
			ApplyFnCombinationOperator(_Problem->AssignmentValue(), f->TableEntry(adr)) ;
			}
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::PostComputationProcessing(void)
{
	// compose complete_elimination_answer
	ARE_Function_TableType v = _AnswerFactor ;
	ARE_Function_TableType v_all_except_first = v ; // this will be combined value of all roots, except for vars[0]
	for (int32_t i = _nBuckets - 1 ; i >= 0 ; i--) {
		BucketElimination::Bucket *b = _Buckets[i] ;
		if (0 == b->DistanceToRoot()) {
			std::vector<BucketElimination::MiniBucket *> & MBs = b->MiniBuckets() ;
			for (BucketElimination::MiniBucket *mb : MBs) {
				ARE::Function & output_fn = mb->OutputFunction() ;
				ApplyFnCombinationOperator(v, output_fn.ConstValue()) ;
				if (0 != i) 
					v_all_except_first = v ;
				}
			// also, just in case, add up root-bucket intermediate function (they should be const functions).
			for (int32_t j = 0 ; j < b->nIntermediateFunctions() ; j++) {
				ARE::Function *f = b->IntermediateFunction(j) ;
				if (NULL == f) continue ;
				if (f->N() > 0) continue ; // this should not happen
				ApplyFnCombinationOperator(v, f->ConstValue()) ;
				if (0 != i) 
					v_all_except_first = v ;
				}
			}
		}
	SetCompleteEliminationResult(v, _CurrentComputationBound) ;

	// compose vars[0] distribution
	int32_t v_0 = GetFirstQueryRootVariable() ; // *(b_0->Signature()) ;
	BucketElimination::Bucket *b_0 = MapVar2Bucket(v_0) ;
	if (_nBuckets > 0 && NULL != b_0) {
//		BucketElimination::Bucket *b_0 = _Buckets[0] ;
		int32_t k_0 = _Problem->K(v_0) ;
		_MarginalSingleVariableDistribution.reserve(k_0) ;
		if (_MarginalSingleVariableDistribution.capacity() == k_0) {
			_MarginalSingleVariableDistribution.resize(k_0, DBL_MAX) ;
			b_0->ComputeFirstVariableDistribution(_MarginalSingleVariableDistribution.data()) ;
			for (int32_t i = 0 ; i < k_0 ; i++) {
				ApplyFnCombinationOperator(_MarginalSingleVariableDistribution[i], v_all_except_first) ;
				}
			MarginalSingleVariableDistributionVar() = v_0 ;
			}
		}

	return 0 ;
}


int32_t BucketElimination::MBEworkspace::SaveReducedProblem(signed char ReducedProblemType, signed char ApproximationBound, std::string & fn, int32_t & nKeepVariables, std::vector<int32_t> & Old2NewVarMap, std::vector<ARE::Function*> & ReducedProblemFunctions)
{
	nKeepVariables = -1 ;
	if (_Problem->N() <= 0) 
		return 0 ;
	if (! _Problem->HasVarOrdering()) 
		return 1 ;

	int32_t i, j ;
	int32_t nFunctions ;
	double global_const ;
	int32_t res = 1 ;
	FILE *fpOUTPUTuai = NULL, *fpOUTPUTvarmapping = NULL ;
	std::string fn_uai ;

	const int32_t *varBTorder = _Problem->VarOrdering_VarList() ;
	const int32_t *var_to_BTorder_map = _Problem->VarOrdering_VarPos() ;

	if (Old2NewVarMap.capacity() < _Problem->N()) {
		Old2NewVarMap.reserve(_Problem->N()) ;
		if (Old2NewVarMap.capacity() < _Problem->N()) 
			return -1 ;
		}
	if (Old2NewVarMap.size() != _Problem->N()) 
		Old2NewVarMap.resize(_Problem->N()) ;
	for (i = 0 ; i < _Problem->N() ; i++) Old2NewVarMap[i] = -3 ;
	// Old2NewVarMap[i] == -3 means the variable is "eliminate"
	// Old2NewVarMap[i] == -2 means the variable is processed and "keep"
	// for each variable, we keep track if any of its children (descendants) are known to have width>3. if yes, this var is "eliminate"; it does not need to be considered for sophisticated query processing.

	if (ReducedProblemType <= 0) {
		for (i = 0 ; i < _Problem->N() ; i++) Old2NewVarMap[i] = -2 ;
		goto eliminate_vars_figured_out ;
		}
	else if (2 == ReducedProblemType) {
		for (i = 0 ; i < _Problem->N() ; ++i) Old2NewVarMap[i] = -3 ;
		std::vector<int32_t> & keep_list = _Problem->QueryVariables() ;
		for (i = 0 ; i < keep_list.size() ; ++i) Old2NewVarMap[keep_list[i]] = -2 ;
		goto eliminate_vars_figured_out ;
		}
	else {
		// process variables, last-to-first, generating output function for "eliminate" variables.
		std::vector<int32_t> key ; // keys are fn arity
		std::vector<int64_t> data ; // data a ptrs to fns
		std::vector<int32_t> helperArray ;
		int32_t iBound = 2 ; // with iBound==2, output function can have 2 arguments, since MB has width 3.
		for (i = _Problem->N()-1 ; i >= 0 ; i--) {
			int32_t v = varBTorder[i] ;
			BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
			BucketElimination::Bucket *parent_of_b = NULL != b ? b->ParentBucket() : NULL ;
			int32_t parent_of_v = NULL != parent_of_b ? parent_of_b->Var(0) : -1 ;
			if (-3 != Old2NewVarMap[v]) { // v is "keep"
				if (parent_of_v >= 0 ? -3 == Old2NewVarMap[parent_of_v] : false) Old2NewVarMap[parent_of_v] = -2 ;
				continue ;
				}
			if (NULL == b) 
				// variable most likely was eliminated from the problem; this var shoule be and stay on as "eliminate".
				{ continue ; }
			// check width of this fn
			if (b->Width() < 0) {
				b->ComputeSignature() ;
				if (b->Width() < 0) 
					goto done ;
				}
			// MB partitioning
			bool abandon = true ;
			int32_t res = b->CreateMBPartitioning(iBound, false, false, ApproximationBound, false, abandon, key, data, helperArray) ;
			if (abandon ? true : 0 != res) {
				Old2NewVarMap[v] = -2 ;
				if (parent_of_v >= 0 ? -3 == Old2NewVarMap[parent_of_v] : false) Old2NewVarMap[parent_of_v] = -2 ;
				continue ;
				}
			// generate output fn
			int64_t totalSumOutputFunctionsNumEntries = 0 ;
			if (0 != b->ComputeOutputFunctions(false, ApproximationBound, totalSumOutputFunctionsNumEntries)) 
				goto done ;
			}
		}

eliminate_vars_figured_out :
	// all entries in Old2NewVarMap[] should be -3 ("eliminate") or -2 ("keep").

	// count "keep" vars; create old2new mapping :
	//		only for "keep" variables. we want this property : if old indeces i<j, then old2new_map[i]<old2new_map[j].
	nKeepVariables = 0 ;
	for (i = 0 ; i < _Problem->N() ; i++) {
		if (-3 != Old2NewVarMap[i]) 
			Old2NewVarMap[i] = nKeepVariables++ ; // this guarantees property : if old indeces i<j, then old2new_map[i]<old2new_map[j].
		}
	// all entries in Old2NewVarMap[] should be -3 ("eliminate" var) or >=0 ("keep" var).

	// save all "keep" variables
	fn_uai = fn ; fn_uai += ".uai" ;
	fpOUTPUTuai = fopen(fn_uai.c_str(), "w") ;
	if (NULL == fpOUTPUTuai) 
		return 1 ;
	::fprintf(fpOUTPUTuai, "MARKOV\n%d\n", (int32_t) nKeepVariables) ;
	// print domain size of "keep" vars; compute global const = sum of output FNs of "eliminate" root buckets + sum of intermediate FNs in "keep" root buckets.
	global_const = AnswerFactor() ;
	for (i = j = 0 ; i < _Problem->N() ; i++) {
		int32_t v = varBTorder[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL != b ? 0 == b->DistanceToRoot() : false) {
			if (Old2NewVarMap[v] < 0) {
				std::vector<BucketElimination::MiniBucket *> & MBs = b->MiniBuckets() ;
				for (BucketElimination::MiniBucket *mb : MBs) {
					ARE::Function & output_fn = mb->OutputFunction() ;
					double const_value = output_fn.ConstValue() ;
					if (const_value == FnCombinationNeutralValue()) 
						continue ;
					ApplyFnCombinationOperator(global_const, const_value) ;
					}
				}
			for (int32_t k = 0 ; k < b->nIntermediateFunctions() ; k++) {
				ARE::Function *f = b->IntermediateFunction(k) ;
				if (NULL == f) continue ;
				if (f->N() > 0) continue ; // this should not happen
				double const_value = f->ConstValue() ;
				if (const_value == FnCombinationNeutralValue()) 
					continue ;
				ApplyFnCombinationOperator(global_const, const_value) ;
				}
			}
		if (Old2NewVarMap[v] >= 0) 
			::fprintf(fpOUTPUTuai, "%s%d", 0 == j++ ? "" : " ", (int32_t) _Problem->K(v)) ;
		}
	// compute/print number of functions
	nFunctions = (global_const == FnCombinationNeutralValue()) ? 0 : 1 ;
	for (i = 0 ; i < _Problem->N() ; i++) {
		int32_t v = varBTorder[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL == b) continue ; // variable most likely was eliminated from the problem
		if (Old2NewVarMap[v] >= 0) {
			nFunctions += b->nOriginalFunctions() + b->nAugmentedFunctions() ;
			}
		}
	fprintf(fpOUTPUTuai, "\n%d", (int32_t) nFunctions) ;
	// create a list of reduced problem funstions
	if (ReducedProblemFunctions.capacity() < nFunctions) {
		ReducedProblemFunctions.reserve(nFunctions) ;
		if (ReducedProblemFunctions.capacity() < nFunctions) 
			goto done ;
		}

	// print argument list for each function
	if (global_const != FnCombinationNeutralValue()) 
		fprintf(fpOUTPUTuai, "\n0") ;
	for (i = 0 ; i < _Problem->N() ; i++) {
		int32_t v = varBTorder[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL == b) continue ; // variable most likely was eliminated from the problem
		if (Old2NewVarMap[v] >= 0) {
			int32_t nFNtotal = b->nOriginalFunctions() + b->nAugmentedFunctions() ;
			for (j = 0 ; j < nFNtotal ; j++) {
				ARE::Function *f = (j < b->nOriginalFunctions()) ? b->OriginalFunction(j) : b->AugmentedFunction(j - b->nOriginalFunctions()) ;
				ReducedProblemFunctions.push_back(f) ;
				fprintf(fpOUTPUTuai, "\n%d", (int32_t) f->N()) ;
				for (int32_t k = 0 ; k < f->N() ; k++) {
					int64_t table_size = f->TableSize() ;
					if (table_size < 0) {
						table_size = f->ComputeTableSize() ;
						if (table_size < 0) continue ;
						}
					if (table_size > 0 && NULL == f->TableData()) 
						continue ;
					int32_t old_argument = f->Argument(k) ;
					int32_t new_argument = Old2NewVarMap[old_argument] ;
					fprintf(fpOUTPUTuai, " %d", new_argument) ;
					}
				}
			}
		}

	// print table of each function
	if (global_const != FnCombinationNeutralValue()) {
		if (_Problem->FunctionsAreConvertedToLogScale()) 
			global_const = pow(10.0, global_const) ;
		fprintf(fpOUTPUTuai, "\n\n1 %g", (double) global_const) ;
		}
	for (i = 0 ; i < _Problem->N() ; i++) {
		int32_t v = varBTorder[i] ;
		BucketElimination::Bucket *b = _Var2BucketMapping[v] ;
		if (NULL == b) continue ; // variable most likely was eliminated from the problem
		if (Old2NewVarMap[v] >= 0) {
			int32_t nFNtotal = b->nOriginalFunctions() + b->nAugmentedFunctions() ;
			for (j = 0 ; j < nFNtotal ; j++) {
				ARE::Function *f = (j < b->nOriginalFunctions()) ? b->OriginalFunction(j) : b->AugmentedFunction(j - b->nOriginalFunctions()) ;
				int64_t table_size = f->TableSize() ;
				if (0 == table_size) 
					fprintf(fpOUTPUTuai, "\n\n1\n%g", (double) f->ConstValue()) ;
				else {
					fprintf(fpOUTPUTuai, "\n\n%lld", (int64_t) table_size) ;
					int32_t last_var = f->Argument(f->N()-1) ;
					int32_t last_var_k = _Problem->K(last_var) ;
					for (int64_t idx = 0 ; idx < table_size ;) {
						fprintf(fpOUTPUTuai, "\n") ;
						for (int32_t k = 0 ; k < last_var_k ; k++, idx++) {
							double entry = (f->TableData())[idx] ;
							if (_Problem->FunctionsAreConvertedToLogScale()) 
								entry = pow(10.0, entry) ;
							fprintf(fpOUTPUTuai, "%s%g", (0 == k) ? "" : " ", (double) entry) ;
							}
						}
					}
				}
			}
		}
	fclose(fpOUTPUTuai) ; fpOUTPUTuai = NULL ;

	// save new_var -> old_var mapping
	{
	std::string fn_varmap(fn) ; fn_varmap += "_var_mapping.txt" ;
	fpOUTPUTvarmapping = fopen(fn_varmap.c_str(), "w") ;
	if (NULL == fpOUTPUTvarmapping) 
		return 1 ;
	fprintf(fpOUTPUTvarmapping, "%d", nKeepVariables) ;
	for (i = 0 ; i < _Problem->N() ; i++) {
		if (Old2NewVarMap[i] >= 0) 
			fprintf(fpOUTPUTvarmapping, "\n%d", i) ;
		}
	fclose(fpOUTPUTvarmapping) ; fpOUTPUTvarmapping = NULL ;
	}

	res = 0 ;
done :
	if (NULL != fpOUTPUTuai) fclose(fpOUTPUTuai) ;
	if (NULL != fpOUTPUTvarmapping) fclose(fpOUTPUTvarmapping) ;
	return res ;
}


int32_t BucketElimination::MBEworkspace::GenerateRandomBayesianNetworkStructure(int32_t N, int32_t K, int32_t P, int32_t C, int32_t ProblemCharacteristic)
{
return 1 ;
/*
	if (NULL == _Problem) 
		return 1 ;
	_Problem->Destroy() ;

	if (0 != _Problem->GenerateRandomUniformBayesianNetworkStructure(N, K, P, C, ProblemCharacteristic)) 
		{ return 1 ; }

	if (0 != _Problem->ComputeGraph()) 
		{ return 1 ; }
	int32_t nComponents = _Problem->nConnectedComponents() ;
	if (nComponents > 1) 
		{ return 1 ; }

	if (0 != _Problem->ComputeMinDegreeOrdering()) 
		{ return 1 ; }

	int32_t width = _Problem->MinDegreeOrdering_InducedWidth() ;
//	if (width < MinWidth || width > MaxWidth) 
//		{ return 1 ; }

	if (0 != _Problem->TestVariableOrdering(_Problem->MinDegreeOrdering_VarList(), _Problem->MinDegreeOrdering_VarPos())) 
		{ return 1 ; }

	if (0 != CreateBuckets(_Problem->N(), _Problem->MinDegreeOrdering_VarList(), true)) 
		{ return 1 ; }

	__int64 spaceO = _Problem->ComputeFunctionSpace() ;
	__int64 spaceN = ComputeNewFunctionSpace() ;
	__int64 space = spaceO + spaceN ;
//	if (space < MinMemory || space > MaxMemory) 
//		{ return 1 ; }

	return 0 ;
*/
}


int32_t BucketElimination::GenerateRandomBayesianNetworksWithGivenComplexity(int32_t nProblems, int32_t N, int32_t K, int32_t P, int32_t C, int32_t ProblemCharacteristic, __int64 MinSpace, __int64 MaxSpace)
{
return 1 ;
/*
	ARE::ARP p("test") ;
	// create BEEM workspace; this includes BE workspace.
	BEworkspace ws(NULL) ;
	ws.Initialize(p, NULL) ;

	time_t ttNow, ttBeginning ;
	time(&ttBeginning) ;
	time_t dMax = 3600 ;

	char s[256] ;
	int32_t i ;
	for (i = 0 ; i < nProblems ; ) {
		time(&ttNow) ;
		time_t d = ttNow - ttBeginning ;
		if (d >= dMax) 
			break ;

		ws.DestroyBuckets() ;
		ws.GenerateRandomBayesianNetworkStructure(N, K, P, C, ProblemCharacteristic) ;

		if (p.nConnectedComponents() > 1) 
			continue ;

		int32_t width = p.MinDegreeOrdering_InducedWidth() ;
		__int64 spaceO = p.ComputeFunctionSpace() ;
		__int64 spaceN = ws.ComputeNewFunctionSpace() ;
		__int64 space = spaceO + spaceN ;

		if (space < MinSpace || space > MaxSpace) 
			continue ;

		// generate nice name
		sprintf(s, "random-test-problem-%d-Space=%lld", (int32_t) ++i, space) ;
		p.SetName(s) ;

		// fill in original functions
		p.FillInFunctionTables() ;
		if (p.CheckFunctions()) 
			{ int32_t error = 1 ; }

		// save UAI08 format
		p.SaveUAI08("c:\\UCI\\problems") ;
		}

	return i ;
*/
}

