#include <stdlib.h>
#include <memory.h>

#include <Function.hxx>
#include <Bucket.hxx>
#include <MBEworkspace.hxx>
#include "Utils/ThreadPool.hxx"

#if defined WINDOWS || _WINDOWS
unsigned int __stdcall BucketSamplingWorkerThreadFn(void *X) 
#elif defined (LINUX)
void *BucketSamplingWorkerThreadFn(void *X)
#endif 
{
	ARE::ThreadPoolThreadContext *cntx = (ARE::ThreadPoolThreadContext *) X ;

	int32_t varElimOp = VAR_ELIMINATION_TYPE_SUM ;

	while (! cntx->_StopAndExit) {
		if (nullptr == cntx->_MB || cntx->_nSamples <= 0 || cntx->_WorkDone) 
			{ SLEEP(50) ; continue ; }
		int32_t res = cntx->_MB->SampleOutputFunction(varElimOp, cntx->_nSamples, cntx->_idx, cntx->_nFeaturesPerSample, cntx->_Samples_signature, cntx->_Samples_values, cntx->_min_value, cntx->_max_value, cntx->_sample_sum) ;
		cntx->_WorkDone = true ;
		}

done :
	cntx->_ThreadHandle = 0 ;
#if defined WINDOWS || _WINDOWS
	_endthreadex(0) ;
	return 0  ;
#else
	return NULL ;
#endif
}

