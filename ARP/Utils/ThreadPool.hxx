#ifndef ThreadPool_HXX_INCLUDED
#define ThreadPool_HXX_INCLUDED

#include <climits>
#include <stdlib.h>
#include <string>
#include <vector>
#include <memory>

#if defined WINDOWS || _WINDOWS
#include <process.h>    /* _beginthread, _endthread */
#endif // WINDOWS

#include "Utils/Mutex.h"
#include "Utils/Sort.hxx"
#include "Utils/MiscUtils.hxx"
#include "Problem/Globals.hxx"
#include "Problem/Workspace.hxx"

#if defined WINDOWS || _WINDOWS
typedef unsigned int (__stdcall *pThreadPoolWorkerThreadFn)(void *X) ;
#elif defined (LINUX)
typedef void *(*pThreadPoolWorkerThreadFn)(void *X) ;
#endif 

class BucketElimination::Bucket ;
class BucketElimination::MiniBucket ;

namespace ARE
{

class ThreadPool ;

class ThreadPoolThreadContext
{
public :
	ThreadPool *_TP ;
	int32_t _idx ; // wrt TP
	bool _StopAndExit ;
#if defined WINDOWS || _WINDOWS
	uintptr_t _ThreadHandle ;
#elif defined (LINUX)
	pthread_t _ThreadHandle ;
#endif
	// task info
	BucketElimination::MiniBucket *_MB ;
	int64_t _nSamples ;
	bool _WorkDone ;
	// data computed
	int32_t _nFeaturesPerSample ; // = _OutputFunction->N()
	std::unique_ptr<int16_t[]> _Samples_signature ; // use .reset() to free memory... size = _nSamples * _nFeaturesPerSample
	std::unique_ptr<float[]> _Samples_values ; // use .reset() to free memory... size = nSamples; these are in log-space...
	float _min_value, _max_value, _sample_sum ; // these are in log-space...
public :
	inline int16_t *Signature(int32_t i) { return _Samples_signature.get() + i*_nFeaturesPerSample ; }
	inline float *Label(int32_t i) { return _Samples_values.get() + i ; }
public :
	ThreadPoolThreadContext(void) 
		: _TP(nullptr), _idx(-1), _StopAndExit(true), _ThreadHandle(0), _MB(nullptr), _nSamples(0), _WorkDone(false) 
	{ }
} ;

class ThreadPool
{
public :
	int32_t _nThreads ;
	pThreadPoolWorkerThreadFn _pThredFn ;
	std::unique_ptr<ThreadPoolThreadContext[]> _Tasks ;

public :
	virtual int32_t Create(void)
	{
		if (_nThreads <= 0) 
			return 1 ;
		if (_nThreads > 1000) 
			_nThreads = 1000 ;
		if (nullptr == _pThredFn) 
			return 2 ;
		std::unique_ptr<ThreadPoolThreadContext[]> tasks(new ThreadPoolThreadContext[_nThreads]) ;
		if (nullptr == tasks) 
			return 3 ;
		_Tasks.reset() ;
		_Tasks = std::move(tasks) ;
		ThreadPoolThreadContext *tsks = _Tasks.get() ;
		for (int32_t i = 0 ; i < _nThreads ; ++i) {
			tsks[i]._ThreadHandle = 0 ;
			tsks[i]._TP = this ;
			tsks[i]._idx = i ;
			tsks[i]._StopAndExit = false ;
			tsks[i]._MB = nullptr ;
			tsks[i]._nSamples = -1 ;
			tsks[i]._WorkDone = false ;
			}
		for (int32_t i = 0 ; i < _nThreads ; ++i) {
#if defined WINDOWS || _WINDOWS
			tsks[i]._ThreadHandle = _beginthreadex(NULL, 0, _pThredFn, &(tsks[i]), 0, NULL) ;
			if (0 == tsks[i]._ThreadHandle) 
				return 99 ;
#else
			pthread_create(&(tsks[i]._ThreadHandle), NULL, _pThredFn, &(tsks[i])) ;
			if (0 != tsks[i]._ThreadHandle) 
				{ tsks[i]._ThreadHandle = 0 ; return 99 ; }
#endif
			}
		return 0 ;
	}
	virtual void StopThreads(int64_t TimeoutInMilliseconds) 
	{
		if (_nThreads <= 0 || nullptr == _Tasks) 
			return ;
		if (TimeoutInMilliseconds < 1)
			TimeoutInMilliseconds = 1 ;
		else if (TimeoutInMilliseconds > 86400000)
			TimeoutInMilliseconds = 86400000 ;
		ThreadPoolThreadContext *tsks = _Tasks.get() ;
		for (int32_t i = 0 ; i < _nThreads ; ++i) tsks[i]._StopAndExit = true ;
		// wait for threads to stop; if out of time, kill them...
		int64_t tStart = ARE::GetTimeInMilliseconds() ;
		while (true) {
			SLEEP(25) ;
			int64_t tNow = ARE::GetTimeInMilliseconds() ;
			int64_t dt = tNow - tStart ;
			bool out_of_time = dt > TimeoutInMilliseconds ;
			int32_t nThreadsRunning = 0 ;
			for (int32_t i = 0 ; i < _nThreads ; ++i) {
				if (0 == tsks[i]._ThreadHandle) 
					continue ;
				++nThreadsRunning ;
				if (! out_of_time) 
					continue ;
#if defined WINDOWS || _WINDOWS
				TerminateThread((HANDLE) tsks[i]._ThreadHandle, 0) ;
				CloseHandle((HANDLE) tsks[i]._ThreadHandle) ;
				tsks[i]._ThreadHandle = 0 ;
#else
				pthread_exit(&(tsks[i]._ThreadHandle)) ;
#endif
				}
			if (out_of_time || 0 == nThreadsRunning) 
				break ; // all threads killed...
			}
	}
	virtual void Destroy(void)
	{
		StopThreads(100) ;
		_Tasks.reset() ;
	}
	ThreadPool(void)
		: 
		_nThreads(0), 
		_pThredFn(nullptr)
	{
	}
	ThreadPool(int32_t nThreads)
		:
		_nThreads(nThreads), 
		_pThredFn(nullptr)
	{
	}
	virtual ~ThreadPool(void)
	{
		Destroy() ;
	}
} ;

extern ARE::ThreadPool BE_sample_generation_tp ;

} // namespace ARE

#endif // ThreadPool_HXX_INCLUDED
