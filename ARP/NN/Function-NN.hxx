#ifndef FunctionNN_HXX_INCLUDED
#define FunctionNN_HXX_INCLUDED

#ifdef WINDOWS
#ifndef NOMINMAX
 #define NOMINMAX
#endif
#include "windows.h"
#endif

#include <climits>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>
#include <fstream>

#include "Utils/Mutex.h"
#include "Utils/Sort.hxx"
#include "Problem/Globals.hxx"
#include "Problem/Function.hxx"
#include "Problem/Workspace.hxx"
#include "Net.h"
#include "DATA_SAMPLES.h"
#include "NNConfig.h"

#ifdef INCLUDE_TORCH
#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/script.h>
#endif // INCLUDE_TORCH

namespace BucketElimination { class Bucket ; class MiniBucket ; }

namespace ARE
{

class ARP ;

class FunctionNN : public Function
{

public :

	bool _modelIsGood ;
	torch::jit::Module _model ;
	std::vector<torch::jit::IValue> _inputs ;
	at::Tensor _input ;
/*
		torch::jit::Module model
        std::vector<torch::jit::IValue> inputs ;
        at::Tensor input ....
*/

public :

	int32_t CreateNNtensor(void) ;

	virtual int32_t AllocateTableData(void)
	{
		// NO TABLE!!!
		return 0 ;
	}

	virtual ARE_Function_TableType TableEntryExNativeAssignment(int32_t* NativeAssignment, const int32_t* K) ;
	virtual ARE_Function_TableType TableEntryEx(int32_t* BEPathAssignment, const int32_t* K) ;
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

public :

	virtual void Initialize(Workspace *WS, ARP *Problem, int32_t IDX)
	{
		Function::Initialize(WS, Problem, IDX) ;
		//here we assume the structure is known just initialize the most simple FF neural network with no training.
	}

	void Destroy(void)
	{
		// TODO own stuff here...
		Function::Destroy() ;

	}

    void load_trained_model()
	{
		// TODO; do we need this fn?
    }

	FunctionNN(void)
		: Function(), 
		_modelIsGood(false)
	{
		// TODO own stuff here...
        _TableData = NULL;

	}

	FunctionNN(Workspace *WS, ARP *Problem, int32_t IDX)
		: 
		_modelIsGood(false)
	{
       Function(WS, Problem, IDX);
       _TableData = NULL;
	}

	virtual ~FunctionNN(void)
	{
		Destroy() ;
	}


} ;

inline Function *FunctionNNConstructor(void)
{
	return new FunctionNN;
}

} // namespace ARE

#endif // FunctionNN_HXX_INCLUDED
