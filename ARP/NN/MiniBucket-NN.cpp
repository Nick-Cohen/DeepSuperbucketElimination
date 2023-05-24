#define NOMINMAX

#include <stdlib.h>
#include <memory.h>
#include <exception>
#include "NNConfig.h"
#include <chrono>

#include <Sort.hxx>
#include <Function.hxx>
#include <Bucket.hxx>
#include <MiniBucket.hxx>
#include <MBEworkspace.hxx>
#include "Utils/MersenneTwister.h"

#include <Function-NN.hxx>

#include "DATA_SAMPLES.h"

//#include <iostream> //delete
//#include <fstream> //delete
static MTRand RNG ;

//Config_NN global_config ;

#ifdef INCLUDE_TORCH
torch::optim::AdamOptions::AdamOptions(double lr, double eps) : lr_(lr), eps_(eps) {}
#endif // INCLUDE_TORCH

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int32_t BucketElimination::MiniBucket::ComputeOutputFunction_NN(int32_t varElimOperator, ARE::Function *FU, ARE::Function *fU, double WMBEweight)
{
    // 4-14-23 NC: Contents found in MiniBucket-NN-original.cpp, trying to get Superbuckets working
    cout << "\nCalled computeoutputfunctionNN" ;
    exit(134);
    return 1;
   }




