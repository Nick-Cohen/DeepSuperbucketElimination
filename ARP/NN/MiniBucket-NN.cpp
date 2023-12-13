#define NOMINMAX

#include <stdlib.h>
#include <memory.h>
#include <exception>
#include <chrono>
#include <random>
#include <cstdlib>

#if defined WINDOWS || _WINDOWS
#include <windows.h>
#else
#include <sys/stat.h>
#endif // WINDOWS

#ifdef INCLUDE_TORCH

#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/script.h>

#endif // INCLUDE_TORCH

#include <Sort.hxx>
#include <Function.hxx>
#include <Bucket.hxx>
#include <MiniBucket.hxx>
#include <MBEworkspace.hxx>
#include "Utils/MersenneTwister.h"

#include <Function-NN.hxx>
#include "NNConfig.h"
#include "DATA_SAMPLES.h"

int32_t ARE::FunctionNN::CreateNNtensor(void)
{
	const int32_t* DomainSizes = _Problem->K() ;
	_OneHotArgVectorLength = -_nArgs ;
	for (int32_t i = 0 ; i < _nArgs ; ++i) {
		int32_t var = _Arguments[i] ;
		int32_t k = DomainSizes[var] ;
		_OneHotArgVectorLength += k ;
		}
	_input = torch::zeros({ 1, _OneHotArgVectorLength }) ;
	_inputs.push_back(_input) ;
	printf("\n1Hot vector created _OneHotArgVectorLength=%d", _OneHotArgVectorLength);
	printf("\n");
	return 0 ;
}

ARE_Function_TableType ARE::FunctionNN::TableEntryExNativeAssignment(int32_t* NativeAssignment, const int32_t* DomainSizes)
{
	ARE_Function_TableType out_value = 0.0 ;

	if (!_modelIsGood)
		return out_value ;

	int32_t res = FillInOneHotNNinput_wrtNativeAssignemnt(_input, NativeAssignment, DomainSizes) ;
	if (0 != res)
		return out_value;
	at::Tensor output = _model.forward(_inputs).toTensor();
	void* ptr = output.data_ptr();
	float* e = (float*)ptr;
	out_value = *e;
//	out_value = output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) ;

	return out_value;
}

ARE_Function_TableType ARE::FunctionNN::TableEntryEx(int32_t* BEPathAssignment, const int32_t* DomainSizes)
{
	ARE_Function_TableType out_value = 0.0;

	if (!_modelIsGood)
		return out_value;

	int32_t res = FillInOneHotNNinput_wrtPermutationList(_input, BEPathAssignment, DomainSizes);
	if (0 != res)
		return out_value;
	at::Tensor output = _model.forward(_inputs).toTensor();
	void* ptr = output.data_ptr();
	float* e = (float*)ptr;
	out_value = *e;
	//	out_value = output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) ;

	return out_value;
}

int32_t DeleteFile(const char *filename)
{
#if defined WINDOWS || _WINDOWS
	BOOL res = DeleteFileA(filename) ;
	return res ? 0 : 1 ;
#else
	int32_t res = std::remove(filename) ;
	return res ;
#endif
	return 1 ;
}


int32_t CheckFileExists(const char *filename)
{
#if defined WINDOWS || _WINDOWS
	bool file_exists = INVALID_FILE_ATTRIBUTES != GetFileAttributesA(filename) ;
	return file_exists ? 0 : 1 ;
#else
	struct stat buffer ;
	int res = stat(filename, &buffer) ;
	return res ;
#endif
	return 1 ;
}

int32_t WaitForFile(const char *filename, int64_t TimeoutInMSec, int64_t SleepTimeInMSec /* 100 */, int64_t & dtWaitPeriod)
{
	dtWaitPeriod = 0 ;
	int32_t resFileExists = CheckFileExists(filename) ;
	if (0 == resFileExists)
		return 0 ;
	int64_t tStart = ARE::GetTimeInMilliseconds() ;
	while (true) {
		SLEEP(SleepTimeInMSec) ;
		int64_t tNow = ARE::GetTimeInMilliseconds() ;
		dtWaitPeriod = tNow - tStart ;
		resFileExists = CheckFileExists(filename) ;
		if (0 == resFileExists)
			return 0 ;
		if (dtWaitPeriod > TimeoutInMSec) {
			return 1 ;
			}
		}

	return 1 ;
}


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


//    int32_t res = cntx._MB->SampleOutputFunction(varElimOp, cntx._nSamples, cntx._idx, cntx._nFeaturesPerSample, cntx._Samples_signature, cntx._Samples_values, cntx._min_value, cntx._max_value, cntx._sample_sum) ;

int32_t BucketElimination::MiniBucket::ComputeOutputFunction_NN(int32_t varElimOperator, ARE::Function *FU, ARE::Function *fU, double WMBEweight)
{
    MBEworkspace *bews = dynamic_cast<MBEworkspace*>(_Workspace) ;
    if (NULL == bews)
        return ERRORCODE_generic ;
    ARE::ARP *problem = bews->Problem() ;
    if (NULL == problem)
        return ERRORCODE_generic ;
	int32_t varElimOp = bews->VarEliminationType() ;

    ARE::FunctionNN *fNN = dynamic_cast<ARE::FunctionNN *>(_OutputFunction) ;
    if (NULL == fNN)
        return 1 ;

	bool data_is_log_space = problem->FunctionsAreConvertedToLogScale() ;

	// for testing, saving superbuckets output for xml file...
	if (false) {
		int32_t res = ComputeOutputFunction(varElimOperator, true, false, nullptr, nullptr, DBL_MAX) ;
		int32_t done = 1 ;
		}

	// generate samples...
	int64_t nSamples = bews->maxNumNNsamples(); if (nSamples < 1) nSamples = 1;
	int32_t nFeaturesPerSample = -1 ; //
	std::unique_ptr<int16_t[]> samples_signature ; 
	std::unique_ptr<float[]> samples_values ;
	float samples_min_value, samples_max_value, samples_sum ;
	{
		std::random_device rd ;
		uint32_t seed = rd() ;
		int32_t resSampling = SampleOutputFunction(varElimOp, nSamples, seed, nFeaturesPerSample, samples_signature, samples_values, samples_min_value, samples_max_value, samples_sum) ;
	}

	// write samples into xml file...
	std::string sFNsamples, sFNnn /* nn.jit file we expect to get back */, sFNsignalling ;
	std::unique_ptr<char[]> sBUF(new char[1024]);
	if (nullptr == sBUF)
		return 1;
	char* buf = sBUF.get();
	{
		std::string s, sPrefix, sPostFix ;
		GenerateSamplesXmlFilename(nullptr, sFNsamples, sFNnn, sFNsignalling, sPrefix, sPostFix, nSamples, samples_min_value, samples_max_value, samples_sum) ;
		FILE *fp = fopen(sFNsamples.c_str(), "w") ;
		fwrite(sPrefix.c_str(), 1, sPrefix.length(), fp) ;
		for (int32_t iS = 0 ; iS < nSamples ; ++iS) {
			int16_t *sample_signature = samples_signature.get() + iS * nFeaturesPerSample ;
			s = "\n   <sample signature=\"" ;
			for (int32_t iSig = 0 ; iSig < nFeaturesPerSample ; ++iSig) {
				if (iSig > 0) s += ';' ;
				sprintf(buf, "%d", (int) sample_signature[iSig]) ; s += buf ;
				}
			sprintf(buf, "\" value=\"%g\"/>", (double) samples_values[iS]) ; s += buf ;
			fwrite(s.c_str(), 1, s.length(), fp) ;
			}
		fwrite(sPostFix.c_str(), 1, sPostFix.length(), fp) ;
		fclose(fp) ;
	}

//sFNnn = "C:\\UCI\\DeepSuperbucketElimination-Nick-github\\problems\\nn-202-cpu.jit";
//sFNnn = "C:\\UCI\\BESampling\\nn-52;63.jit";
//sFNnn = "C:\\UCI\\BESampling\\nn-57;66;75.jit";
//sFNnn = "C:\\UCI\\BESampling\\nn-25;36;47.jit";
//sFNsignalling = "C:\\UCI\\DeepSuperbucketElimination-Nick-github\\problems\\ready-202.jit";

	// construct command line string
	sprintf(buf, "python3 /home/cohenn1/SDBE/Super_Buckets/ARP/NN/NN_Train.py --samples \"%s\" --nn_path \"%s\" --done_path \"%s\"", sFNsamples.c_str(), sFNnn.c_str(), sFNsignalling.c_str());
	// just in case delete signalling file...
//	DeleteFile(sFNsignalling.c_str());
	// launch python training script
	printf("\nWILL RUN COMMAND LINE : \n   ");
	printf(buf);
	printf("\nabc");
	std::system("pwd");
	std::system(buf);

	int64_t nnWaitTimeoutInMsec = 86400000 ; // 86400000 = 24hours
	int64_t SleepTimeInMSec = 100, dtWaitPeriod = -1 ;
	int32_t resFileWait = WaitForFile(sFNsignalling.c_str(), nnWaitTimeoutInMsec, SleepTimeInMSec, dtWaitPeriod) ;
	if (0 == resFileWait) {
		printf("\nOK : found file %s", sFNnn.c_str());
		try {
			fNN->_model = torch::jit::load(sFNnn.c_str());
			}
		catch (...) {
			printf("\nEXCEPTION : %s", sFNnn.c_str());
			printf("");
			exit(98);
			}
		printf("\nOK : loaded file %s", sFNnn.c_str());
		fNN->_modelIsGood = true ;
        fNN->CreateNNtensor() ;
		printf("\nOK : created tensor %s", sFNnn.c_str());
		printf("\nOK : all done %s", sFNnn.c_str());
		}
	else {
		printf("\nERROR : failed to find file %s", sFNnn.c_str());
		exit(99);
		}

    return 0 ;
}





