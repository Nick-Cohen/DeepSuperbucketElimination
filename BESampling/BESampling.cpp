#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <fstream>
#include <thread>
#include <time.h>
#include <cmath>
#include <limits>

#include <iostream>

#if defined WINDOWS || _WINDOWS
#include <process.h>    /* _beginthread, _endthread */
#else
#include <sys/time.h>
#endif // WINDOWS

#include <torch/torch.h>
#include <torch/nn/module.h>
#include <torch/script.h>
template <typename T>
void pretty_print(const std::string& info, T&& data)
{
	std::cout << info << std::endl;
	std::cout << data << std::endl << std::endl;
}

#include "CVO/VariableOrderComputation.hxx"
#include "BE/Bucket.hxx"
#include "BE/MBEworkspace.hxx"
#include "Utils/ThreadPool.hxx"

#include <NNConfig.h>
Config_NN global_config;

static MTRand RNG ;

#ifdef _MSC_VER
#ifndef _WIN64 
// error - this project is meant as 64-bit...
#endif
#ifndef _M_X64
// error - this project is meant as 64-bit...
#endif
#endif 

//#define PERFORM_SINGLTON_CONSISTENCY_CHECK

#if defined WINDOWS || _WINDOWS
#else
double get_wall_time() {
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		//  Handle error
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
ARE::FnConstructor deepMBE_mboutputfncnstrctr(BucketElimination::MiniBucket & MB)
{
    // TODO : implement logic to decide what kind of output fn to use for the given MB...
//    if(MB->Width() == 20)
//            return ARE::FunctionNNConstructor;
    return ARE::FunctionConstructor ;
}
// #include <torch/torch.h>
double get_cpu_time() {
	return (double)clock() / CLOCKS_PER_SEC;
}
#endif // WINDOWS

double logabsdiffexp10(double num1, double num2) {
	if (num1 == num2) {
		return -std::numeric_limits<double>::infinity();
	}
	num1 = num1 * std::log(10.0);
	num2 = num2 * std::log(10.0);
	double num_max = num1 < num2 ? num2 : num1 ; // std::max(num1, num2);
	double num_min = num1 > num2 ? num2 : num1 ; // std::min(num1, num2);
	double ans = (num_min + std::log(expm1(num_max - num_min))) / std::log(10.0) ;
	// std::cout << ans << std::endl;
	return ans;
}

int32_t main(int32_t argc, char* argv[])
{
/*
#ifdef INCLUDE_TORCH
	torch::Tensor tensor2 = torch::eye(3);
    std::cout << tensor2 << std::endl;
#endif // INCLUDE_TORCH
*/

	if (false) {
//		torch::Tensor tensor = torch::eye(3);
//		pretty_print("Eye tensor: ", tensor);

		std::string fn_samples("C:\\UCI\\DeepSuperbucketElimination-Nick-github\\BESampling\\samples-39;104;141.xml") ;
		std::string fn_weights("C:\\UCI\\DeepSuperbucketElimination-Nick-github\\ARP\\NN\\nn-39;104;141.jit") ;
// signature="0;0;0;0;0;0;0;0;1;0;0;0;0" value="-6.71354" predicted value = -6.764173984527588
// <samples n="18432" nFeaturesPerSample="13" outputfnscope="85;89;93;97;98;101;134;177;179;181;183;185;187" outputfnvariabledomainsizes="2;2;2;2;2;2;2;3;3;2;2;2;2" datainlogspace="Y">
		torch::jit::Module model = torch::jit::load(fn_weights.c_str()) ;

		std::vector<torch::jit::IValue> inputs ;
		at::Tensor input = torch::zeros({ 1, 15 }) ;
		int l = input.numel() ;
		inputs.push_back(input);
		void *ptr = input.data_ptr() ;
		((float*)ptr)[9] = 1 ;
		torch::jit::IValue & ip = inputs[0] ;
//		torch::zeros({ 1, 15 }));
//		inputs.push_back(torch::ones({ 1, 15 })) ;

//at::Tensor t1 = torch::zeros({1, 1}) ;
//std::cout << "one = " << t1 << "\n";


//		i.index_put_({ torch::tensor({ 1,9 }) }, t1);

//		at::Tensor & i9 = i.slice(1, 9, 9, 1) ;
//		i9.set_data(t1);
//		i.slice(0, 9, 9, 1).set_data(t1);
//		auto Ti = i.flatten();
//		int len = Ti.size(0);

std::cout << "input = " << ip << "\n" ;
bool isT = ip.isTensor() ;
		// set [9] to '1'
		// ...

		at::Tensor output = model.forward(inputs).toTensor();
		std::cout << "value is : " << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

		return 0;
	}

#if defined WINDOWS || _WINDOWS
	time_t wall0 ;
	time(&wall0) ;
#else
	double wall0 = get_wall_time();
	double cpu0 = get_cpu_time();
	double wall1, cpu1;
#endif // WINDOWS

	int32_t nParams = argc ;
	unsigned long randomGeneratorSeed = 0 ;
	std::string problem_filename, evidence_filename, vo_filename ;
	int32_t nArgs = (nParams - 1) >> 1 ;
	int32_t time_limit = 60;
	int32_t print_period = 1;
	bool findPracticalVariableOrder = true ;
	int32_t iB = -1 ;
	int64_t EClimit = -1 ;
	ARE::VarElimOrderComp::ObjectiveToMinimize objCodePrimary = ARE::VarElimOrderComp::Width, objCodeSecondary = ARE::VarElimOrderComp::None ;
	std::string algorithm ;
	double exactZ = 0.0;
	int32_t v2sample = -1, nsamples = -1 ;
	int32_t vQuery = -1 ;
	bool log_console = false ;

	// printf("nArgs=%d nParams=%d", nArgs, nParams) ;
	if (nParams <= 1) {
		printf("\nLIST OF ARGUMENTS :") ;
		printf("\n   -s = random seed") ;
		printf("\n   -fUAI = UAI format file that defines the problem") ;
		printf("\n   -fEV = evidence file; optional") ;
		printf("\n   -fVO = variable order file") ;
		printf("\n   -iB = i-bound; optional; if not given, max i-bound will be used so that table memory is 1GB") ;
		printf("\n   -EClimit = limit on the elimination complexity = number of elimination combinations; default = single var elimination only...") ;
		printf("\n   -vQ = query variable; optional") ;
		printf("\n   -query = optional; default is product-sum") ;
		return 1 ;
		}
	if (1 + 2 * nArgs != nParams) {
		printf("\nBAD COMMAND LINE; will exit ...") ;
		return 1 ;
		}

	std::string sQuery("product-sum") ;
	for (int32_t i = 0 ; i < nArgs; i++) {
		std::string sArgID = (NULL != argv[1 + 2 * i] ? argv[1 + 2 * i] : "") ;
		std::string sArg = (NULL != argv[2 * (i + 1)] ? argv[2 * (i + 1)] : "") ;
		if (0 == stricmp("-s", sArgID.c_str()))
			randomGeneratorSeed = std::strtoul(sArg.c_str(), NULL, 0) ;
		else if (0 == stricmp("-fUAI", sArgID.c_str()))
			problem_filename = sArg ;
		else if (0 == stricmp("-fEV", sArgID.c_str()))
			evidence_filename = sArg ;
		else if (0 == stricmp("-fVO", sArgID.c_str()))
			vo_filename = sArg ;
		else if (0 == stricmp("-printPeriod", sArgID.c_str()))
			print_period = atoi(sArg.c_str());
		else if (0 == stricmp("-t", sArgID.c_str()))
			time_limit = atoi(sArg.c_str());
		else if (0 == stricmp("-iB", sArgID.c_str()))
			iB = atoi(sArg.c_str());
		else if (0 == stricmp("-EClimit", sArgID.c_str()))
			EClimit = std::stoll(sArg.c_str()) ; // _atoi64(sArg.c_str());
		else if (0 == stricmp("-query", sArgID.c_str()))
			sQuery = sArg ;
//		else if (0 == stricmp("-Verbose", sArgID.c_str()))
//			verbose = atoi(sArg.c_str()) > 0 ;
		else if (0 == stricmp("-v2sample", sArgID.c_str()))
			v2sample = atoi(sArg.c_str()) ;
		else if (0 == stricmp("-nsamples", sArgID.c_str()))
			nsamples = atoi(sArg.c_str()) ;
		else if (0 == stricmp("-a", sArgID.c_str()))
			algorithm = sArg ;
		else if (0 == stricmp("-fpvo", sArgID.c_str()))
			findPracticalVariableOrder = '1' == sArg[0] || 'y' == sArg[0] || 'Y' == sArg[0] ;
		else if (0 == stricmp("-O1", sArgID.c_str()))
			objCodePrimary = (ARE::VarElimOrderComp::ObjectiveToMinimize) atoi(sArg.c_str()) ;
		else if (0 == stricmp("-O2", sArgID.c_str()))
			objCodeSecondary = (ARE::VarElimOrderComp::ObjectiveToMinimize) atoi(sArg.c_str()) ;
		else if (0 == stricmp("-exactZ", sArgID.c_str()))
			exactZ = atof(sArg.c_str());
		else if (0 == stricmp("-vQ", sArgID.c_str()))
			vQuery = atoi(sArg.c_str());
		else if (0 == stricmp("-verbose", sArgID.c_str()))
			log_console = '1' == sArg[0] || 'y' == sArg[0] || 'Y' == sArg[0] ;
        else if (0 == stricmp("-h_dim",sArgID.c_str() ))
            global_config.h_dim = atoi(sArg.c_str());
        else if (0 == stricmp("-batch_size",sArgID.c_str() ))
            global_config.batch_size = atoi(sArg.c_str());
        else if (0 == stricmp("-n_epochs",sArgID.c_str() ))
            global_config.n_epochs = atoi(sArg.c_str());
        else if (0 == stricmp("-lr",sArgID.c_str() ))
            global_config.lr = atof(sArg.c_str());
        else if (0 == stricmp("-sample_size",sArgID.c_str() ))
            global_config.sample_size = atoi(sArg.c_str());
        else if (0 == stricmp("--out_file",sArgID.c_str() ))
            global_config.out_file = sArg;
        else if (0 == stricmp("--out_file2",sArgID.c_str() ))
            global_config.out_file2 = sArg;
        else if (0 == stricmp("--network",sArgID.c_str() ))
            global_config.network = sArg;
        else if (0 == stricmp("--sampling_method",sArgID.c_str() ))
            global_config.s_method = sArg;
        else if (0 == stricmp("--width_problem",sArgID.c_str() ))
            global_config.width_problem = atoi(sArg.c_str());
        else if (0 == stricmp("--loss_weight", sArgID.c_str()))
            global_config.loss_weight = atof(sArg.c_str());
        else if (0 == stricmp("--loss_weight_mse", sArgID.c_str()))
            global_config.loss_weight_mse = atof(sArg.c_str());
        else if (0 == stricmp("--train_stop", sArgID.c_str()))
            global_config.train_stop = sArg;
        else if (0 == stricmp("--stop_iter", sArgID.c_str()))
            global_config.stop_iter = atoi(sArg.c_str());
        else if (0 == stricmp("--loss",sArgID.c_str() ))
            global_config.loss = sArg;
        else if (0 == stricmp("--do_sum", sArgID.c_str()))
            global_config.do_sum = atoi(sArg.c_str());
        else if (0 == stricmp("--l2_lambda", sArgID.c_str()))
            global_config.l2_lambda = atof(sArg.c_str());
        else if (0 == stricmp("--l2", sArgID.c_str()))
            global_config.l2 = atoi(sArg.c_str());
        else if (0 == stricmp("--do_weight", sArgID.c_str()))
            global_config.do_weight = atoi(sArg.c_str());
        else if (0 == stricmp("--n_layers", sArgID.c_str()))
            global_config.n_layers = atoi(sArg.c_str());
        else if (0 == stricmp("--var_dim", sArgID.c_str()))
            global_config.var_dim = atoi(sArg.c_str());
        else if (0 == stricmp("--l2_loss", sArgID.c_str()))
            global_config.l2_loss = atoi(sArg.c_str());
        else if (0 == stricmp("--var_samples", sArgID.c_str()))
            global_config.var_samples = atoi(sArg.c_str());
        else if (0 == stricmp("--epsilon",sArgID.c_str() ))
            global_config.epsilon = atof(sArg.c_str());
        else if (0 == stricmp("--lb",sArgID.c_str() ))
            global_config.lb = atoi(sArg.c_str());
        else if (0 == stricmp("--up",sArgID.c_str() ))
            global_config.up = atoi(sArg.c_str());
        else if (0 == stricmp("--input_norm",sArgID.c_str() ))
            global_config.input_norm = atoi(sArg.c_str());
		}

    std::string o_file =  global_config.out_file + "plot.txt";
    printf("Writing to -- %s", o_file.c_str());

	int32_t maxNumProcessorThreads = std::thread::hardware_concurrency() ; // this requires c++11
	if (maxNumProcessorThreads <= 0)
		maxNumProcessorThreads = 1 ;

	if (0 == problem_filename.length() || 0 == vo_filename.length())
		return 1 ;

	ARE::ARP p("mbe") ;
//	p.SetOperators(FN_COBINATION_TYPE_PROD, VAR_ELIMINATION_TYPE_SUM) ;
	int32_t res_setQ = p.SetOperators(sQuery.c_str()) ;
	if (0 != res_setQ) {
		printf("\nInvalid query(%s)", sQuery.c_str()) ;
		return 1 ;
		}
	int32_t resLoad = p.LoadFromFile(problem_filename) ;
	if (0 != resLoad)
		return 10 ;
	int32_t resPostCA = p.PerformPostConstructionAnalysis() ;
	if (0 != resPostCA)
		return 11 ;
	printf("\nproblem_filename = %s, order file = %s", problem_filename.c_str(), vo_filename.c_str()) ;
	printf("\niB = %d, EClimit = %d", iB, EClimit) ;
	printf("\nProblem loaded : nVars=%d minK=%d maxK=%d avgK=%g", p.N(), p.minK(), p.maxK(), p.avgK()) ;
	if (iB > p.N())
		iB = p.N() ;
	if (vQuery >= 0 && vQuery < p.N()) 
		p.AddQueryVariable(vQuery) ;

	int32_t nEV = -1 ;
	if (evidence_filename.length() > 0) {
		int32_t resLoadE = p.LoadFromFile_Evidence(evidence_filename, nEV) ;
		if (0 != resLoadE)
			return 20 ;
		int32_t resElimE = p.EliminateEvidence() ;
		if (0 != resElimE)
			return 21 ;
		}

	int32_t resELDV = p.EliminateSingletonDomainVariables() ;
	if (0 != resELDV)
		return 30 ;

	// load vo
	{
		FILE *fpVO = fopen(vo_filename.c_str(), "rb") ;
		if (NULL == fpVO)
			return 40 ;
		// get file size
		fseek(fpVO, 0, SEEK_END) ;
		int32_t filesize = ftell(fpVO) ;
		fseek(fpVO, 0, SEEK_SET) ;
		char *buf = new char[filesize + 1] ;
		if (NULL == buf)
		{
			fclose(fpVO) ; return 41 ;
		}
		int32_t L = fread(buf, 1, filesize, fpVO) ;
		fclose(fpVO) ;
		if (filesize != L)
		{
			delete[] buf ; return 42 ;
		}
		buf[filesize] = 0 ;
		int32_t OrderType = 0 ; // 0=OrderType means variable elimination order. 1=OrderType means variable bucket-tree order.
		int32_t resLoadVO = p.LoadVariableOrderingFromBuffer(OrderType, 1, true, buf) ;
		delete[] buf ;
		if (0 != resLoadVO)
			return 43 ;
	}

#ifdef PERFORM_SINGLTON_CONSISTENCY_CHECK
	int32_t nNewSingletonDomainVariables = 0, nVarsWithReducedDomainSize = 0 ;
	int32_t res_SC = p.ComputeSingletonConsistency(nNewSingletonDomainVariables, nVarsWithReducedDomainSize) ;
	if (res_SC < 0) {
		// problem inconsistent
	}
#endif
	BucketElimination::MBEworkspace ws ;
	int32_t delete_used_tables = 0 ; // keep tables in  memory (0) so that we can sample from them...
	int32_t resWSinit = ws.Initialize(p, true, NULL, delete_used_tables) ;
	if (0 != resWSinit)
		return 50 ;
	// if we have query variables, don't delete intermediate tables...
	// TODO : we can delete most tables, except the ones at the very end (at the top of the bucket tree)...
	if (p.QueryVariables().size() > 0) 
		ws.SetDeleteUsedTables(false) ;

	bool isAO = true ;
	int32_t resCB = ws.CreateBuckets(isAO, true /* keep original bucket signatures; later when do MB processing, don't want to overwrite orig signature */, true) ;
	if (0 != resCB)
		return 51 ;

	// do superbuckets if needed
	if (EClimit < 1) 
		EClimit = -1 ;
	if (EClimit > 0) {
		if (iB <= 0) {
			printf("\nERROR iB=%d <= 0; cannot run SB...", (int) iB) ;
			return 52 ;
			}
		ws.iBound() = iB ;
		ws.EClimit() = EClimit ;
		int32_t res_SB = ws.CreateSuperBuckets() ;
		if (0 != res_SB) {
			printf("\nSuperBuckets failed ...") ;
			return 53 ;
			}
		printf("\nSuperBuckets : nMerges = %d", ws.nSBmerges()) ;
		}

	// NOTE : the MaxNumVarsInBucket() is BE (not MBE) based!!! i.e. as if i-bound=inf
	if (p.VarOrdering_InducedWidth() < 0 && ws.MaxNumVarsInBucket() >= 0) {
		ws.SetVarOrdering_MaxCliqueSize(ws.MaxNumVarsInBucket()) ;
		p.SetVarOrdering_InducedWidth(ws.MaxNumVarsInBucket() - 1) ;
		}

	// wipe out augmented functions; also signatures...
	ws.DestroyAugmentedFunctions() ;

	ws.MaxSpaceAllowed_Log10() = 10.0 ; // 10GB
	int32_t iBoundMin = 2 ;
	int32_t ib, nBP, maxDPB ; double spaceused ;
	bool noPartitioning = true ;
	ws.MBoutputFnTypeWhenOverIBound() = MB_outputfn_type_NN ; // if MB output fn is too large, use NN to approximate it...
	if (iB < 0) {
		int64_t tIBfindS = ARE::GetTimeInMilliseconds() ;
		// this will find best iB (in the process compute partitioning) ...
		int32_t resFindMinIB = ws.FindIBoundForSpaceAllowed(iBoundMin, ib, spaceused, nBP, maxDPB) ;
		int64_t tIBfindE = ARE::GetTimeInMilliseconds() ;
		if (ib < 0 || ib > ws.Problem()->N()) {
			printf("\nFindIBoundForSpaceAllowed failure; res==%d ib=%d", resFindMinIB, (int)iB) ;
			return 61 ;
			}
		iB = ib ;
		}
	else {
		ws.iBound() = iB ;
		int32_t resMBE = ws.CreateMBPartitioning(false, false, 0, noPartitioning, 0) ;
		if (0 != resMBE) {
			printf("\nCreateMBPartitioning failure; res==%d", resMBE) ;
			return 62 ;
			}
		}

	// compute MBE induced_width; this takes into account actual i-bound
	int32_t resMBEindw = ws.ComputeMaxNumVarsInBucket(true) ;
	if (0 != resMBEindw) {
		printf("\nComputeMaxNumVarsInBucket failure; res==%d", resMBEindw) ;
		return 63 ;
		}
	ws.SetPseudoWidth(ws.MaxNumVarsInBucket() - 1) ;

	// compute max superbucket
    BucketElimination::Bucket *maxSB = ws.GetBucketWithMostElimVariables() ;

	printf("\nmaxSB nVars is %d", nullptr != maxSB ? (int) maxSB->nVars() : -1) ;
	fflush(stdout) ;
	
	// print some stats
    // printf("\n*************************");
    // printf("I am almost successful");
	printf("\nSTATS : nBuckets=%d MaxNumVarsInBucket=%d nPartitionedBuckets=%d", (int) ws.nBuckets(), (int) ws.MaxNumVarsInBucket(), (int) ws.nBucketsWithPartitioning()) ;
	fflush(stdout);
	
	signed char approx_bound = 1 ; // 1=max
	bool do_moment_matching = true ;
	int32_t resCOFs_w = -1 ;
	time_t tStart = 0 ; time(&tStart) ;
	int64_t totalSumOutputFunctionsNumEntries = 0 ;
	try {
		resCOFs_w = ws.ComputeOutputFunctions(do_moment_matching, log_console, approx_bound, totalSumOutputFunctionsNumEntries) ;
		}
	catch (...) {
		// if MBE failed, because out of memory, exception may be thrown; we will land here then.
		int32_t MBE_memory_error = 1 ;
		}
	time_t tEnd = 0 ; time(&tEnd) ;
	int64_t tElapsed = tEnd - tStart ;
	if (0 != resCOFs_w) {
		printf("\nws.ComputeOutputFunctions() error; res=%d", resCOFs_w) ;
		}
	int32_t resPCP_w = ws.PostComputationProcessing() ;
	double BEvalue_w = ws.CompleteEliminationResult(approx_bound) ;
	printf("\nwMBE done; result=%g runtime=%lldsec tablesmemory=%lld bytes", BEvalue_w, tElapsed, (int64_t) sizeof(double)*totalSumOutputFunctionsNumEntries) ;


	// This code below is for testing
	bool test_sampling = false ;
	if (test_sampling && nullptr != maxSB ? maxSB->nMiniBuckets() > 0 : false) {
		int32_t varElimOp = VAR_ELIMINATION_TYPE_SUM ;
		ARE::ThreadPoolThreadContext cntx ; // fill in _MB, etc data members...
		cntx._MB = maxSB->MiniBuckets()[0] ;
		ARE::Function & maxSBmb0outputfn = cntx._MB->OutputFunction() ; // output fn of cntx._MB
		cntx._nSamples = 100000 ;
		// cntx._WorkDone = false ;
		if (nullptr == cntx._MB || cntx._nSamples <= 0 /*|| cntx._WorkDone*/) {
			printf("\nERROR : sampling output fn...") ;
			return 1 ;
			}
		ARE::Function & fMBoutput = cntx._MB->OutputFunction() ;
		int32_t res = cntx._MB->SampleOutputFunction(varElimOp, cntx._nSamples, cntx._idx, cntx._nFeaturesPerSample, cntx._Samples_signature, cntx._Samples_values, cntx._min_value, cntx._max_value, cntx._sample_sum) ;
		{ // check samples
			// N is number of variables
//			int32_t assignment[N] ;
			std::unique_ptr<int32_t[]> assignment(new int32_t[p.N()]) ;
			for (int32_t iS = 0; iS < cntx._nSamples; ++iS) {
				int16_t *sample_signature = cntx._Samples_signature.get() + iS * maxSBmb0outputfn.N() ;
				// fill in assignment
				for (int32_t iV = 0 ; iV < maxSBmb0outputfn.N() ; ++iV) {
					int32_t v = maxSBmb0outputfn.Argument(iV) ; // v is argument of maxSBmb0outputfn
					assignment[v] = sample_signature[iV] ;
					}
				// value of the maxSBmb0outputfn corresponding to the given assignment...
				double fn_value_of_the_assignment = maxSBmb0outputfn.TableEntryExNativeAssignment(assignment.get(), p.K()) ;
				// check if values match!
				double d = fabs(fn_value_of_the_assignment - ((double) cntx._Samples_values[iS])) ;
				// 0.00000000000000000000123 has 3 digits of precision --> 1.23e-21
				if (d > 1.0e-16) { // use tiny number instead of 0
					// don't match!
					int error = 1 ;
					}
				}
		}
		// cntx._WorkDone = true ;
		// Could loop through all samples to check that they match the correct OutputFunction
		printf("\nSampleOutputFunction: res = %d, nSamples = %d, num features / sample = %d", res, (int)cntx._nSamples, (int)cntx._nFeaturesPerSample);
		for (int i = 0; i < 10 && i < (int)cntx._nSamples; i++) {
			printf("\n Sample Signature is: ");
			for (int j = 0; j < (int)cntx._nFeaturesPerSample; j++)
				printf("%d ", cntx._Samples_values[i*((int)cntx._nFeaturesPerSample)+j]);
			printf("   Value%d: %f", i, (float)cntx._Samples_values[i]);
		}
		}

	printf("\n") ;
	exit(0) ; // Minibucket is done. Sampling based info continues after this.

	if (0.0 == BEvalue_w) {
		int here = 1 ;
		}

	// print query var
	if (p.QueryVariables().size() > 0) {
		int32_t vq = p.QueryVariables()[0] ;
		int32_t kq = p.K(vq) ;
		std::vector<double> & dist_vq = ws.MarginalSingleVariableDistribution() ;
		printf("\nwMBE done; marginal query var distribution var=%d", vq) ;
		for (int32_t i = 0 ; i < dist_vq.size() ; ++i) {
			printf("\n   P(value=%d)=%g", i, dist_vq[i]) ;
			}
		}

	// NOTE ::::: MBE elimination is done; the rest here is BE sampling & DBE...

    printf(" ComputeOutputFunction_NN Time: %f Count: %d Average: %f\n", ws.time_ComputeOutputFunction_NN, ws.count_ComputeOutputFunction_NN, ws.time_ComputeOutputFunction_NN/ws.count_ComputeOutputFunction_NN);
    printf(" TableEntryEx Time: %f Count: %d Average: %f\n", ws.time_TableEntryEx, ws.count_TableEntryEx, ws.time_TableEntryEx / ws.count_TableEntryEx);
    printf(" Train Time: %f Count: %d Average: %f\n", ws.time_Train, ws.count_Train, ws.time_Train / ws.count_Train);


    std::string o_global_error = global_config.out_file + "global_errors.txt";
    std::ofstream fo;
    fo.open(o_global_error,std::ios_base::app);
    if (fo.is_open())
    {
        fo << BEvalue_w << "\t" << tElapsed; //change to (float) tElapsed/3600
        fo << std::endl;
        fo.close();
    }

    std::ofstream to_write;
    to_write.open(o_file,std::ios_base::app);

    if (to_write.is_open())
    {
        to_write << BEvalue_w  << std::endl ;
        to_write.close();
    }

    std::string o_file2 = global_config.out_file2 + ".txt";

    std::ofstream f;
    f.open(o_file2,std::ios_base::app);

	int32_t k_max = ws.Problem()->maxK() ;
    if (f.is_open())
    {
        f << (int)iB << '\t' << k_max << '\t' << global_config.width_problem << '\t' << (int) ws.nBuckets() << '\t' << (int) ws.count_ComputeOutputFunction_NN << '\t' << global_config.avg_val_mse << '\t' << global_config.avg_test_mse << '\t' << global_config.avg_val_w_mse << '\t' << global_config.avg_test_w_mse << '\t' << global_config.max_wmse << '\t' <<global_config.avg_samples_req << '\t' << global_config.avg_w_test_mse << "\t" << global_config.avg_w_test_err << "\t" << global_config.avg_lambda_test_err << "\t" << global_config.max_avglog_err << "\t" << global_config.max_log_err << "\t"<< global_config.max_log_width << "\t" << global_config.max_log_seq<< '\t' << global_config.max_test_err << "\t" << global_config.max_width<< '\t' << global_config.max_seq_no << '\t'<< global_config.count << '\t'<< global_config.max_lambda_e << "\t" << global_config.width_lambda_e << "\t" << BEvalue_w << '\t' << (float) tElapsed/3600 << '\t' << global_config.negative_samples <<  std::endl ;
        f.close();
    }

	// IF YOU GET TO THIS POINT, ALL WORK is DONE.... !!!
	// YOU CAN EXIT NOW!!!

    BucketElimination::Bucket *B = ws.GetBucketWithMostVariables() ;
	int32_t V = B->V() ;
	printf("\nlargest bucket v=%d", (int32_t) V) ;
	if (v2sample < 0) 
		v2sample = V ;
	if (nsamples <= 0) 
		nsamples = 1 ;
	BucketElimination::Bucket *B_ = ws.MapVar2Bucket(v2sample) ;
	if (NULL == B_) {
		printf("\nERROR : B_ == NULL") ;
		exit(1) ;
		}
	if (B_->V() != v2sample) {
		printf("\nERROR : B_->V() != v2sample") ;
		exit(1) ;
		}
	std::vector<BucketElimination::MiniBucket *> & MBs = B_->MiniBuckets() ;
	printf("\nwill sample v=%d nsamples=%d bSize=%d vars nMBs=%d", (int32_t) v2sample, (int32_t) nsamples, (int32_t) B_->Width(), (int32_t) MBs.size()) ;
	if (MBs.size() <= 0) {
		printf("\nERROR : nMBS <= 0") ;
		exit(1) ;
		}


	ARE::Function & output_fn = MBs[0]->OutputFunction() ;
    printf("\noutput fn size=%lld entries n = %d\n", (int64_t) output_fn.TableSize(), (int) output_fn.N()) ;
	// samples randomly
//	std::vector<int32_t> signature ; signature.reserve(output_fn.N()) ; signature.resize(output_fn.N(), -1) ;
//	for (int32_t i = 0 ; i < output_fn.N() ; ++i) signature[i] = output_fn.Argument(i) ;
//	output_fn.ComputeArgumentsPermutationList(signature.size(), signature.data()) ;
	std::vector<int32_t> signature_assignment ; signature_assignment.reserve(output_fn.N()) ; signature_assignment.resize(output_fn.N(), -1) ;
	printf("\nfn signature : ") ;
	for (int32_t j = 0 ; j < output_fn.N() ; ++j) {
		if (0 == j) printf("%d", (int32_t) output_fn.Argument(j)) ;
		else printf(" %d", (int32_t) output_fn.Argument(j)) ;
		}
	for (int32_t i = 0 ; i < nsamples ; ++i) {
		// generate signature assignment
		for (int32_t j = 0 ; j < output_fn.N() ; ++j) {
			int32_t v = output_fn.Argument(j) ;
			int32_t domain_size_of_v = p.K(v) ;
			int32_t value = RNG.randInt(domain_size_of_v-1) ;
			signature_assignment[j] = value ;
			}
		// fetch value

		int64_t fn_idx = output_fn.ComputeFnTableAdr_wrtSignatureAssignment(signature_assignment.data(), p.K()) ;
		if (fn_idx < 0 || fn_idx >= output_fn.TableSize()) {
			printf("\nERROR : fn_idx out of bounds...") ;
			continue ;
			}
		double fn_value = output_fn.TableEntry(fn_idx) ;
		// DEBUG : print sample
		printf("\nsample : ") ;
		for (int32_t j = 0 ; j < output_fn.N() ; ++j) {
			if (0 == j) printf("%d", (int32_t) signature_assignment[j]) ;
			else printf(" %d", (int32_t) signature_assignment[j]) ;
			}
		printf(" = %g logscale (idx=%lld)", fn_value, fn_idx) ;
		}

	return 0 ;
}

