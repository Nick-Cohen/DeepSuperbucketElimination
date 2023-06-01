#define NOMINMAX

#include <stdlib.h>
#include <memory.h>
#include <exception>
#include <chrono>
#include <random>

#include <Sort.hxx>
#include <Function.hxx>
#include <Bucket.hxx>
#include <MiniBucket.hxx>
#include <MBEworkspace.hxx>
#include "Utils/MersenneTwister.h"

#include <Function-NN.hxx>
#include "NNConfig.h"
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

	bool data_is_log_space = Problem()->FunctionsAreConvertedToLogScale() ;

	// generate samples...
	int64_t nSamples = 100 ;
	int32_t nFeaturesPerSample = -1 ;
	std::unique_ptr<int16_t[]> samples_signature ; 
	std::unique_ptr<float[]> samples_values ;
	float samples_min_value, samples_max_value, samples_sum ;
	{
		std::random_device rd ;
		uint32_t seed = rd() ;
		int32_t resSampling = SampleOutputFunction(varElimOp, nSamples, seed, nFeaturesPerSample, samples_signature, samples_values, samples_min_value, samples_max_value, samples_sum) ;
	}

	// write samples into xml file...
	const char *fn = "samples.xml" ;
	{
		std::unique_ptr<char[]> sBUF(new char[4096]) ;
		if (nullptr == sBUF) 
			return 1 ;
		char *buf = sBUF.get() ;
		std::string s ;
		FILE *fp = fopen(fn, "w") ;
		fwrite("<?xml version=\"1.0\" encoding=\"UTF-8\"?>", 1, 38, fp) ;
		sprintf(buf, "\n<samples n=\"%d\" nFeaturesPerSample=\"%d\" datainlogspace=\"%c\">", (int) nSamples, (int) nFeaturesPerSample, data_is_log_space ? 'Y' : 'N') ; fwrite(buf, 1, strlen(buf), fp) ;
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
		fwrite("\n</samples>", 1, 11, fp) ;
		fclose(fp) ;
	}

/*

   // std::cout<<"in minibucket-NN ----";
    auto start = std::chrono::high_resolution_clock::now();
    int32_t i, k;

    bool convert_exp=true;
    if(0 == global_config.network.compare("net"))
        convert_exp=false;
    else
        std::cout<<"masked_net.....";
        //if(global_config.l==1)
        //{
        //    convert_exp=false;      //so as to run both
        //}

    // TODO : if _OutputFunction is of type FunctionNN, then do .....
    ARE::FunctionNN *fNN = dynamic_cast<ARE::FunctionNN *>(_OutputFunction) ;
    if (NULL == fNN)
        return 1 ;

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
	if (_nFunctions < 1) {
		ARE::Function & OutputFN = OutputFunction() ;
		ARE_Function_TableType & f_const_value = OutputFN.ConstValue() ;
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
    if (nVars() < 1)
        // nothing to do; should not happen; however, allow this to pass as ok; calling fn should be able to handle this.
        return 0 ;

	// 2022-02-16 KK : vars used later but never declared; declare here...
	float min_error, max_error, avg_error ;

    const int32_t w = _SortedSignature.size() ; // Width() ;
    if (w < 0 || _nFunctions < 1)
        return 0 ;
    const int32_t *signature = Signature() ;

    // generate some number of random samples...
    Config config;
    //int32_t nSamples = global_config.sample_size;

    int32_t nSamples;
    int32_t w_in = fNN->N();

    int l = global_config.n_layers +1;
    float temp = ((l-1)*pow(w_in,2) + l*w_in + 4);
    float pd = temp*log(temp/l);

    std::cout<<"pseudo-dimension" << pd <<"\t" << w_in << "\t" << l << "\t" << temp;
    //int32_t nSamples = int((pd+log(1/0.001))/(0.1*s*(1-v_s)));

    double s = 0.1;
    double v_s = 0.1;
    if(global_config.var_samples==1)
    {
        nSamples = int(w_in*2500);
    }
    else if(global_config.var_samples==2)
    {
        nSamples = int(w_in*5000);
    }
    else if(global_config.var_samples==0)
        nSamples = global_config.sample_size;
    else{
        if(global_config.epsilon>0){
            float delta=0.001;
            std::cout<<"samples based on epsilon -----";
            //if(global_config.n_layers==1)
            nSamples = int(global_config.c*(pd + log(1/delta))/global_config.epsilon);
           // else
             //   nSamples = int(global_config.c*(pd + log(1/delta))/global_config.epsilon);
        }
        else
            nSamples = global_config.alpha*w_in;

        nSamples = nSamples/(1-s);
    }

    if (nSamples>1000000)
        nSamples=1000000;
    if (global_config.lb>0)
        if (nSamples<40000)
            nSamples=40000;

    if(global_config.up>0)
        if (nSamples>700000)
            nSamples=700000;

    std::cout<<"nSamples----" <<nSamples;
    std::vector<int32_t> values_, vars_ ;
    values_.resize(problem->N(), 0) ;
    vars_.resize(problem->N(), 0) ;

//    printf("problem->N() %d \n", problem->N());
    if (values_.size() != problem->N() || vars_.size() != problem->N())
        return 1 ;

    int32_t *vals = values_.data();
    int32_t *vars = vars_.data();

    k = problem->K(_V);

    for ( i = 0 ; i < fNN->N() ; ++i){
        vars[i] = fNN->Argument(i) ;
    }

    printf("\n In Minibucket NN --");
    vars[i] = _V;

    int n_val_samples, n_train_samples, n_test_samples;

    if(global_config.var_samples==0){
        s = 0.8;
        v_s = 0.2;
        n_val_samples = int(v_s*s*nSamples);
        n_train_samples = int((1-v_s)*s*nSamples);
        n_test_samples = int((1-s)*nSamples);
    }
    else{
        n_val_samples = int(s*nSamples);
        n_train_samples = int((1-s)*nSamples);
        n_test_samples = 50000;
    }


    fNN->train_samples = n_train_samples;
    fNN->val_samples = n_val_samples;
    fNN->test_samples = n_test_samples;

    std::vector<std::vector<int32_t>> train_samples, val_samples,test_samples;
    train_samples.resize(n_train_samples, vector<int>(fNN->N()));
    val_samples.resize(n_val_samples, vector<int>(fNN->N()));
    test_samples.resize(n_test_samples, vector<int>(fNN->N()));

    std::vector<float> train_sample_values, val_sample_values, test_sample_values;
    train_sample_values.resize(n_train_samples);
    val_sample_values.resize(n_val_samples);
    test_sample_values.resize(n_test_samples);

    ARE_Function_TableType const_factor = bews->FnCombinationNeutralValue() ;

    int32_t nFNs = 0, n_fNN_in = 0 ;

    std::vector<ARE::Function *> flist ; flist.reserve(_nFunctions) ; if (flist.capacity() != _nFunctions) return 1 ;
    for (int32_t j = 0 ; j < _nFunctions ; j++) {
        ARE::Function *f = _Functions[j] ;
        if (NULL == f) return ERRORCODE_fn_ptr_is_NULL ;
        ARE::FunctionNN *fNN_in = dynamic_cast<ARE::FunctionNN*>(f) ;
        if (nullptr != fNN_in){
            ++n_fNN_in ;
        }
        if (0 == f->N()) {
            bews->ApplyFnCombinationOperator(const_factor, f->ConstValue()) ;
        }
        else {
            flist.push_back(f) ;
            f->ComputeArgumentsPermutationList(w, vars); }
    }
    // 2021-07-30 : n_fNN_in is the number of input function of type ARE::FunctionNN (NN-learned function)

    float zero = 0.1*pow(10,-34);
    int count_non_zero =0, t_non_zero=0, v_non_zero=0, test_non_zero=0;
    int burnTime=nSamples/10;
    float running_sum=0;

        // TODO Below here, edit to use the superbucket sampling
        printf("Uniform Sampling -----");

        ARE::ThreadPoolThreadContext cntx ;
        int32_t res = cntx._MB->SampleOutputFunction(varElimOp, nSamples, cntx._idx, cntx._nFeaturesPerSample, cntx._Samples_signature, cntx._Samples_values, cntx._min_value, cntx._max_value, cntx._sample_sum) ;


        //ofstream myfile ("bucket_samples_level_" + std::to_string(this->Workspace()->count_Train) + ".txt",std::ios_base::app);

        //for (i = 0; i < nSamples + n_test_samples; ++i) {
        for (i = 0; i < nSamples ; ++i) {
            // generate assignment to fNN arguments
            // printf("\n********************\n");
            ARE_Function_TableType V = bews->VarEliminationDefaultValue();

            for (int32_t j = 0; j < fNN->N(); ++j) {
                int32_t v = fNN->Argument(j);
                int32_t domain_size_of_v = problem->K(v);
                //            printf("%d ", domain_size_of_v);
                int32_t value = RNG.randInt(domain_size_of_v - 1);
                vals[j] = value;
            }
            // enumerate all current variable values; compute bucket value for each configuration and combine them using elimination operator...

            for (int32_t j = 0; j < k; ++j) {
                vals[fNN->N()] = j;
                ARE_Function_TableType v = bews->FnCombinationNeutralValue();
                // compute value for this configuration : fNN argument assignment + _V=j
                for (int32_t l = 0; l < _nFunctions; ++l) {
                    ARE::Function *f = _Functions[l];
                    if (NULL == f) continue;
                    //std::cout<<"calling function ---";
                    double fn_v = f->TableEntryEx(vals,
                                                  problem->K()); //This would make us problem specifially when privious ones be NN
                    bews->ApplyFnCombinationOperator(v, fn_v);
                }
                ApplyVarEliminationOperator(varElimOperator, problem->FunctionsAreConvertedToLogScale(), V, v);
            }

            bews->ApplyFnCombinationOperator(V, const_factor);
            //if (V > fNN->ln_max_value) {
            //    fNN->ln_max_value = V;
            //}
            c1++;

            if (i < n_train_samples) {
                for (int32_t m = 0; m < fNN->N(); ++m) {
                    //if(convert_exp)
                    //    train_samples[i][m] = 2*(vals[m]/domain_size_of_v) - 1;
                    //else
                    train_samples[i][m] = vals[m]; //vals[fNN->ArgumentsPermutationList()[m]]; // vals[fnn->_Argument]
                }
                if (convert_exp) {
                    V = exp(V);
                }
                if (V >= zero) {
                    t_non_zero++;
                    // printf("sample value - %f, count -- %d",V,count_non_zero);
                }

                train_sample_values[i] = V; //should be V

                if (V < fNN->ln_min_value)
                    fNN->ln_min_value = V;

                if (V != zero) {
                    count_non_zero++;
                }
                if (V > fNN->ln_max_value)
                    fNN->ln_max_value = V;
            } else if (i >= n_train_samples && i < n_train_samples + n_val_samples) {
                for (int32_t m = 0; m < fNN->N(); ++m) {
                    //if(convert_exp)
                    //    val_samples[i - n_train_samples][m] = 2*(vals[m]/domain_size_of_v) - 1;
                    //else
                    val_samples[i - n_train_samples][m] = vals[m]; //vals[fNN->ArgumentsPermutationList()[m]]; // vals[fnn->_Argument]
                    //val_samples[i - n_train_samples][m] = vals[m]; //vals[fNN->ArgumentsPermutationList()[m]]; // vals[fnn->_Argument]
                }
                if (convert_exp) {
                    V = exp(V);
                }
                if (V >= zero) {
                    v_non_zero++;
                }

                val_sample_values[i - n_train_samples] = V; //should be V

            } else if (i >= n_train_samples + n_val_samples && i < n_train_samples + n_val_samples + n_test_samples) {
                for (int32_t m = 0; m < fNN->N(); ++m) {
                    //if(convert_exp)
                    //    test_samples[i - n_train_samples - n_val_samples][m] = 2*(vals[m]/domain_size_of_v) - 1;
                    //else
                    test_samples[i - n_train_samples - n_val_samples][m] = vals[m];
                    //test_samples[i - n_train_samples - n_val_samples][m] = vals[m]; //vals[fNN->ArgumentsPermutationList()[m]]; // vals[fnn->_Argument]
                }
                if (convert_exp) {
                    V = exp(V);
                }
                if (V >= zero) {
                    test_non_zero++;
                }
                test_sample_values[i - n_train_samples - n_val_samples] = V; //should be V
            }
        }
    }

    std::string o_file = global_config.out_file + "plot.txt";
    std::ofstream f;
    f.open(o_file,std::ios_base::app);
    printf("sampling finished-----");

    // End of sampling portion

    if(count_non_zero<0.001*nSamples)
    {
        printf("exiting the program because number of non-zero values is %d out of %d samples",count_non_zero,nSamples);
        if (f.is_open()) {
            f <<  "program exited" << std::endl;
        }
        f.close();
        //exit(0);
    } else
    {
       // if (f.is_open()) {
       //     f << count_non_zero << '\t' << t_non_zero << '\t' << v_non_zero << '\t' << test_non_zero << '\t';
       // }
        f.close();
    }

    DATA_SAMPLES *DS_train, *DS_val, *DS_test;
    t_non_zero=0, v_non_zero=0, test_non_zero=0;

    fNN->log_sum_exp(train_sample_values,n_train_samples);

    DS_train = fNN->samples_to_data(train_samples, train_sample_values, fNN->N(), n_train_samples, t_non_zero);
    // printf("samples to data done ---");
     DS_val = fNN->samples_to_data(val_samples, val_sample_values, fNN->N(), n_val_samples,v_non_zero);

    printf("samples created  ---%d %d", n_train_samples, n_val_samples);

    std::cout<<"\n"<<fNN->ln_min_value<<fNN->ln_max_value;
    bool to_save=false;
    //f(_V==1448)
    //    to_save=true;

    fNN->Train(DS_train, DS_val,to_save,_V);

    DS_test = fNN->samples_to_data(test_samples, test_sample_values, fNN->N(), n_test_samples,test_non_zero);
    fNN->test(DS_test,to_save);

     min_error = fNN->local_min_e;
     max_error = fNN->local_max_e;
     avg_error = fNN->local_avg_e;

    _Bucket->_ApproxError_Cmltv_Max = max_error;
    _Bucket->_ApproxError_Cmltv_Min = min_error;
    _Bucket->_ApproxError_Cmltv_Avg = avg_error;
    _Bucket->_ApproxError_Cmltv_wMax = fNN->local_w_max_e;
    _Bucket->_ApproxError_Cmltv_wMin = fNN->local_w_min_e;
    _Bucket->_ApproxError_Cmltv_wAvg = fNN->local_w_avg_e;
    _Bucket->_ApproxError_Cmltv_var = fNN->local_var_e;
    std::cout<< "Bucket global errors ------"<< '\n' << fNN->local_var_e << '\t'<<_Bucket->_ApproxError_Cmltv_Max<< "\t" << _Bucket->_ApproxError_Cmltv_Min << "\t" << _Bucket->_ApproxError_Cmltv_Avg << "\t";

    std::vector<BucketElimination::Bucket*> children;
    std::vector<int32_t> children_v;
    _Bucket->GetChildren(children);
    int children_with_error=0;
    for (i=0;i<children.size();i++){
        _Bucket->_ApproxError_Cmltv_Max += children[i]->_ApproxError_Cmltv_Max;
        _Bucket->_ApproxError_Cmltv_Min += children[i]->_ApproxError_Cmltv_Min;
        _Bucket->_ApproxError_Cmltv_Avg += children[i]->_ApproxError_Cmltv_Avg;
        _Bucket->_ApproxError_Cmltv_wMax += children[i]->_ApproxError_Cmltv_wMax;
        _Bucket->_ApproxError_Cmltv_wMin += children[i]->_ApproxError_Cmltv_wMin;
        _Bucket->_ApproxError_Cmltv_wAvg += children[i]->_ApproxError_Cmltv_wAvg;
        _Bucket->_ApproxError_Cmltv_var += children[i]->_ApproxError_Cmltv_var;

        if(children[i]->_ApproxError_Cmltv_Avg>0)
            children_with_error +=1;
        children_v.push_back(children[i]->V());
        //std::cout<<children_v[i]<< "\n";
    }

    //std::cout<< "Bucket global errors ------"<<_Bucket->_ApproxError_Cmltv_Max<< "\t" << _Bucket->_ApproxError_Cmltv_Min << "\t" << _Bucket->_ApproxError_Cmltv_Avg << "\t";
    // printf("Testing done --\n");
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    double duration = microseconds*1.0 / 1000000;
//        printf(" Table Duration: %f\n", duration);
    this->Workspace()->time_ComputeOutputFunction_NN += duration;
    this->Workspace()->count_ComputeOutputFunction_NN++;
    //std::string o_file = global_config.out_file + ".txt";
    //std::ofstream f;
    f.open(o_file,std::ios_base::app);

    if (f.is_open())
    {
        printf("writing to the file %s", o_file.c_str());
        if(0==global_config.s_method.compare("rs"))
            f<< fNN->ln_max_value <<'\t' << fNN->ln_min_value<<'\t'<< n_fNN_in << '\t' << (1-v_s)*s*nSamples << '\t' << c1<< std::endl ;
        else
            f<< fNN->ln_max_value <<'\t' << fNN->ln_min_value<<'\t'<< n_fNN_in << '\t' << (1-s)*nSamples << '\t' << children_with_error << "\t" << avg_error << "\t" <<  fNN->local_w_avg_e << "\t" << fNN->local_var_e << "\t" << _Bucket->_ApproxError_Cmltv_Max  << "\t" << _Bucket->_ApproxError_Cmltv_wAvg << '\t' << _Bucket->_ApproxError_Cmltv_var << std::endl ;
    //    f << c1 << '\t' << this->Workspace()->time_ComputeOutputFunction_NN << '\t'  << this->Workspace()->time_ComputeOutputFunction_NN/this->Workspace()->count_ComputeOutputFunction_NN << '\t' << this->Workspace()->time_TableEntryEx << '\t' << this->Workspace()->time_TableEntryEx / this->Workspace()->count_TableEntryEx << '\t' << this->Workspace()->time_Train << '\t' << this->Workspace()->time_Train / this->Workspace()->count_Train << std::endl ;
    //    f << c1 << std::endl ;
        f.close();
    }

    BucketElimination::Bucket *b_ancestor = _Bucket->ParentBucket();
    std::cout<< b_ancestor->V();

    std::string o_global_error = global_config.out_file + "global_errors.txt";
    std::ofstream fo;
    fo.open(o_global_error,std::ios_base::app);
    if (fo.is_open())
    {
        //fo << _V << "\t" <<  w_in << "\t" << children_with_error << "\t" << min_error << "\t" << max_error << "\t" << avg_error << "\t" << fNN->local_w_min_e << "\t" << fNN->local_w_max_e << "\t"<< fNN->local_w_avg_e << "\t" << _Bucket->_ApproxError_Cmltv_Min<< "\t" << _Bucket->_ApproxError_Cmltv_Max << "\t" << _Bucket->_ApproxError_Cmltv_Avg <<  "\t" <<_Bucket->_ApproxError_Cmltv_wMin<< "\t" << _Bucket->_ApproxError_Cmltv_wMax << "\t" << _Bucket->_ApproxError_Cmltv_wAvg <<  "\t";
        fo << _V << "\t" <<  w_in << "\t" << children_with_error << "\t" << min_error << "\t" << avg_error << "\t" << fNN->local_w_avg_e << "\t" << max_error << "\t" << _Bucket->_ApproxError_Cmltv_Min << "\t" << _Bucket->_ApproxError_Cmltv_Avg << "\t" << _Bucket->_ApproxError_Cmltv_wAvg << _Bucket->_ApproxError_Cmltv_Max << "\t" ;
        for(int m=0;m<children_v.size();m++)
            fo << children_v[m] << "\t";
        fo << std::endl;
        fo.close();
    }

    printf(" ComputeOutputFunction_NN Time: %f Count: %d Average: %f\n", this->Workspace()->time_ComputeOutputFunction_NN, this->Workspace()->count_ComputeOutputFunction_NN, this->Workspace()->time_ComputeOutputFunction_NN/this->Workspace()->count_ComputeOutputFunction_NN);
    printf(" TableEntryEx Time: %f Count: %d Average: %f\n", this->Workspace()->time_TableEntryEx, this->Workspace()->count_TableEntryEx, this->Workspace()->time_TableEntryEx / this->Workspace()->count_TableEntryEx);
    printf(" Train Time: %f Count: %d Average: %f\n", this->Workspace()->time_Train, this->Workspace()->count_Train, this->Workspace()->time_Train / this->Workspace()->count_Train);
    printf("Bucket number ---- %d", _V);

*/

    return 0 ;
}





