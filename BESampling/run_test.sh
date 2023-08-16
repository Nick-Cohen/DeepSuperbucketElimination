#!/bin/bash

# problem_name="pedigree1.uai"
problem_name="grid10x10.f10.uai"
problem="../problems/$problem_name"
elim_order="../problems/$problem_name.vo"
results_prefix="../problems/${problem_name:0:-4}"

# Dummy variables below for now

# Hyper-parameters of the NN
learning_rate="0.001"		# Optimization learning rate
n_epochs="500"				# Maximum number of epochs to train the net
batch_size="256"			# Number of training sequences in each batch
network="net"				# using net "net" or masked net "masked-net"
s_method="is"				# Whether loss is a weighted loss "is" or uniform "us"
width_problem="13"			# width for the problem, training occurs for buckets >width_problem
stop_iter="2"				# condition to stop training when val error increases cont. for more than "stop_iter" times

# Variability in NNs
dim="1"					# vary the hidden dimensions of the neural net
e="0.35"					# needed to determine #samples. Eq: nSamples = int((pd + log(1/delta))/global_config.epsilon);

# WMB variables
iB=8
ecl=10
nsamples=10000

# Kalev's cmd line input -fUAI C:\UCI\pedigree\pedigree1.uai -fVO C:\UCI\pedigree\pedigree1.uai.vo -iB 13 -EClimit 1000000, gets 15 merges



echo ./build/BESampling -fUAI $problem -fVO $elim_order -iB $iB -nsamples $nsamples -EClimit $ecl -nsamples 10 -batch_size $batch_size -lr $learning_rate -n_epochs $n_epochs --network $network --out_file ${results_prefix}${epsilon}_${dim}_${s_m} --sampling_method $s_m --width_problem $width_problem --stop_iter $stop_iter --var_dim $dim --epsilon $epsilon 
./build/BESampling -fUAI $problem -fVO $elim_order -iB $iB -nsamples $nsamples -EClimit $ecl -nsamples 10 -batch_size $batch_size -lr $learning_rate -n_epochs $n_epochs --network $network --out_file ${results_prefix}${epsilon}_${dim}_${s_m} --sampling_method $s_m --width_problem $width_problem --stop_iter $stop_iter --var_dim $dim --epsilon $epsilon 
