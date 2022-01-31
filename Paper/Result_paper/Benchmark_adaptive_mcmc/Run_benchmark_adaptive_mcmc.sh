#!/bin/bash


# Shell script that launches the simulations for comparing adaptive mcmc-proposals for the Schlögl 
# and Ornstein model by calling Test_adaptive_mcmc_ornstein and Test_adaptive_mcmc_schlogl scripts. 
# This script was run on a cluster, and if someone wants to reproduce the results it is recomended to 
# do the same (to not overload the RAM). Furthermore, to reproduce the results the julia-path 
# must be changed. 
# Args:
# $1 : Running option: [Schlogl_pilot, Schlogl, Ornstein_pilot, Ornstein, Plot_result, Simulate_data]. The second last 
#      calls a R-script for plotting the results. The last simulates the three data-sets used for each model. 
# $2 : Sampler to use (AM, GenAM, RAM)
# $3 : Data-set to use (which to use dependes on model, see below)


# Args:
#     $1 : Sampler to use (AM, GenAM, RAM)
#     $2 : Data-set (12, 123, 12345)
run_schlogl_pilot_cluster ()
{
    #~/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg1 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg2 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg3 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg4 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg5 $2 1 pilot &
}


# Args:
#     $1 : Sampler to use (AM, GenAM, RAM)
#     $2 : Data-set (12, 123, 12345)
#     $3 : Number of repitions to run 
run_schlogl_cluster ()
{
    #~/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg1 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg2 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg3 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg4 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_schlogl.jl $1 sg5 $2 $3 not_pilot &
    
}


# Args:
#     $1 : Sampler to use (AM, GenAM, RAM)
#     $2 : Data-set (12, 123, 12345)
run_ornstein_pilot_cluster ()
{
    #~/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg1 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg2 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg3 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg4 $2 1 pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg5 $2 1 pilot &
}


# Args:
#     $1 : Sampler to use (AM, GenAM, RAM)
#     $2 : Data-set (12, 123, 12345)
#     $3 : Number of repitions to run 
run_ornstein_cluster ()
{
    #~/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg1 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg2 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg3 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg4 $2 $3 not_pilot &
    /home/sebpe/julia-1.5.2/bin/julia ./Paper/Result_paper/Benchmark_adaptive_mcmc/Test_adaptive_mcmc_ornstein.jl $1 sg5 $2 $3 not_pilot &
    
}


simulate_data_desktop ()
{
    # Simulate Ornstein-model 
    echo "Ornstien model"
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Single_ornstein 1
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Single_ornstein 12
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Single_ornstein 123

    # Simulate Schogl-model 
    echo "Schlögl model"
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Single_schlogl_ssa 12 
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Single_schlogl_ssa 123
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Single_schlogl_ssa 12345
}


# If running Schlogl not pilot  
if [ $1 == "Simulate_data" ];then

    echo "Simulating data for benchmark"
    simulate_data_desktop $2 $3

    wait 
fi 


# If input argument is to run pilot-model 1 
if [ $1 == "Schlogl_pilot" ];then
    echo "Running pilot runs for schlogl"

    # $2 = sampler to use 
    # $3 = data-set
    run_schlogl_pilot_cluster $2 $3 

    wait 

    echo "Done with pilot runs for schlogl"
fi


# If running Schlogl not pilot  
if [ $1 == "Schlogl" ];then
    
    run_schlogl_cluster $2 $3 $4

    wait 
fi 


# If input argument is to run pilot-model 1 
if [ $1 == "Ornstein_pilot" ];then
    echo "Running pilot runs for ornstein"

    # $2 = sampler to use 
    # $3 = data-set
    run_ornstein_pilot_cluster $2 $3 

    wait 

    echo "Done with pilot runs for ornstein"
fi


# If running Schlogl not pilot  
if [ $1 == "Ornstein" ];then

    # $2 : Sampler to use (AM, GenAM, RAM)
    # $3 : Data-set (12, 123, 12345)
    # $4 : Number of repitions to run 
    run_ornstein_cluster $2 $3 $4

    wait 
fi 


if [ $1 == "Plot_result" ];then 

    Rscript Process_adaptive_mcmc.R 2> /dev/null

fi 
