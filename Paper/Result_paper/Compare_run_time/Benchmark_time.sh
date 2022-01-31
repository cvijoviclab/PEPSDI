#!/bin/bash


# Shell script that launches the simulations for comparing run-times for the Schlögl model. 
# This script was run on a cluster for a short pilot run (to obtain a good start-guess for each case) and 
# If someone wants to reproduce the results it is recomended to do the same (to not overload the RAM). 
# Furthermore, to reproduce the results the julia-path. 
# The actual benchmarking of timing results was run on a desktop (to avoid the impact of other persons using 
# the cluster). 
# Args:
# $1 : Running option: [Pilot_cluster, Timing_desktop, Simulate_desktop, Plot_result], the second last simulates the 
#      data for the benchmark. Here, smaller time-intervall is considered compared to Fig. 3 in the paper due to 
#      computational reasons. The last options plots the results using a R-script. 


run_schlogl_pilot_cluster ()
{

    echo "Tuning particles for new sampler"

    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 20 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 40 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 60 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 80 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 100 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 200 new "true" &

    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 20 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 40 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 60 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 80 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 100 new "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 200 new "true" &

    wait 

    echo "Tuning particles for old sampler"

    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 20 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 40 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 60 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 80 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.999 100 old "true" &

    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 20 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 40 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 60 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 80 old "true" &
    /home/sebpe/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl 0.99 100 old "true" &

}

simulate_data_desktop ()
{

    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Multiple_schlogl_benchmark 20 
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Multiple_schlogl_benchmark 40 
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Multiple_schlogl_benchmark 60 
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Multiple_schlogl_benchmark 80 
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Multiple_schlogl_benchmark 100 
    /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia ./Paper/Generate_simulated_data/Generate_simulated_data.jl Multiple_schlogl_benchmark 200 
    wait 

}


# $1 Times to run 
# $2 Correlation level 
# $3 N-individuals 
# $4 Sampler (old or new)
run_timing_benchmark ()
{
    echo "New method rho = 0.999"

    for (( i = 1; i <= $1; i++ )); do 

        /home/sebpe/julia-1.5.2-linux-x86_64/julia-1.5.2/bin/julia --threads 2 ./Paper/Result_paper/Compare_run_time/Benchmark_times_schlögl.jl $2 $3 $4 "false" &

        if (( $i % 2 == 0 ));then 
            wait 
        fi
    done
    
    wait 

}


timing_schlögl_benchmark ()
{
    echo "Running for old-samplers"
    run_timing_benchmark 3 0.999 20 old
    echo "N-individuals = 40"
    run_timing_benchmark 3 0.999 40 old
    echo "N-individuals = 60"
    run_timing_benchmark 3 0.999 60 old
    echo "N-individuals = 80"
    run_timing_benchmark 3 0.999 80 old
    echo "N-individuals = 100"
    run_timing_benchmark 3 0.999 100 old

    echo "Running for new-samplers"
    run_timing_benchmark 3 0.999 20 new
    echo "N-individuals = 40"
    run_timing_benchmark 3 0.999 40 new
    echo "N-individuals = 60"
    run_timing_benchmark 3 0.999 60 new
    echo "N-individuals = 80"
    run_timing_benchmark 3 0.999 80 new
    echo "N-individuals = 100"
    run_timing_benchmark 3 0.999 100 new
    echo "N-individuals = 200"
    run_timing_benchmark 3 0.999 200 new

}


# If running Schlogl not pilot  
if [ $1 == "Pilot_cluster" ];then
    
    run_schlogl_pilot_cluster

    wait 
fi 


# If running Schlogl not pilot  
if [ $1 == "Timing_desktop" ];then
    
    timing_schlögl_benchmark

    wait 
fi 


# If running Schlogl not pilot  
if [ $1 == "Simulate_desktop" ];then
    
    simulate_data_desktop

    wait 
fi 


# If running Schlogl not pilot  
if [ $1 == "Plot_result" ];then

    Rscript Compare_time.R 2> /dev/null

    wait 
fi 
