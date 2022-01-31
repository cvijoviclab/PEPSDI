#=
    File to launch benchmark of checking run-time for option 1 and option 2 
    for the Schlogl model. 

    ARGS[1] correlation level (e.g 0.99, 0.999)
    ARGS[2] number of individuals (either 40, 80, 120, 160 or 200)
    ARGS[3] sampler (new or old)
    ARGS[4] true or false (true pilot run is performed)

    For computational efficiency only 10,000 samples are taken for each 
    number of individual (and correlation level). The results are stored in 
    intermediate, and processed by a R-script.
=# 

using Random
using Distributions
using CSV
using DataFrames
using LinearAlgebra
using Printf
using Dates
tmp = push!(LOAD_PATH, pwd() * "/Code")
using PEPSDI


# Load model equations to use 
include(pwd() * "/Code/Models/Schlogl_ssa.jl")


# Sanity check input 
rho_level = parse(Float64, ARGS[1])
n_ind = parse(Int64, ARGS[2])
if !(n_ind ∈ [20, 40, 60, 80, 100, 200])
    @printf("Error, number of individuals must be either 20, 40, 80, 60, 100 or 200")

    exit(1)
end
run_pilot = parse(Bool, ARGS[4])


function run_timing(rho, n_individuals; pilot=true, sampler="new")

    P_mat = [1]
    my_model = init_sde_model(schlogl_alpha_kappa, 
                              schlogl_beta_kappa, 
                              schlogl_x0!, 
                              schlogl_obs_sde, 
                              schlogl_prop_obs, 
                              1, 1, P_mat) 

    # Parameter info for population parameters and individual parameters 
    prior_mean = [Normal(7.0, 5.0)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf)]
    prior_sigma = [Normal(2.0, 0.5)]
    prior_kappa = [Normal(-1.0, 5.0), Normal(-3, 5.0)]
    
    pop_param_info = init_pop_param_info(prior_mean, 
                                         prior_scale, 
                                         prior_sigma, 
                                         prior_pop_kappa = prior_kappa,
                                         pos_pop_kappa = false, 
                                         log_pop_kappa = true,
                                         pos_pop_sigma= true)

    
    ind_val_path = pwd() * "/Intermediate/Simulated_data/SDE/Multiple_ind/Schlogl_benchmark/Param" * string(n_individuals) * ".csv"
    ind_param = log.( convert(Array{Float64, 2}, CSV.read(ind_val_path, DataFrame)))

    ind_param_info = init_ind_param_info(ind_param, length(prior_mean), 
        log_scale=true, pos_param=false)


    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Multiple_ind/Schlogl_benchmark/N_ind" * string(n_individuals) * ".csv"
    dir_save = "Schlogl_benchmark_time/N_ind" * string(n_individuals) * "_rho" * string(rho) * "_" * sampler * "/"
    file_loc = init_file_loc(path_data, dir_save, multiple_ind=true)

    # Filter information 
    dt = 5e-2   
    filter_opt = init_filter(ModDiffusion(), dt, n_particles=100, rho=rho)

    # Tuning of particles 
    tune_part_data = init_pilot_run_info(pop_param_info, 
                                         n_particles_pilot=100, 
                                         n_samples_pilot=1, 
                                         rho_list=[0.999], 
                                         n_times_run_filter=50, 
                                         init_kappa = [-2.17, -8.73], 
                                         init_mean = [7.2], 
                                         init_scale = [0.1])

    # Sampler 
    cov_mat_ind = diagm([0.1])
    cov_mat_pop = diagm([0.1, 0.1, 0.5 / 10.0]) 
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat=cov_mat_ind, step_before_update=500)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=500)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalDiag(), n_warm_up=200)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances = [0.05, 0.05 ,0.05])

    n_samples = 10000

    if sampler == "new"

        if pilot == true
            tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
                file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)
        else
            exp_id = 1
            stuff = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
                    filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=exp_id)
        end

    else    

        if pilot == true

            tune_particles_opt1(tune_part_data, pop_param_info, ind_param_info, 
                file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt)
        else

            stuff = run_PEPSDI_opt1(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
                filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, pilot_id=1)
                
        end
    end

end      

run_timing(rho_level, n_ind, pilot=run_pilot, sampler=ARGS[3])