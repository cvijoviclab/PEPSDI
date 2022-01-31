#= 
    Running the inference for the Schlögl-model. For a closer description 
    of each step see the notebook in the Code/Examples notebook. 

    Args:
        ARGS[1] : Option to run (Run_pilot, Run_inference, Investigate_c1)
            The last option will, given existing inference results, run the 
            analysis to investigate the impact of changing c1 (Fig. S3)
=# 


using Distributions # For placing priors 
using Random # For setting seed 
using LinearAlgebra # For matrix operations 
using DataFrames # For reading csv-files
using CSV # For reading csv-files
using Printf # Formatted printing 
tmp = push!(LOAD_PATH, pwd() * "/Code") # Push PEPSDI into load-path 
using PEPSDI # Load PEPSDI 

    
# Defining the model 

function schlogl_alpha(du, u, p, t)
    c = p.c
    kappa = p.kappa
    kappa3 = 23.1

    h_vec1 = kappa[1] * u[1] * (u[1] - 1)
    h_vec2 = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[1]
    h_vec4 = kappa3 * u[1]

    @views du[1] = h_vec1 - h_vec2 + h_vec3 - h_vec4
end

function schlogl_beta(du, u, p, t)
    c = p.c
    kappa = p.kappa
    kappa3 = 23.1
    
    h_vec1 = kappa[1] * u[1] * (u[1] - 1)
    h_vec2 = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[1]
    h_vec4 = kappa3 * u[1]

    @views du[1, 1] = h_vec1 + h_vec2 + h_vec3 + h_vec4
end

function schlogl_u0!(u0, p) 
    u0[1] = 0.0
end

function schlogl_h(y_mod, u, p, t)
    
    # g = X[1] -> y = X[1]
    y_mod[1] = u[1]
end

function schlogl_g(y_obs, y_mod, error_param, t, dim_obs)
    
    # Since y_obs ~ N(y_mod, xi^2) the likelihood can be calculated 
    # via the normal distribution. Perform calculations on log-scale 
    # for stabillity. 
    prob::FLOAT = 0.0
    noise = error_param[1]
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    return exp(prob)
end

    
function run_inference_schlogl(run_pilot::Bool)
    
    P_mat = [1]
    my_model = init_sde_model(schlogl_alpha, 
                              schlogl_beta, 
                              schlogl_u0!, 
                              schlogl_h, 
                              schlogl_g, 
                              1, 1, P_mat) 

    # Parameter info for eta-parameters 
    prior_mean = [Normal(7.0, 10.0)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf)]
    prior_sigma = [Normal(2.0, 0.5)]

    # Cell-consant parameters 
    prior_kappa = [Normal(-1.0, 10.0), Normal(-3, 10.0)]
    
    pop_param_info = init_pop_param_info(prior_mean, 
                                         prior_scale, 
                                         prior_sigma, 
                                         prior_pop_kappa = prior_kappa,
                                         pos_pop_kappa = false, 
                                         log_pop_kappa = true,
                                         pos_pop_sigma=false)

    # Initial value for the individual parameters 
    ind_val = [6.2]
    ind_param_info = init_ind_param_info(ind_val, length(prior_mean), log_scale=true, pos_param=false)

    path_data = pwd() * "/Intermediate/Simulated_data/SSA/Multiple_ind/schlogl/schlogl.csv"
    file_loc = init_file_loc(path_data, "Schlogl_model", multiple_ind=true)

    # Filter information 
    dt = 5e-2   
    filter_opt = init_filter(ModDiffusion(), dt, rho=0.999) # Strong correlation 

    # Tuning of particles 
    tune_part_data = init_pilot_run_info(pop_param_info, 
                                         n_particles_pilot=500, 
                                         n_samples_pilot=5000, 
                                         rho_list=[0.999], 
                                         n_times_run_filter=50, 
                                         init_kappa=log.([1.8e-1, 2.5e-4]))

    # Sampler 
    cov_mat_ind = diagm([0.16])
    cov_mat_pop = diagm([0.25, 0.25, 0.5 / 10.0]) ./ 10
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat=cov_mat_ind, step_before_update=500)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=500)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalDiag(), n_warm_up=50)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances = [0.01, 0.01 ,0.01])

    # Tune particles for opt2 when running PEPSDI 
    if run_pilot == true
        tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
            file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)
    end

    if run_pilot == false
        exp_id = 1
        n_samples = 500000
        tmp = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
            filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=exp_id)
    end
    
end


# Function for predicting the outcome of increasing c1 
function predict_c1()

    P_mat = [1]
    my_model = init_sde_model(schlogl_alpha, 
                              schlogl_beta, 
                              schlogl_u0!, 
                              schlogl_h, 
                              schlogl_g, 
                              1, 1, P_mat) 

    # Parameter info for eta-parameters 
    prior_mean = [Normal(7.0, 10.0)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf)]
    prior_sigma = [Normal(2.0, 0.5)]

    # Cell-consant parameters 
    prior_kappa = [Normal(-1.0, 10.0), Normal(-3, 10.0)]
    
    pop_param_info = init_pop_param_info(prior_mean, 
                                         prior_scale, 
                                         prior_sigma, 
                                         prior_pop_kappa = prior_kappa,
                                         pos_pop_kappa = false, 
                                         log_pop_kappa = true,
                                         pos_pop_sigma=false)

    # Initial value for the individual parameters 
    ind_val = [6.2]
    ind_param_info = init_ind_param_info(ind_val, length(prior_mean), log_scale=true, pos_param=false)

    path_data = pwd() * "/Intermediate/Simulated_data/SSA/Multiple_ind/schlogl/schlogl.csv"
    file_loc = init_file_loc(path_data, "Schlogl_model", multiple_ind=true)

    # Filter information 
    dt = 5e-2   
    filter_opt = init_filter(ModDiffusion(), dt, rho=0.999) # Strong correlation 

    # Sampler 
    cov_mat_ind = diagm([0.16])
    cov_mat_pop = diagm([0.25, 0.25, 0.5 / 10.0]) ./ 10
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat=cov_mat_ind, step_before_update=500)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=500)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalDiag(), n_warm_up=50)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances = [0.01, 0.01 ,0.01])

    # Setting up to call pvc-function 

    # Observed data for each individual
    ind_data_arr = init_ind_data_arr(file_loc, filter_opt)

    n_individuals = length(ind_data_arr)
    pop_sampler = init_pop_sampler(pop_sampler_opt, n_individuals, length(prior_mean))

    # Individual data arrays 
    param_tmp = init_pop_param_curr(pop_param_info)
    dist_ind_param = calc_dist_ind_param(param_tmp, pop_sampler, 1)
    ind_param_info_arr = init_param_info_arr(ind_param_info, dist_ind_param, n_individuals)

    # Relevant dimensions 
    dim_mean = length(pop_param_info.init_pop_param_mean)
    n_individuals = length(ind_data_arr)
    n_kappa_sigma = length(vcat(pop_param_info.prior_pop_param_kappa, pop_param_info.prior_pop_param_sigma))

    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info)
    @printf("n_individuals = %d\n", n_individuals)
    mcmc_sampler_ind_arr = init_mcmc_arr(mcmc_sampler_ind, ind_param_info_arr, n_individuals)
    file_loc.dir_save *= "/" * mcmc_sampler_ind.name_sampler * "/"

    # Initialise the chains using starting values and set current values 
    n_samples = 500000
    mcmc_chains = init_chains(pop_param_info, ind_param_info_arr, n_individuals, n_samples)

    # Read data to populate chains (be able to write pvc-funciton)
    dir_data = pwd() * "/Intermediate/Multiple_individuals/Schlogl_model/Ram_sampler/Npart100_nsamp500000_corr0.999_exp_id1_run1/"
    data_mean = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Mean.csv", DataFrame))
    data_scale = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Scale.csv", DataFrame))
    data_kappa_sigma = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Kappa_sigma.csv", DataFrame))
    data_corr = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Corr.csv", DataFrame))

    mcmc_chains.mean .= data_mean'
    mcmc_chains.scale .= data_scale'
    mcmc_chains.kappa_sigma .= data_kappa_sigma'
    n_param = length(prior_mean)
    n_ind_param = 1
    for i in 1:n_samples
        @views mcmc_chains.corr[:, :, i] .= data_corr[((i-1)*n_ind_param+1):(i*n_ind_param), 1:n_ind_param]
    end

    @printf("Running pvc\n")

    pvc_mixed(my_model, mcmc_chains, ind_data_arr, pop_param_info, ind_param_info_arr, 
        file_loc, filter_opt, pop_sampler; n_runs=10000)

end


if ARGS[1] == "Run_pilot"
    run_inference_schlogl(true)

elseif ARGS[1] == "Run_inference"
    run_inference_schlogl(false)

elseif ARGS[1] == "Investigate_c1"
    predict_c1()

end