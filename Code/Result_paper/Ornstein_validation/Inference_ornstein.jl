#= 
    Running the inference for the Ornstein-model to validate implementation against 
    Wiqvist et al.    
=# 


using Distributions # For placing priors 
using Random # For setting seed 
using LinearAlgebra # For matrix operations 
using Plots
tmp = push!(LOAD_PATH, pwd() * "/Code") # Push PEPSDI into load-path 
using PEPSDI # Load PEPSDI 


# Model equations 
include(pwd() * "/Code/Models/Ornstein.jl")


# Run the validation for the Ornstein-uhlenbeck model using PEPSDI running option 1. 
function run_ornstein_opt1()

    Random.seed!(1234)
    rho = 0.99

    sde_mod = init_sde_model(alpha_ornstein_full, 
                             beta_ornstein_full, 
                             calc_x0_ornstein!, 
                             ornstein_obs, 
                             prob_ornstein_full,
                             1, 1)

    # Priors as in Wiqvist et al.
    prior_mean = [Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(0.0, 1.0)]
    prior_scale = [Gamma(2.0, 1.0), Gamma(2.0, 0.5), Gamma(2.0, 1.0)]
    prior_sigma = [Gamma(1.0, 0.4)]

    # Same parameter options as Wiqvist et al. 
    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, log_pop_kappa=true, pos_pop_kappa=false, precision_scale=true)
    ind_param_info = init_ind_param_info("mean", length(prior_mean), 
        log_scale=true, pos_param=false)

    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Multiple_ind/Ornstein_val/Ornstein_val.csv"
    file_loc = init_file_loc(path_data, "Ornstein_val_opt1", multiple_ind=true)

    # Filter information 
    dt = 1e-2
    filter_opt = init_filter(BootstrapEm(), dt, rho=rho)

    # Sampler 
    mcmc_sampler_ind = init_mcmc(GenAmSampler(), ind_param_info)
    mcmc_sampler_pop = init_mcmc(GenAmSampler(), pop_param_info)
    pop_sampler_opt = init_pop_sampler_opt(PopOrnstein())

    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=1500, n_samples_pilot=5000, 
        rho_list=[0.99], n_times_run_filter=100, init_sigma = [0.4])

    # Function for tuning the number of particles for a mixed-effects model. 
    tune_particles_opt1(tune_part_data, pop_param_info, ind_param_info, 
        file_loc, sde_mod, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt)

    n_samples = 60000
    stuff = run_PEPSDI_opt1(n_samples, pop_param_info, ind_param_info, file_loc, sde_mod, 
        filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, pilot_id=1)

end


# Run the validation for the Ornstein-uhlenbeck model using PEPSDI running option 1. 
function run_ornstein_opt2()

    Random.seed!(1234)
    rho = 0.99

    sde_mod = init_sde_model(alpha_ornstein_full, 
                             beta_ornstein_full, 
                             calc_x0_ornstein!, 
                             ornstein_obs, 
                             prob_ornstein_full,
                             1, 1)

    # Priors as in Wiqvist et al.
    prior_mean = [Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(0.0, 1.0)]
    prior_scale = [Gamma(2.0, 1.0), Gamma(2.0, 0.5), Gamma(2.0, 1.0)]
    prior_sigma = [Gamma(1.0, 0.4)]

    # Same parameter options as Wiqvist et al. 
    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, log_pop_kappa=true, pos_pop_kappa=false, precision_scale=true)
    ind_param_info = init_ind_param_info("mean", length(prior_mean), 
        log_scale=true, pos_param=false)

    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Multiple_ind/Ornstein_val/Ornstein_val.csv"
    file_loc = init_file_loc(path_data, "Ornstein_val_opt2", multiple_ind=true)

    # Filter information 
    dt = 1e-2
    filter_opt = init_filter(BootstrapEm(), dt, rho=rho)

    # Sampler 
    mcmc_sampler_ind = init_mcmc(GenAmSampler(), ind_param_info)
    mcmc_sampler_pop = init_mcmc(GenAmSampler(), pop_param_info)
    pop_sampler_opt = init_pop_sampler_opt(PopOrnstein())
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances = [0.01])

    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=1500, n_samples_pilot=5000, 
        rho_list=[0.99], n_times_run_filter=100, init_sigma = [0.4])

    # Function for tuning the number of particles for a mixed-effects model. 
    tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
        file_loc, sde_mod, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)

    n_samples = 60000
    stuff = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, sde_mod, 
        filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=1)

end


if ARGS[1] == "PEPSDI_opt1"
    run_ornstein_opt1()

elseif ARGS[1] == "PEPSDI_opt2"
    run_ornstein_opt2()

end
