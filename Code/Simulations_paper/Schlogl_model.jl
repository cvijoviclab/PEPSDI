#= 
    Running the inference for the Schlögl-model. For a closer description 
    of each step see the notebook in the Code/Examples notebook. 
=# 


using Distributions # For placing priors 
using Random # For setting seed 
using LinearAlgebra # For matrix operations 
using Plots
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

    
function run_inference_schlogl()
    
    
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
    prior_sigma = [Normal(2.0, 0.1)]

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
    tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
        file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)
    
    exp_id = 1
    n_samples = 500000
    tmp = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
        filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=exp_id)
    
end


run_inference_schlogl()