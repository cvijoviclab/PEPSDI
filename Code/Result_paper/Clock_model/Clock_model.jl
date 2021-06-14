#= 
    Running the inference for the Clock-model. For a closer description 
    of each step see the notebook in the Code/Examples notebook. 
=# 


using Distributions # For placing priors 
using Random # For setting seed 
using LinearAlgebra # For matrix operations 
using Plots
tmp = push!(LOAD_PATH, pwd() * "/Code") # Push PEPSDI into load-path 
using PEPSDI # Load PEPSDI 

    
# Defining the model 

function clock_h_vec!(u, h_vec, p, t)
    c = p.c
    const_term = 0.2617993877991494
    h_vec[1] = c[1] * (1 + sin(const_term * t))
    h_vec[2] = c[2] * u[1]
    h_vec[3] = c[3] * u[1]
    h_vec[4] = c[4] * u[2]    
end
# The extrande algorithms requires that the maximum potential propensity can be calculated. 
function clock_h_vec_max!(u, h_vec, p, t_start, t_end)
    c = p.c
    
    h_vec[1] = c[1] * 2
    h_vec[2] = c[2] * u[1]
    h_vec[3] = c[3] * u[1]
    h_vec[4] = c[4] * u[2]    
end


function clock_u0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 0
    x0[2] = 0
end


function clock_h(y_mod, u, p, t)
    y_mod[1] = u[2]
end


function clock_g(y_obs, y_mod, error_param, t, dim_obs)

    # Since y_obs ~ N(y_mod, xi^2) the likelihood can be calculated 
    # via the normal distribution. Perform calculations on log-scale 
    # for stabillity. 

    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)
    
    return exp(prob)
end


function run_clock_model(run_pilot::Bool)

    rho = 0.0

    # Set up extrande model 
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])    
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model = ExtrandModel(clock_h_vec!, clock_h_vec_max!, clock_u0!, clock_h, clock_g, 2, 1, 4, S_left - S_right)

    # Priors for η
    prior_mean = [Normal(0.0, 2.0),
                  Normal(2.0, 5.0), 
                  Normal(0.0, 5.0), 
                  Normal(0.0, 5.0)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf)]
    prior_corr = LKJ(4, 3.0)
    
    # Priors for ξ
    prior_sigma = [Normal(2.0, 0.5)]
    

    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, 
        prior_pop_corr=prior_corr)

    # Individual parameters inferred on log-scale 
    ind_param_info = init_ind_param_info("mean", 4, log_scale=true, pos_param=false)

    # Data for 40 individuals 
    path_data = pwd() * "/Intermediate/Simulated_data/SSA/Multiple_ind/Clock/Clock.csv"
    file_loc = init_file_loc(path_data, "Clock_model", multiple_ind=true)

    # Filter information, extrande filter here we cannot correlate particles
    filter_opt = init_filter(BootstrapExtrand())

    # Samplers 
    cov_mat_ind = diagm([0.1, 0.1, 0.1, 0.1])
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat=cov_mat_ind)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=diagm([0.1]))
    pop_sampler_opt = init_pop_sampler_opt(PopNormalLKJ(), n_warm_up=20)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal())

    # Options for pilot-run 
    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=2000, n_samples_pilot=5000, 
        rho_list=[0.0], n_times_run_filter=100, init_sigma = [0.2], init_mean=log.([2.0, 35.0, 1.0, 2.0]))
        
    if run_pilot == true
        tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
            file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)
    end
    
    if run_pilot == false
        exp_id = 1
        n_samples = 50000
        tmp = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
            filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=exp_id)
    endn

end


if ARGS[1] == "Run_pilot"
    run_clock_model(true)

elseif ARGS[1] == "Run_inference"
    run_clock_model(false)

end

