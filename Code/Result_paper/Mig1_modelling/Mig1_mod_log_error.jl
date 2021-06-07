using Random 
using Distributions
using LinearAlgebra
tmp = push!(LOAD_PATH, pwd() * "/Code")
using PEPSDI


# Propensity vector for model 1A in the paper 
function hvec_1A_mod(u, h_vec, p, t)
    
    c = p.c
    kappa = p.kappa
    co_val = p.covariates[1]

    c1 = c[1]
    c2 = c[2]
    c3 = kappa[1]
    c4 = kappa[2]
    Mig1c0 = c[3]
    Mig1n0 = c[4]

    c5 = 40.0
    c6 = c5 * Mig1n0 / Mig1c0

    if co_val == 2
        c1 *= kappa[3] 
        c2 *= kappa[4]
    end

    if t <= 1.5
        c1 *= 0.0
        c2 *= 0.0
    end

    Mig1c, Mig1n, X = u

    h_vec[1] = c1 * Mig1c
    h_vec[2] = c2 
    h_vec[3] = c3 * X
    h_vec[4] = c4 * X * Mig1n
    h_vec[5] = c5 * Mig1n
    h_vec[6] = c6 * Mig1c

end
# Propensity vector for model 1B in the paper 
function hvec_1B_mod(u, h_vec, p, t)
    
    c = p.c
    kappa = p.kappa
    co_val = p.covariates[1]

    c1 = c[1]
    c2 = c[2]
    c3 = kappa[1]
    c4 = kappa[2]
    Mig1c0 = c[3]
    Mig1n0 = c[4]

    c5 = 40.0
    c6 = c5 * Mig1n0 / Mig1c0

    if t <= 1.5
        c1 *= 0.0
        c2 *= 0.0
    end

    Mig1c, Mig1n, X = u

    h_vec[1] = c1 * Mig1c
    h_vec[2] = c2 
    h_vec[3] = c3 * X
    h_vec[4] = c4 * X * Mig1n
    h_vec[5] = c5 * Mig1n
    h_vec[6] = c6 * Mig1c

end
# Propensity vector for model 2A in the paper 
function hvec_2A_mod(u, h_vec, p, t)
    
    c = p.c
    kappa = p.kappa
    co_val = p.covariates[1]

    c1 = c[1]
    c2 = c[2]
    c3 = kappa[1]
    c4 = kappa[2]
    c5 = kappa[3]
    c8 = kappa[4]
    Mig1c0 = c[3]
    Mig1n0 = c[4]

    if co_val == 2
        c1 *= kappa[5] 
        c2 *= kappa[6]
    end

    if t <= 1.5
        c1 *= 0.0
        c2 *= 0.0
    end

    c6 = 40.0
    c7 = c6 * Mig1n0 / Mig1c0

    Reg1, Mig1c, Mig1n, X = u

    h_vec[1] = c1 
    h_vec[2] = c2 
    h_vec[3] = c3 * Mig1c * Reg1
    h_vec[4] = c4  * X
    h_vec[5] = c5 * X * Mig1n
    h_vec[6] = c6 * Mig1n
    h_vec[7] = c7 * Mig1c
    h_vec[8] = c8 * Reg1
end
# Propensity vector for model 2B in the paper 
function hvec_2B_mod(u, h_vec, p, t)
    
    c = p.c
    kappa = p.kappa

    c1 = c[1]
    c2 = c[2]
    c3 = kappa[1]
    c4 = kappa[2]
    c5 = kappa[3]
    c8 = kappa[4]
    Mig1c0 = c[3]
    Mig1n0 = c[4]

    if t <= 1.5
        c1 *= 0.0
        c2 *= 0.0
    end

    c6 = 40.0
    c7 = c6 * Mig1n0 / Mig1c0

    Reg1, Mig1c, Mig1n, X = u

    h_vec[1] = c1 
    h_vec[2] = c2 
    h_vec[3] = c3 * Mig1c * Reg1
    h_vec[4] = c4  * X
    h_vec[5] = c5 * X * Mig1n
    h_vec[6] = c6 * Mig1n
    h_vec[7] = c7 * Mig1c
    h_vec[8] = c8 * Reg1
end


# Calc initial values for model structure 1
function calc_x0_struct1(x0, p)

    c = p.c

    Mig1c0 = c[3]
    Mig1n0 = c[4]

    x0[1] = round(Mig1c0)
    x0[2] = round(Mig1n0)
    x0[3] = 0
end
# Calc initial values for model structure 2 
function calc_x0_struct2(x0, p)

    c = p.c

    Mig1c0 = c[3]
    Mig1n0 = c[4]

    x0[1] = 0
    x0[2] = round(Mig1c0)
    x0[3] = round(Mig1n0)
    x0[4] = 0
end


# Calculate observation, y, for model structure 1
function calc_obs_struct1(y, u, p, t)

    y[1] = log(u[2] / u[1])
end
# Calculate observation, y, for model structure 2 
function calc_obs_struct2(y, u, p, t)

    y[1] = log(u[3] / u[2])
end


# Probability to observe observation model 
function model_prob_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])

    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)
    return exp(prob)
end


function model_1A(n_samples; pilot=false)

    
    S_left = convert(Array{Int16, 2}, [1 0 0; 0 0 0; 0 0 1; 0 1 1; 0 1 0; 1 0 0])
    S_right = convert(Array{Int16, 2}, [0 1 0; 0 0 1; 0 0 0; 1 0 1; 1 0 0; 0 1 0])
    my_model = PoisonModel(hvec_1A_mod, 
                           calc_x0_struct1, 
                           calc_obs_struct1, 
                           model_prob_obs,
                           UInt16(3), UInt16(1), UInt16(6), S_left - S_right)

    prior_mean = [Normal(3.3, 2.0), 
                  Normal(2.0, 3.0),
                  Normal(7.3, 0.2),
                  Normal(4.6, 0.2)]
    prior_kappa = [Normal(-6, 3.0), Normal(0.0, 3), Normal(0.6, 0.5), Normal(0.0, 2.0)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   Gamma(0.3, 0.3),
                   Gamma(0.3, 0.3)]
    prior_corr = LKJ(4, 0.1)
    prior_sigma = [Gamma(0.5, 0.5)]

    # Parameter information 
    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, 
            prior_pop_corr=prior_corr, prior_pop_kappa=prior_kappa, log_pop_kappa=true, pos_pop_kappa=false)
    ind_param_info = init_ind_param_info("mean", length(prior_mean), 
        log_scale=true, pos_param=false)

    # Observed data
    path_data = pwd() * "/Intermediate/Experimental_data/Data_fructose/Fructose_data_log.csv"
    file_loc = init_file_loc(path_data, "Mig1_mod_1A_log", multiple_ind=true, cov_name = ["fruc"], cov_val=[2.0, 0.05], dist_id=[1, 31])

    # Filter options 
    dt = 5e-3; rho = 0.999
    filter_opt = init_filter(BootstrapPois(), dt, n_particles=100, rho=rho)

    # Sampler options 
    cov_mat = diagm([0.2, 0.2, 0.01, 0.01])
    cov_mat_pop = diagm([0.2, 0.2, 0.2, 0.2, 0.01])
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat = cov_mat, step_before_update=200)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=200)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalLKJ(), n_warm_up=50)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances=[0.01, 0.01, 0.01, 0.01, 0.0001])

    # Tuning of particles 
    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=200, n_samples_pilot=1500, 
        rho_list=[0.999], n_times_run_filter=40)

    if pilot == true
        tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
            file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)

        return 0
    else
        stuff = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
            filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=1)
        return stuff
    end

end


function model_1B(n_samples; pilot=false)

    
    S_left = convert(Array{Int16, 2}, [1 0 0; 0 0 0; 0 0 1; 0 1 1; 0 1 0; 1 0 0])
    S_right = convert(Array{Int16, 2}, [0 1 0; 0 0 1; 0 0 0; 1 0 1; 1 0 0; 0 1 0])
    my_model = PoisonModel(hvec_1B_mod, 
                           calc_x0_struct1, 
                           calc_obs_struct1, 
                           model_prob_obs,
                           UInt16(3), UInt16(1), UInt16(6), S_left - S_right)

    prior_mean = [Normal(3.3, 0.5), 
                  Normal(3.7, 0.5),
                  Normal(2.0, 3.0),
                  Normal(2.0, 3.0),
                  Normal(7.3, 0.2),
                  Normal(4.6, 0.2)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf),
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf),
                   Gamma(0.3, 0.3),
                   Gamma(0.3, 0.3)]
    prior_kappa = [Normal(-6, 3.0), Normal(0.0, 3)]
    prior_corr = LKJ(4, 0.1)
    prior_sigma = [Gamma(0.4, 0.4)]

    # Parameter information 
    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, 
            prior_pop_corr=prior_corr, prior_pop_kappa=prior_kappa, log_pop_kappa=true, pos_pop_kappa=false)
    ind_param_info = init_ind_param_info("mean", 4, log_scale=true, pos_param=false)

    # Observed data
    path_data = pwd() * "/Intermediate/Experimental_data/Data_fructose/Fructose_data_log.csv"
    file_loc = init_file_loc(path_data, "Mig1_mod_1B_log", multiple_ind=true, cov_name = ["fruc"], cov_val=[2.0, 0.05], dist_id=[1, 31])

    # Filter options 
    dt = 5e-3; rho = 0.999
    filter_opt = init_filter(BootstrapPois(), dt, n_particles=100, rho=rho)

    # Sampler options 
    cov_mat = diagm([0.2, 0.2, 0.01, 0.01])
    cov_mat_pop = diagm([0.2, 0.2, 0.01])
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat = cov_mat, step_before_update=200)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=200)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances=[0.01, 0.01, 0.0001])

    index_one = [1, 3, 5, 6]
    index_two = [2, 4, 5, 6]
    tag_one = collect(22:37)
    tag_two = collect(1:21)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalLjkTwo(), index_one, index_two, 
        tag_one, tag_two, 4, n_warm_up=100)

    # Tuning of particles 
    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=200, n_samples_pilot=1500, 
        rho_list=[0.999], n_times_run_filter=40)

    if pilot == true
        tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
            file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)

        return 0
    else
        stuff = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
            filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=1)
        return stuff
    end

end


function model_1B(n_samples; pilot=false)

    
    S_left = convert(Array{Int16, 2}, [0 0 0 0; 0 0 0 0; 1 1 0 0; 0 0 0 1; 0 0 1 1; 0 0 1 0; 0 1 0 0; 1 0 0 0])
    S_right = convert(Array{Int16, 2}, [1 0 0 0; 0 0 0 1; 1 0 1 0; 0 0 0 0; 0 1 0 1; 0 1 0 0; 0 0 1 0; 0 0 0 0])
    my_model = PoisonModel(hvec_2A_mod, 
                           calc_x0_struct2, 
                           calc_obs_struct2, 
                           model_prob_obs,
                           UInt16(4), UInt16(1), UInt16(8), S_left - S_right)

    prior_mean = [Normal(6.9, 0.5), 
                  Normal(2.0, 3.0),
                  Normal(7.3, 0.2),
                  Normal(4.6, 0.2)]
    prior_kappa = [Normal(-4, 3.0), Normal(-6, 3.0), Normal(0.0, 3), Normal(-0.28, 0.5), Normal(0.6, 0.5), Normal(0.0, 2.0)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   Gamma(0.3, 0.3),
                   Gamma(0.3, 0.3)]
    prior_corr = LKJ(4, 0.1)
    prior_sigma = [Gamma(0.5, 0.5)]


    # Parameter information 
    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, 
            prior_pop_corr=prior_corr, prior_pop_kappa=prior_kappa, log_pop_kappa=true, pos_pop_kappa=false, 
            init_pop_kappa = log.([0.00922, 0.00205, 1.18, 0.65, exp(0.0), exp(-0.0)]), 
            init_pop_mean = log.([900, 9.12, 1500, 55.9]))
    ind_param_info = init_ind_param_info("mean", length(prior_mean), 
        log_scale=true, pos_param=false)

    # Observed data
    path_data = pwd() * "/Intermediate/Experimental_data/Data_fructose/Fructose_data_log.csv"
    file_loc = init_file_loc(path_data, "Mig1_mod_2A_log", multiple_ind=true, cov_name = ["fruc"], cov_val=[2.0, 0.05], dist_id=[1, 31])

    # Filter options 
    dt = 5e-3; rho = 0.999
    filter_opt = init_filter(BootstrapPois(), dt, n_particles=100, rho=rho)

    # Sampler options 
    cov_mat = diagm([0.2, 0.2, 0.01, 0.01])
    cov_mat_pop = diagm([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.01])
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat = cov_mat, step_before_update=200)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=200)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalLKJ(), n_warm_up=50)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0001])

    # Tuning of particles 
    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=200, n_samples_pilot=1500, 
        rho_list=[0.999], n_times_run_filter=40)

    if pilot == true
        tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
            file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)

        return 0
    else
        stuff = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
            filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=1)
        return stuff
    end

end


function model_2B(n_samples; pilot=false)

    
    S_left = convert(Array{Int16, 2}, [0 0 0 0; 0 0 0 0; 1 1 0 0; 0 0 0 1; 0 0 1 1; 0 0 1 0; 0 1 0 0; 1 0 0 0])
    S_right = convert(Array{Int16, 2}, [1 0 0 0; 0 0 0 1; 1 0 1 0; 0 0 0 0; 0 1 0 1; 0 1 0 0; 0 0 1 0; 0 0 0 0])
    my_model = PoisonModel(hvec_2B_mod, 
                           calc_x0_struct2, 
                           calc_obs_struct2, 
                           model_prob_obs,
                           UInt16(4), UInt16(1), UInt16(8), S_left - S_right)

    prior_mean = [Normal(6.9, 0.5), 
                  Normal(7.3, 0.5),
                  Normal(2.0, 3.0),
                  Normal(2.0, 3.0),
                  Normal(7.3, 0.2),
                  Normal(4.6, 0.2)]
    prior_scale = [truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf), 
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf),
                   truncated(Cauchy(0.0, 2.5), 0.0, Inf),
                   Gamma(0.3, 0.3),
                   Gamma(0.3, 0.3)]
    prior_sigma = [Gamma(0.3, 0.3)]
    prior_corr = LKJ(4, 0.1)
    prior_kappa = [Normal(-4, 3.0), Normal(-6, 3.0), Normal(0.0, 3), Normal(-0.28, 0.5)]

    # Parameter information 
    pop_param_info = init_pop_param_info(prior_mean, prior_scale, prior_sigma, 
            prior_pop_corr=prior_corr, prior_pop_kappa=prior_kappa, log_pop_kappa=true, pos_pop_kappa=false, 
            init_pop_kappa = log.([0.00922, 0.00205, 1.18, 0.65]),
            init_pop_mean = log.([900, 1200, 9.12, 9.12, 1500, 55.9]))
    ind_param_info = init_ind_param_info("mean", 4, 
        log_scale=true, pos_param=false)

    # Observed data
    path_data = pwd() * "/Intermediate/Experimental_data/Data_fructose/Fructose_data_log.csv"
    file_loc = init_file_loc(path_data, "Mig1_mod_2B_log", multiple_ind=true, cov_name = ["fruc"], cov_val=[2.0, 0.05], dist_id=[1, 31])

    # Filter options 
    dt = 5e-3; rho = 0.999
    filter_opt = init_filter(BootstrapPois(), dt, n_particles=100, rho=rho)

    # Sampler options 
    cov_mat = diagm([0.2, 0.2, 0.01, 0.01])
    cov_mat_pop = diagm([0.2, 0.2, 0.2, 0.2, 0.01])
    mcmc_sampler_ind = init_mcmc(RamSampler(), ind_param_info, cov_mat = cov_mat, step_before_update=200)
    mcmc_sampler_pop = init_mcmc(RamSampler(), pop_param_info, cov_mat=cov_mat_pop, step_before_update=200)
    kappa_sigma_sampler_opt = init_kappa_sigma_sampler_opt(KappaSigmaNormal(), variances=[0.01, 0.01, 0.01, 0.01, 0.0001])

    index_one = [1, 3, 5, 6]
    index_two = [2, 4, 5, 6]
    tag_one = collect(22:37)
    tag_two = collect(1:21)
    pop_sampler_opt = init_pop_sampler_opt(PopNormalLjkTwo(), index_one, index_two, 
        tag_one, tag_two, 4, n_warm_up=100)

    # Tuning of particles 
    tune_part_data = init_pilot_run_info(pop_param_info, n_particles_pilot=200, n_samples_pilot=1500, 
        rho_list=[0.999], n_times_run_filter=40)

    if pilot == true
        tune_particles_opt2(tune_part_data, pop_param_info, ind_param_info, 
            file_loc, my_model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt)

        return 0
    else
        stuff = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, my_model, 
            filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, kappa_sigma_sampler_opt, pilot_id=1)
        return stuff
    end

end


if ARGS[1] == "Model_1A"
    model_1A(100; pilot=true)
    model_1A(40000; pilot=false)
end


if ARGS[1] == "Model_1B"
    model_1B(100; pilot=true)
    model_1B(40000; pilot=false)
end


if ARGS[1] == "Model_2A"
    model_2A(100; pilot=true)
    model_2A(40000; pilot=false)
end


if ARGS[1] == "Model_2B"
    model_2B(100; pilot=true)
    model_2B(10000; pilot=false)
end



