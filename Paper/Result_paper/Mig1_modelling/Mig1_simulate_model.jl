#= 
    Simulating the best Mig1-model to investigate effect of removing correlation 
    and to see the effect of the c1-parameter on Mig1-nuclear import. 

    Args:
        ARGS[1] : Option to run (Investigate_c1, No_corr)
=# 


using Random 
using Distributions
using LinearAlgebra
using Printf
using DataFrames
using CSV
tmp = push!(LOAD_PATH, pwd() * "/Code")
using PEPSDI

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

# Calculate observation, y, for model structure 2 
function calc_obs_struct2(y, u, p, t)

    y[1] = u[3] / u[2]
end

# Probability to observe observation model 
function model_prob_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])

    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)
    return exp(prob)
end


# Reach the posterior visual check function to simulate effect of c1 
function predict_c1_mig1_model()

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
    path_data = pwd() * "/Intermediate/Experimental_data/Data_fructose/Fructose_data.csv"
    file_loc = init_file_loc(path_data, "Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/pred/c1/", multiple_ind=true, cov_name = ["fruc"], cov_val=[2.0, 0.05], dist_id=[1, 31])

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
    n_samples = 40000
    mcmc_chains = init_chains(pop_param_info, ind_param_info_arr, n_individuals, n_samples)

    # Read data to populate chains (be able to write pvc-funciton)
    dir_data = pwd() * "/Intermediate//Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/"
    data_mean = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Mean.csv", DataFrame))
    data_scale = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Scale.csv", DataFrame))
    data_kappa_sigma = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Kappa_sigma.csv", DataFrame))
    data_corr = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Corr.csv", DataFrame))

    mcmc_chains.mean .= data_mean'
    mcmc_chains.scale .= data_scale'
    mcmc_chains.kappa_sigma .= data_kappa_sigma'
    n_param = length(prior_mean)
    n_ind_param = 4
    for i in 1:n_samples
        @views mcmc_chains.corr[:, :, i] .= data_corr[((i-1)*n_ind_param+1):(i*n_ind_param), 1:n_ind_param]
    end

    @printf("Running prediciton of c1\n")

    i = 1

    
    burn_in = 0.2
    n_runs = 132 * 10000

    # Setup for drawing model parameters 
    n_param = size(mcmc_chains.mean)[1]
    n_samples = size(mcmc_chains.mean)[2]
    min_sample_use = convert(Int, floor(n_samples*burn_in))
    indices_use = min_sample_use:n_samples
    n_samples_use = length(indices_use)
    index_sample = rand(DiscreteUniform(1, n_samples_use), n_runs)
    

    # Array for holding model-parameters 
    log_kappa = pop_param_info.log_pop_param_kappa
    log_sigma = pop_param_info.log_pop_param_sigma
    n_kappa_sigma = length(vcat(log_kappa, log_sigma))
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:n_kappa_sigma]
    kappa_index = .!sigma_index

    # Correct transformation of individual parameters 
    log_ind_param = ind_param_info_arr[1].log_ind_param

    # Mcmc-chains to draw for 
    pop_mean = mcmc_chains.mean
    pop_scale = mcmc_chains.scale
    pop_corr = mcmc_chains.corr
    pop_kappa = mcmc_chains.kappa_sigma[kappa_index, :]
    pop_sigma = mcmc_chains.kappa_sigma[sigma_index, :]

    # Array for model-parameters 
    mod_param_arr = init_model_parameters_arr(ind_param_info_arr, pop_param_info, my_model, length(ind_data_arr), ind_data_arr)
    model_param = mod_param_arr[1]
    model_param.covariates .= 2.0
    dist_id = 1

    n_ind_param = length(ind_param_info_arr[1].log_ind_param)
    ind_param_arr = Array{FLOAT, 2}(undef, (n_runs, n_ind_param))

    val_save = Array{Float64, 2}(undef, (n_runs, 4))

    # Selecting random samples from posterior 
    i = 1
    while i < n_runs

        i_sample = indices_use[index_sample[i]]
        mean_vec = pop_mean[:, i_sample]
        scale_vec = pop_scale[:, i_sample]
        corr_mat = pop_corr[:, :, i_sample]

        dist_ind_param = calc_dist_ind_param(init_pop_param_curr(mean_vec, scale_vec, corr_mat), pop_sampler, dist_id)

        # Draw new individual parameters and populate model-param-struct 
        new_param = rand(dist_ind_param, 1)[:, 1]
        map_sigma_to_mod_param_mixed!(model_param, log_sigma, pop_sigma[:, i_sample])
        map_kappa_to_mod_param_mixed!(model_param, log_kappa, pop_kappa[:, i_sample])
        update_ind_param_ind!(model_param.individual_parameters, new_param, log_ind_param)

        u0_vec = Array{Int32, 1}(undef, my_model.dim)
        my_model.calc_x0!(u0_vec, model_param.individual_parameters)

        local t_vec, y_vec
        try 
            t_vec, y_vec = solve_poisson_model(my_model, (0.0, 15.0), model_param.individual_parameters, 5e-3)
        catch
            continue
        end

        # Save the relevant values
        val_save[i, 1] = model_param.individual_parameters.c[1]
        val_save[i, 2] = y_vec[3, 1000] / y_vec[2, 1000]
        val_save[i, 3] = y_vec[3, 2000] / y_vec[2, 2000]
        val_save[i, 4] = y_vec[3, end] / y_vec[2, end]

        i += 1
    end

    data_save = convert(DataFrame, val_save)
    rename!(data_save, ["c1", "t1", "t2", "t3"])
    dir_save = pwd() * "/Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/"
    CSV.write(dir_save * "Pred_c1.csv", data_save)

end


function predict_no_corr()

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
    path_data = pwd() * "/Intermediate/Experimental_data/Data_fructose/Fructose_data.csv"
    dir_save = pwd() * "/Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/pred/no_corr/Ram_sampler"
    if !isdir(dir_save)
        mkpath(dir_save)
    end
    file_loc = init_file_loc(path_data, "Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/pred/no_corr/", multiple_ind=true, cov_name = ["fruc"], cov_val=[2.0, 0.05], dist_id=[1, 31])

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
    n_samples = 40000
    mcmc_chains = init_chains(pop_param_info, ind_param_info_arr, n_individuals, n_samples)

    # Read data to populate chains (be able to write pvc-funciton)
    dir_data = pwd() * "/Intermediate//Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/"
    data_mean = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Mean.csv", DataFrame))
    data_scale = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Scale.csv", DataFrame))
    data_kappa_sigma = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Kappa_sigma.csv", DataFrame))
    data_corr = convert(Array{FLOAT, 2}, CSV.read(dir_data * "Corr.csv", DataFrame))

    mcmc_chains.mean .= data_mean'
    mcmc_chains.scale .= data_scale'
    mcmc_chains.kappa_sigma .= data_kappa_sigma'
    n_param = length(prior_mean)
    n_ind_param = 4
    for i in 1:n_samples
        @views mcmc_chains.corr[:, :, i] .= convert(Array{Float64, 2}, I(n_ind_param))
    end

    @printf("Running pvc no correlation\n")
    run_pvc_mixed(my_model, mcmc_chains, ind_data_arr, pop_param_info, ind_param_info_arr, file_loc, filter_opt, pop_sampler)

end



if ARGS[1] == "Investigate_c1"
    predict_c1_mig1_model()

elseif ARGS[1] == "No_corr"
    predict_no_corr()

end
