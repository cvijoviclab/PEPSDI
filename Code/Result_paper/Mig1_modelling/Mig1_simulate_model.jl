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
    dir_save = pwd() * "/Intermediate/Multiple_individuals/Mig1_mod_2B/Ram_sampler/Npart100_nsamp40000_corr0.999_exp_id4_run1/pred/c1/Ram_sampler"
    if !isdir(dir_save)
        mkpath(dir_save)
    end
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

    @printf("Running pvc\n")

    cov_val = file_loc.cov_val
    n_loops = length(file_loc.cov_val)
    cov_val_name = file_loc.cov_name[1]
    cov_tag = "_" * cov_val_name .* string.(cov_val)

    pvc_mixed(my_model, mcmc_chains, ind_data_arr, pop_param_info, ind_param_info_arr, 
        file_loc, filter_opt, pop_sampler; n_runs=10000, cov_val=cov_val[1], name_cov=cov_val_name)

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