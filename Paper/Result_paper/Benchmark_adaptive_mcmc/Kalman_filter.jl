using Distributions
using Printf
using Random
using ProgressMeter
using LinearAlgebra
using CSV 
using DataFrames
using Plots 
using StatsPlots
tmp = push!(LOAD_PATH, pwd() * "/src") # Push PEPSDI into load-path 
using PEPSDI # Load PEPSDI 


# Code taken from Wiqvist, Kalman filter for the Ornstein-Uhlenbeck model 
function kalman_filter(y::Array{Float64, 1}, σ_ϵ::Float64, log_c::Array{Float64, 1}, dt::Float64)::Float64

    T = length(y)

    #println(loglik_est[m])
    θ_1::Float64 = exp(log_c[1])
    θ_2::Float64 = exp(log_c[2])
    θ_3::Float64 = exp(log_c[3])

    # set model
    B::Float64 = θ_1*θ_2
    A::Float64 = -θ_1
    σ::Float64 = θ_3
    C = 1
    S::Float64 = σ_ϵ^2

    # start values for the kalman filter
    P_start = var(y)
    x_hat_start = 0.0

    P_k = P_start
    x_k = x_hat_start

    loglik_est::Float64 = 0.0

    # main loop
    for k = 1:T

        x_k = exp(A*dt)*x_k + (1/A)*(exp(dt*A)-1)*B
        P_k = exp(A*dt)*P_k*exp(A*dt) + σ^2*(1/(2*A))*(exp(2*A*dt)-1)

        ϵ_k = y[k]-C*x_k
        R_k = C*P_k*C + S
        K = P_k*C*inv(R_k)
        x_k = x_k + K*ϵ_k
        P_k = P_k - K*R_k*K

        loglik_est = loglik_est -0.5*(log(det(R_k)) + ϵ_k*inv(R_k)*ϵ_k)

    end

    return loglik_est
end


function check_implementation()

    data_set = "123"
    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Ornstein/Ornstein" * data_set * ".csv"
    file_loc = init_file_loc(path_data, "Ornstein_test_proposal")
    data_obs = CSV.read(file_loc.path_data, DataFrame)
    y_data = data_obs["obs"]
    t_vec = data_obs["time"]
    delta_t = t_vec[2] - t_vec[1]

    log_c = [-0.7, 2.3, -0.9]
    sigma = 0.3

    param_vec1 = -1.5:0.05:0.0
    param_vec2 = 0.0:0.05:3.0
    param_vec3 = -2.0:0.05:0.0

    log_lik_vec1 = Array{Float64, 1}(undef, length(param_vec1))
    log_lik_vec2 = Array{Float64, 1}(undef, length(param_vec2))
    log_lik_vec3 = Array{Float64, 1}(undef, length(param_vec3))

    for i in 1:length(param_vec1)
        log_c[1] = param_vec1[i]
        log_lik_vec1[i] = kalman_filter(y_data, sigma, log_c, delta_t)
    end
    log_c = [-0.7, 2.3, -0.9]
    for i in 1:length(param_vec2)
        log_c[2] = param_vec2[i]
        log_lik_vec2[i] = kalman_filter(y_data, sigma, log_c, delta_t)
    end
    log_c = [-0.7, 2.3, -0.9]
    for i in 1:length(param_vec3)
        log_c[3] = param_vec3[i]
        log_lik_vec3[i] = kalman_filter(y_data, sigma, log_c, delta_t)
    end

    p1 = plot(param_vec1, log_lik_vec1)
    p1 = vline!([-0.7])
    p1 = title!("Parameter = c1")

    p2 = plot(param_vec2, log_lik_vec2)
    p2 = vline!([2.3])
    p2 = title!("Parameter = c2")

    p3 = plot(param_vec3, log_lik_vec3)
    p3 = vline!([-0.9])
    p3 = title!("Parameter = c3")
    
    return p1, p2, p3
end


function run_mcmc(n_samples, mcmc_sampler, param_info, y_data, delta_t)
    
    # Information regarding number of parameters to infer 
    n_param_infer = length(param_info.prior_ind_param) + length(param_info.prior_error_param)
    i_range_ind = 1:length(param_info.prior_ind_param)
    i_range_error = length(param_info.prior_ind_param)+1:n_param_infer
    prior_dists = vcat(param_info.prior_ind_param, param_info.prior_error_param)
    positive_proposal = vcat(param_info.ind_param_pos, param_info.error_param_pos)

    # Struct for model parameters 
    mod_param = DynModInput(param_info.init_ind_param, param_info.init_error_param, [0.0])
        
    # Storing the log-likelihood values 
    log_lik_val = Array{FLOAT, 1}(undef, n_samples)

    # Storing mcmc-chain 
    param_sampled = Array{FLOAT, 2}(undef, (n_param_infer, n_samples))
    param_sampled[:, 1] = vcat(param_info.init_ind_param, param_info.init_error_param)
    x_prop = Array{FLOAT, 1}(undef, n_param_infer)
    x_old = param_sampled[1:n_param_infer, 1]

    log_c = Array{Float64, 1}(undef, 3)
    log_c .= x_old[1:3]
    sigma = x_old[4]

    # Calculate likelihood, jacobian and prior for initial parameters 
    log_prior_old = calc_log_prior(x_old, prior_dists, n_param_infer)
    log_jacobian_old = calc_log_jac(x_old, positive_proposal, n_param_infer)
    log_lik_old = kalman_filter(y_data, sigma, log_c, delta_t)
    log_lik_val[1] = log_lik_old

    @showprogress 1 "Running sampler..." for i in 2:n_samples
        
        # Propose new-parameters (when old values are used)
        propose_parameters(x_prop, x_old, mcmc_sampler, n_param_infer, positive_proposal)
        
        log_c .= x_prop[1:3]
        sigma = x_prop[4]
        # Calculate new jacobian, log-likelihood and prior prob >
        log_prior_new = calc_log_prior(x_prop, prior_dists, n_param_infer)
        log_jacobian_new = calc_log_jac(x_prop, positive_proposal, n_param_infer)
        log_lik_new = kalman_filter(y_data, sigma, log_c, delta_t)
        # Acceptange probability
        log_u = log(rand())
        log_alpha = (log_lik_new - log_lik_old) + (log_prior_new - log_prior_old) + 
            (log_jacobian_old - log_jacobian_new)

        # In case of very bad-parameters (NaN) do not accept 
        if isnan(log_alpha)
            log_alpha = -Inf 
        end

        # Accept 
        if log_u < log_alpha
            log_lik_old = log_lik_new
            log_prior_old = log_prior_new
            log_jacobian_old = log_jacobian_new
            param_sampled[:, i] .= x_prop
            x_old .= x_prop
        # Do not accept 
        else
            param_sampled[:, i] .= x_old
        end

        log_lik_val[i] = log_lik_old

        # Update-adaptive mcmc-sampler 
        update_sampler!(mcmc_sampler, param_sampled, i, log_alpha)
    end

    return log_lik_val, param_sampled
end


function run_kalman_ornstein(n_samples, data_set; seed = 123)

    prior_ind_param = [Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(0.0, 1.0)]
    prior_error_param = [Gamma(1.0, 0.4)]

    param_info = init_param(prior_ind_param, 
                            prior_error_param, 
                            ind_param_pos=false, 
                            ind_param_log=true,
                            error_param_pos=true, 
                            init_ind_param=[-0.7, 2.3, -0.9], 
                            init_error_param=[0.3])

    cov_mat = diagm([0.1, 0.1, 0.1, 0.1])
    mcmc_sampler = init_mcmc(RandomWalk(), param_info, step_length=1.0, cov_mat=cov_mat)

    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Ornstein/Ornstein" * data_set * ".csv"
    file_loc = init_file_loc(path_data, "Ornstein_test_proposal")
    data_obs = CSV.read(file_loc.path_data, DataFrame)
    y_data = data_obs["obs"]
    t_vec = data_obs["time"]
    delta_t = t_vec[2] - t_vec[1]

    n_samples_use = 60000
    Random.seed!(seed)
    log_lik, param_sampled = run_mcmc(n_samples_use, mcmc_sampler, param_info, y_data, delta_t)
    samples = param_sampled[:, 20000:end]
    
    cov_mat = cov(param_sampled') .* 2.38^2 / 3
    mcmc_sampler = init_mcmc(RandomWalk(), param_info, step_length=1.0, cov_mat=cov_mat)
    Random.seed!(seed + 1)
    log_lik, param_sampled = run_mcmc(n_samples, mcmc_sampler, param_info, y_data, delta_t)

    i_take = 10000:20:970000    
    data_post = param_sampled[:, i_take]'
    data_post = convert(DataFrame, data_post)
    rename!(data_post, ["c1", "c2", "c3", "sigma1"])

    dir_save = pwd() * "/Intermediate/Single_individual/Ornstein_test_proposal/Kalman_" * data_set * "/"
    if !isdir(dir_save)
        mkdir(dir_save)
    end

    CSV.write(dir_save * "Kalman.csv", data_post)
end

n_samples = 970000
run_kalman_ornstein(n_samples, "1")
run_kalman_ornstein(n_samples, "12")
run_kalman_ornstein(n_samples, "123")