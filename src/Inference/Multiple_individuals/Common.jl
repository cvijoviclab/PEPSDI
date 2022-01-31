#=
    Common functions utlised by both PEPSDI running options 1 and 2. 
=#


"""
    init_pop_param_curr(mean_vec::T1, 
                        scale_vec::T1,
                        corr_mat::T2,
                        penalty=false) where 
                        {T1 <: Array{<:AbstractFloat, 1},
                        T2 <: Array{<:AbstractFloat, 2}}

Intialise PopParam-struct for storing current η-parameters. 

Create structs of current parameters from provided mean-vector, scale-vector 
and correlation matrix. 
"""
function init_pop_param_curr(mean_vec::T1, 
                             scale_vec::T1,
                             corr_mat::T2,
                             penalty=false) where 
                             {T1 <: Array{<:AbstractFloat, 1},
                             T2 <: Array{<:AbstractFloat, 2}}

    mean_vec::Array{FLOAT, 1} = convert(Array{FLOAT, 1}, mean_vec)
    scale_vec::Array{FLOAT, 1} = convert(Array{FLOAT, 1}, scale_vec)
    corr_mat::Array{FLOAT, 2} = convert(Array{FLOAT, 2}, corr_mat)

    # Add support for penalty
    if penalty == false
        penalty_vec = Array{FLOAT, 1}(undef, 0)
    end

    pop_par_current::PopParam = PopParam(mean_vec, scale_vec, penalty_vec, corr_mat)

    return pop_par_current
end
"""
    init_pop_param_curr(pop_param_info::ParamInfoPop)

Creates struct using initial values in pop_param_info

See also: [`PopParamInfo`](@ref)
"""
function init_pop_param_curr(pop_param_info::ParamInfoPop)

    mean_vec::Array{FLOAT, 1} = pop_param_info.init_pop_param_mean
    scale_vec::Array{FLOAT, 1} = pop_param_info.init_pop_param_scale
    corr_mat::Array{FLOAT, 2} = pop_param_info.init_pop_param_corr

    penalty_vec = Array{FLOAT, 1}(undef, 0)

    pop_par_current::PopParam = PopParam(mean_vec, scale_vec, penalty_vec, corr_mat)

    return pop_par_current
end


"""
init_chains(pop_param_info::ParamInfoPop, 
            ind_param_info_arr, 
            n_individuals::T1, 
            n_samples::T1; 
            penalty=false) where T1<:Signed

Initialise struct for storing mcmc-chains in Gibbs-sampler. 

Initial values are provided by pop_param_info and ind_param_info_arr. Overall, 
likelihood for kappa-sigma and individual parameters is stored, togehter with 
chains for individual-parameters, η-parameters, error-parameters and κ-parameters. 

See also: [`init_ind_param_info`, init_pop_param_info](@ref)
"""
function init_chains(pop_param_info::ParamInfoPop, 
                     ind_param_info_arr, 
                     n_individuals::T1, 
                     n_samples::T1; 
                     penalty=false) where T1<:Signed

    # Dimensions of chains 
    dim_mean = length(pop_param_info.init_pop_param_mean)
    dim_kappa_sigma = length(pop_param_info.init_pop_param_sigma) + length(pop_param_info.init_pop_param_kappa)
    dim_corr = pop_param_info.prior_pop_param_corr.d

    # Arrays mcmc-chains are stored 
    pop_mean = Array{FLOAT, 2}(undef, (dim_mean, n_samples))
    pop_scale = Array{FLOAT, 2}(undef, (dim_mean, n_samples))
    pop_corr = Array{FLOAT, 3}(undef, (dim_corr, dim_corr, n_samples))
    pop_kappa_sigma = Array{FLOAT, 2}(undef, (dim_kappa_sigma, n_samples))
    ind_param = Array{Array{FLOAT, 2}, 1}(undef, n_individuals)
    for i in 1:n_individuals
        ind_param[i] = Array{FLOAT, 2}(undef, (ind_param_info_arr[i].n_param, n_samples))
    end

    if penalty == false
        penalty = Array{FLOAT, 2}(undef, (0, 0))
    end

    # Arrays for log-likelihood
    log_lik_ind = Array{FLOAT, 2}(undef, (n_samples, n_individuals))
    log_lik_kappa_sigma = Array{FLOAT, 1}(undef, (n_samples))

    # Initial values for the chains 
    pop_mean[:, 1] .= pop_param_info.init_pop_param_mean
    pop_scale[:, 1] .= pop_param_info.init_pop_param_scale
    pop_corr[:, :, 1] .= pop_param_info.init_pop_param_corr
    # Kappa, sigma and individual parameters 
    pop_kappa_sigma[:, 1] .= vcat(pop_param_info.init_pop_param_kappa, pop_param_info.init_pop_param_sigma)
    for i in 1:n_individuals
        ind_param[i][:, 1] .= ind_param_info_arr[i].init_ind_param
    end

    # Chains struct 
    chains = ChainsMixed(pop_mean, pop_scale, pop_corr, pop_kappa_sigma, 
        ind_param, penalty, log_lik_ind, log_lik_kappa_sigma)

    return chains 
end


"""
    write_result_file(mcmc_chains::ChainsMixed, 
                      n_samples::T, 
                      file_loc::FileLocations, 
                      filter_opt, 
                      pop_param_info::ParamInfoPop, 
                      exp_id) where T<:Signed

Write chains (η, σ, κ and individual-parameters) to subdirectory to dir_save in file_loc struct.

All parameters are saved to separate files in same directory. Correlation level, 
and number of particles in filter_opt are used to name the directory.
"""
function write_result_file(mcmc_chains::ChainsMixed, 
                           n_samples::T, 
                           file_loc::FileLocations, 
                           filter_opt, 
                           pop_param_info::ParamInfoPop, 
                           exp_id, 
                           run_time_sampler) where T<:Signed

    n_ind_param = size(mcmc_chains.ind_param[1])[1]
    n_mean_param = length(pop_param_info.prior_pop_param_mean)
    n_kappa_sigma = size(mcmc_chains.kappa_sigma)[1]
    n_kappa = length(pop_param_info.prior_pop_param_kappa)
    n_sigma = length(pop_param_info.prior_pop_param_sigma)

    # Ensure directory for storing results exists 
    iterator = 1
    dir_save::String = ""
    while true 
        dir_save = file_loc.dir_save * "/" * 
            "Npart" * string(filter_opt.n_particles) * 
            "_nsamp" * string(n_samples) * 
            "_corr" * string(filter_opt.rho) * 
            "_exp_id" * string(exp_id) * 
            "_run" * string(iterator) * "/"

        # Create directory if does not exist 
        if !isdir(dir_save)
            mkpath(dir_save)
            break
        end
        iterator += 1
    end
            
    file_ind = dir_save * "Ind_param.csv"
    file_kappa_sigma = dir_save * "Kappa_sigma.csv"
    file_mean = dir_save * "Mean.csv"
    file_scale = dir_save * "Scale.csv"
    file_corr = dir_save * "Corr.csv"
    file_log_lik = dir_save * "Log_lik.csv"    

    # Column names for the files 
    col_name_mean = Array{String, 1}(undef, n_mean_param)
    col_name_scale = Array{String, 1}(undef, n_mean_param)
    col_name_ind = Array{String, 1}(undef, n_ind_param+1)
    for i in 1:n_mean_param
        string_i = string(i)
        col_name_mean[i] = "mu" * string_i
        col_name_scale[i] = "scale" * string_i
    end
    for i in 1:n_ind_param
        col_name_ind[i] = "c" * string(i)
    end
    col_name_ind[end] = "id"
    col_name_kappa_sigma = Array{String, 1}(undef, n_kappa_sigma)
    for i in 1:n_kappa_sigma
        if i <= n_kappa
            col_name_kappa_sigma[i] = "kappa" * string(i)
        else
            col_name_kappa_sigma[i] = "sigma" * string(i-n_kappa)
        end
    end

    # Write simple-table values 
    mean_data = convert(DataFrame, mcmc_chains.mean')
    scale_data = convert(DataFrame, mcmc_chains.scale')
    kappa_sigma_data = convert(DataFrame, mcmc_chains.kappa_sigma')
    rename!(mean_data, col_name_mean)
    rename!(scale_data, col_name_scale)
    rename!(kappa_sigma_data, col_name_kappa_sigma)
    CSV.write(file_mean, mean_data)
    CSV.write(file_scale, scale_data)
    CSV.write(file_kappa_sigma, kappa_sigma_data)

    # Write individual parameters 
    n_ind = length(mcmc_chains.ind_param)
    data_ind = vcat(mcmc_chains.ind_param[1], repeat([1], size(mcmc_chains.ind_param[1])[2])')'
    for i in 2:n_ind
        data_ind_new = vcat(mcmc_chains.ind_param[i], repeat([i], size(mcmc_chains.ind_param[i])[2])')'
        data_ind = vcat(data_ind, data_ind_new)
    end
    data_ind = convert(DataFrame, data_ind)
    println(col_name_ind)
    rename!(data_ind, col_name_ind)
    CSV.write(file_ind, data_ind)

    # Write correlation matrix 
    data_corr = hcat(mcmc_chains.corr[:, :, 1], repeat([1.0], n_ind_param))
    for i in 2:n_samples
        data_corr_new = hcat(mcmc_chains.corr[:, :, i], repeat([i], n_ind_param))
        data_corr = vcat(data_corr, data_corr_new)
    end
    data_corr = convert(DataFrame, data_corr)
    rename!(data_corr, col_name_ind)
    CSV.write(file_corr, data_corr)

    # Write log-likelihood
    n_individuals = size(mcmc_chains.log_lik_ind)[2]
    data_log_lik = Array{FLOAT, 2}(undef, (0, 2))
    for i in 1:n_individuals
        data_tmp = Array{FLOAT, 2}(undef, (n_samples-1, 2))
        data_tmp[:, 1] .= mcmc_chains.log_lik_ind[2:end, i]
        data_tmp[:, 2] .= i
        data_log_lik = vcat(data_log_lik, data_tmp)
    end
    log_lik_data = convert(DataFrame, data_log_lik)
    rename!(log_lik_data, ["log_lik", "id"])
    CSV.write(file_log_lik, log_lik_data)

    # Save the run-time 
    data_run_time = zeros(1, 1)
    data_run_time[1, 1] = run_time_sampler.value
    data_run_time = convert(DataFrame, data_run_time)
    rename!(data_run_time, ["Run_time"])
    CSV.write(dir_save * "Run_time.csv", data_run_time)

    # Change file_loc dir-save
    file_loc.dir_save = dir_save
end


"""
    map_sigma_to_mod_param_mixed!(mod_param_arr_j::ModelParameters, 
                                  sigma_log::Array{Bool, 1},
                                  sigma_current)

Map current sigma-value to entry j in model_param_arr in Gibbs-sampler. 

Sigma_log tracks parameters estimated on log-scale. 
"""
function map_sigma_to_mod_param_mixed!(mod_param_arr_j::ModelParameters, 
                                       sigma_log::Array{Bool, 1},
                                       sigma_current)

    mod_param_arr_j.error_parameters .= sigma_current
    @views mod_param_arr_j.error_parameters[sigma_log] .= exp.(sigma_current[sigma_log])
    
end


"""
    map_kappa_to_mod_param_mixed!(mod_param_arr_j::ModelParameters, 
                                  kappa_log::Array{Bool, 1},
                                  kappa_current)

Map current kappa-value to entry j in model_param_arr in Gibbs-sampler. 

kappa_log tracks parameters estimated on log-scale. 
"""
function map_kappa_to_mod_param_mixed!(mod_param_arr_j::ModelParameters, 
                                       kappa_log::Array{Bool, 1},
                                       kappa_current)

    mod_param_arr_j.individual_parameters.kappa .= kappa_current
    @views mod_param_arr_j.individual_parameters.kappa[kappa_log] .= exp.(kappa_current[kappa_log])

end


"""
    update_ind_param_ind!(ind_par::DynModInput, x_prop, log_scale::Array{Bool, 1})

Map current individual parameter value (x_prop) to indivudal_parameters in entry j in mod_param_arr. 

log_scale tracks parameters estimated on log-scale. 
"""
function update_ind_param_ind!(ind_par::DynModInput, x_prop, log_scale::Array{Bool, 1})
    ind_par.c .= x_prop

    # Ensure correct transformation of parameters 
    @inbounds @simd for k in 1:length(log_scale)
        if log_scale[k]
            ind_par.c[k] = exp(ind_par.c[k])
        end
    end
end