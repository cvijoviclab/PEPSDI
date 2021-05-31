#=
    Function creaing arrays of smaler data-structurs which are required 
    by Gibbs-sampler. 
=# 


"""
init_param_info_arr(ind_param_info::ParamInfoIndPre, 
                    dist_ind_param, 
                    n_individuals::T)::Array{ParamInfoInd, 1} where T<:Signed

Initialise array storing individual parameter information per individual from user provided ind_param_info. 

In contrast to ParamInfoIndPre ParamInfoInd is strictly typed. This array makes it 
possible in future development to have different parameters (and models) between 
individuals. Currently, only allows different initial values. 

See also: [`init_ind_param_info`](@ref)
"""
function init_param_info_arr(ind_param_info::ParamInfoIndPre, 
                             dist_ind_param, 
                             n_individuals::T)::Array{ParamInfoInd, 1} where T<:Signed

    ind_param_info_arr::Array{ParamInfoInd, 1} = Array{ParamInfoInd, 1}(undef, n_individuals)
    n_param = length(dist_ind_param)

    # Create param-info struct and mcmc-sampler for each individual 
    for i in 1:n_individuals
        pos_ind_param::Array{Bool, 1} = calc_bool_array(ind_param_info.pos_ind_param, n_param)
        log_ind_param::Array{Bool, 1} = calc_bool_array(ind_param_info.log_ind_param, n_param)
        local init_val::Array{FLOAT, 1} 
        if typeof(ind_param_info.init_ind_param) <: Array{<:AbstractFloat, 2}
            init_val = calc_param_init_val(ind_param_info.init_ind_param[i, :], [dist_ind_param])
        else
            init_val = calc_param_init_val(ind_param_info.init_ind_param, [dist_ind_param])
        end

        ind_param_info_arr[i] = deepcopy(ParamInfoInd(init_val, pos_ind_param, log_ind_param, 
            length(init_val)))
    end

    return ind_param_info_arr
end
function init_param_info_arr(ind_param_info::ParamInfoIndPre, 
                             dist_ind_param_arr::T1, 
                             n_individuals::T)::Array{ParamInfoInd, 1} where {T<:Signed, T1<:Array{<:Any, 1}}

    ind_param_info_arr::Array{ParamInfoInd, 1} = Array{ParamInfoInd, 1}(undef, n_individuals)
    n_param = length(dist_ind_param_arr[1])

    # Create param-info struct and mcmc-sampler for each individual 
    for i in 1:n_individuals
        pos_ind_param::Array{Bool, 1} = calc_bool_array(ind_param_info.pos_ind_param, n_param)
        log_ind_param::Array{Bool, 1} = calc_bool_array(ind_param_info.log_ind_param, n_param)
        local init_val::Array{FLOAT, 1} 
        if typeof(ind_param_info.init_ind_param) <: Array{<:AbstractFloat, 2}
            init_val = calc_param_init_val(ind_param_info.init_ind_param[i, :], [dist_ind_param_arr[i]])
        else
            init_val = calc_param_init_val(ind_param_info.init_ind_param, [dist_ind_param_arr[i]])
        end

        ind_param_info_arr[i] = deepcopy(ParamInfoInd(init_val, pos_ind_param, log_ind_param, 
            length(init_val)))
    end

    return ind_param_info_arr
end


"""
    init_model_parameters_arr(ind_param_info_arr::Array{ParamInfoInd, 1}, 
                              pop_param_info::ParamInfoPop, 
                              model, 
                              n_individuals::T)::Array{ModelParameters, 1} where T<:Signed

Initialise array of model-parameters (see [`init_ind_data`](@ref)) for individual parameters.
"""
function init_model_parameters_arr(ind_param_info_arr::Array{ParamInfoInd, 1}, 
                                   pop_param_info::ParamInfoPop, 
                                   model, 
                                   n_individuals::T, 
                                   ind_data_arr::T0)::Array{ModelParameters, 1} where {T0<:Array{IndData, 1}, T<:Signed}

    # Add unique per individual parameters 
    mod_param_arr::Array{ModelParameters, 1} = Array{ModelParameters, 1}(undef, n_individuals)
    for i in 1:n_individuals
        # Deepcopy to avoid problems when running in parallel 
        error_param::Array{FLOAT, 1} = deepcopy(pop_param_info.init_pop_param_sigma)
        kappa::Array{FLOAT, 1} = deepcopy(pop_param_info.init_pop_param_kappa)
        ind_param::Array{FLOAT, 1} = ind_param_info_arr[i].init_ind_param
        mod_param_arr[i] = deepcopy(init_model_parameters(ind_param, error_param, model, kappa=kappa, 
            covariates=ind_data_arr[i].cov_val))
    end

    return mod_param_arr
end


"""
    init_filter_opt_arr(filter_opt, n_individuals::T) where T<:Signed 

Initialise array of filters (see [`init_filter`](@ref)) for individuals.

Each individual has a filter to allow individual based number of particles. Note, as filter_opt
is provided by the user each individual uses the same algorithm (e.g Bootstrap EM).

See also: [`init_filter`](@ref)
"""
function init_filter_opt_arr(filter_opt, n_individuals::T) where T<:Signed
    filter_opt_arr::Array{typeof(filter_opt), 1} = Array{typeof(filter_opt), 1}(undef, n_individuals)
    for i in 1:n_individuals
        filter_opt_arr[i] = deepcopy(filter_opt)
    end

    return filter_opt_arr
end


"""
    init_ind_data_arr(file_loc::FileLocations, filter_opt)::Array{IndData, 1}

Initialise array of observed data (see [`init_ind_data`](@ref)). Each element corresponds to one individual. 

See also: [`init_ind_data`](@ref)
"""
function init_ind_data_arr(file_loc::FileLocations, filter_opt)::Array{IndData, 1}

    # Read data and extract all individuals 
    data_obs::DataFrame = CSV.read(file_loc.path_data, DataFrame, types=Dict(:id=>Int64))
    
    id_list_pre::Array{Int64, 1} = data_obs[!, :id][:]
    id_list::Array{Int64, 1} = unique(id_list_pre)

    # Collect the data into an array 
    ind_data_arr::Array{IndData, 1} = Array{IndData, 1}(undef, length(id_list))
    j::Int16 = 1
    for i in id_list
        data_ind_j::DataFrame = filter(row -> row[:id] == i, data_obs)
        ind_data_arr[j] = deepcopy(init_ind_data(data_ind_j, filter_opt, cov_name=file_loc.cov_name))
        j += 1
    end

    return ind_data_arr
end


"""
    init_mcmc_arr(mcmc_sampler_ind, ind_param_info_arr::Array{ParamInfoInd, 1}, n_individuals::T1) where T1<:Signed

Initialise array of mcmc-samplers (see [`init_mcmc`](@ref)) for individual parameters.

Each individual has a sampler to allow individual based sampler-tuning. Note, as mcmc_sampler_ind
is provided by the user each individual uses the same algorithm (e.g GenAmSampler).

See also: [`init_mcmc`](@ref)
"""
function init_mcmc_arr(mcmc_sampler_ind, ind_param_info_arr::Array{ParamInfoInd, 1}, n_individuals::T1) where T1<:Signed

    sampler_use = calc_sampler_type(mcmc_sampler_ind)
    
    mcmc_sampler_ind_arr::Array{typeof(mcmc_sampler_ind), 1} = Array{typeof(mcmc_sampler_ind), 1}(undef, n_individuals)
    for i in 1:n_individuals
        # To preserve user input regarding n-steps before update 
        if typeof(mcmc_sampler_ind) <: McmcSamplerRandWalk
            mcmc_sampler_ind_arr[i] = deepcopy(init_mcmc(sampler_use, ind_param_info_arr[i]))
        else
            mcmc_sampler_ind_arr[i] = deepcopy(init_mcmc(sampler_use, ind_param_info_arr[i],
                step_before_update=mcmc_sampler_ind.steps_before_update))
        end

        # Ensure that user provided cov-mat is transfered 
        mcmc_sampler_ind_arr[i].cov_mat .= mcmc_sampler_ind.cov_mat
    end

    return mcmc_sampler_ind_arr
end
function init_mcmc_arr(mcmc_sampler_pop, pop_param_info::ParamInfoPop, n_individuals::T1) where T1<:Signed

    sampler_use = calc_sampler_type(mcmc_sampler_pop)
    
    println("n_individuals = $n_individuals")
    mcmc_sampler_pop_arr::Array{typeof(mcmc_sampler_pop), 1} = Array{typeof(mcmc_sampler_pop), 1}(undef, n_individuals)
    for i in 1:n_individuals
        # To preserve user input regarding n-steps before update 
        if typeof(mcmc_sampler_pop) <: McmcSamplerRandWalk
            mcmc_sampler_pop_arr[i] = deepcopy(init_mcmc(sampler_use, pop_param_info))
        else
            mcmc_sampler_pop_arr[i] = deepcopy(init_mcmc(sampler_use, pop_param_info,
                step_before_update = mcmc_sampler_pop.steps_before_update))
        end

        # Ensure that user provided cov-mat is transfered 
        mcmc_sampler_pop_arr[i].cov_mat .= mcmc_sampler_pop.cov_mat
    end

    return mcmc_sampler_pop_arr
end


"""
    init_rand_num_arr(filter_opt, n_individuals::T) where T<:Signed 

init_rand_num_arr(ind_data_arr::T0, model_arr::T1, filter_opt_arr, 
    n_individuals::T2) where {T0<:Array{IndData, 1}, T1<:Array{<:SdeModel, 1}, T2<:Signed}

Each individual has a uniqe random-number-struct allowing for varying number of observations 
etc. Note, random-numbers are updated with the same correlation level. 

See also: [`create_rand_num_sde`](@ref)
"""
function init_rand_num_arr(ind_data_arr::T0, model_arr::T1, filter_opt_arr, 
    n_individuals::T2) where {T0<:Array{IndData, 1}, T1<:Array{<:SdeModel, 1}, T2<:Signed}

    rand_num_arr::Array{RandomNumbers, 1} = Array{RandomNumbers, 1}(undef, n_individuals)
    for i in 1:n_individuals
        rand_num_arr[i] = create_rand_num(ind_data_arr[i], model_arr[i], filter_opt_arr[i])
    end

    return rand_num_arr
end
function init_rand_num_arr(ind_data_arr::T0, model_arr::T1, filter_opt_arr, 
    n_individuals::T2) where {T0<:Array{IndData, 1}, T1<:Array{<:PoisonModel, 1}, T2<:Signed}

    rand_num_arr::Array{RandomNumbers, 1} = Array{RandomNumbers, 1}(undef, n_individuals)
    for i in 1:n_individuals
        rand_num_arr[i] = create_rand_num(ind_data_arr[i], model_arr[i], filter_opt_arr[i])
    end

    return rand_num_arr
end
function init_rand_num_arr(ind_data_arr::T0, model_arr::T1, filter_opt_arr, 
    n_individuals::T2) where {T0<:Array{IndData, 1}, T1<:Array{<:SsaModel, 1}, T2<:Signed}

    rand_num_arr::Array{RandomNumbersSsa, 1} = Array{RandomNumbersSsa, 1}(undef, n_individuals)
    for i in 1:n_individuals
        rand_num_arr[i] = create_rand_num(ind_data_arr[i], model_arr[i], filter_opt_arr[i])
    end

    return rand_num_arr
end
function init_rand_num_arr(ind_data_arr::T0, model_arr::T1, filter_opt_arr, 
    n_individuals::T2) where {T0<:Array{IndData, 1}, T1<:Array{<:ExtrandModel, 1}, T2<:Signed}

    rand_num_arr::Array{RandomNumbersSsa, 1} = Array{RandomNumbersSsa, 1}(undef, n_individuals)
    for i in 1:n_individuals
        rand_num_arr[i] = create_rand_num(ind_data_arr[i], model_arr[i], filter_opt_arr[i])
    end

    return rand_num_arr
end