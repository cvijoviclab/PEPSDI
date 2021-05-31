"""
    log_pdf_no_const(dist::Gamma{<:AbstractFloat}, x)

Calculate log-pdf without proportionality constants. 

Currently only scalar input of x is supported. 
"""
function log_pdf_no_const(dist::Gamma{<:AbstractFloat}, x::FLOAT)
    alpha = shape(dist)
    theta = scale(dist)
    #(alpha - 1) * log(x) - x / theta 
    
    return logpdf(dist, x)
end
function log_pdf_no_const(dist::Normal{<:AbstractFloat}, x::FLOAT)
    mu = mean(dist)
    sigma = std(dist)
    #-0.5 * log(sigma) + 1.0/sigma * (x - mu)^2
    
    return logpdf(dist, x)
end
function log_pdf_no_const(dist::Uniform{<:AbstractFloat}, x::FLOAT)
    #mu = mean(dist)
    #sigma = std(dist)
    #-0.5 * log(sigma) + 1.0/sigma * (x - mu)^2
    
    return logpdf(dist, x)
end


"""
    calc_log_jac(param::Array{FLOAT, 1}, param_pos::Array{Bool, 1}, n_param)

Calculate log-jacobian for parameters proposed via exponential-transformation. 

Param-array must be a one-dimensional array of equal length to param_pos which 
contains information of which parameters are enforced as positive. 
"""
function calc_log_jac(param, param_pos::Array{Bool, 1}, n_param::T)::FLOAT where T<:Signed
    log_jac::FLOAT = 0.0
    @simd for i in 1:n_param
        if param_pos[i] == true
            log_jac -= log(deepcopy(param[i]))
        end
    end

    return log_jac 
end


"""
    calc_log_prior(x, prior_dist::T1, n_param) where T1<:Array{<:Distribution, 1}

Calculate log-prior for array x based on the priors-distributions array of length n_param. 
"""
function calc_log_prior(x, prior_dist::T1, n_param::T2)::FLOAT where {T1<:Array{<:Distribution, 1}, T2<:Signed}

    log_prior::FLOAT = 0.0
    @simd for i in 1:n_param
        @inbounds log_prior += log_pdf_no_const(prior_dist[i], x[i])
    end

    return log_prior
end


"""
    init_file_loc(path_data::String, model_name::String; 
        dir_save::String="", multiple_ind=false)

Initalise FileLocations-struct for a model. 

If the user does not provide a dir-save, the sampling results 
are stored in intermediate-directory (strongly recomended)

See also: [`FileLocations`](@ref)
"""
function init_file_loc(path_data::String, model_name::String; 
    dir_save::String="", multiple_ind=false, cov_name::T1=[""], 
    cov_val=Array{FLOAT, 1}(undef, 0), dist_id=ones(Int64, 0))::FileLocations where T1<:Array{<:String, 1}

    # Sanity ensure input 
    if model_name[1] == "/"
        model_name = model_name[2:end]
    end

    if dir_save == "" && multiple_ind == false
        dir_save = pwd() * "/Intermediate/Single_individual/" * model_name
    elseif dir_save == "" && multiple_ind == true 
        dir_save = pwd() * "/Intermediate/Multiple_individuals/" * model_name
    end

    if length(dist_id) != 0
        if length(dist_id) != length(cov_val)
            @printf("Error: Length of dist-id does not match the length of cov-val")
        end
    end
    if length(cov_val) == 0
        dist_id = ones(Int64, 1)
    end
    if length(cov_val) != 0 && length(dist_id) == 0
        dist_id = ones(Int64, length(cov_val))
    end


    file_loc = FileLocations(path_data, model_name, dir_save, cov_name, cov_val, dist_id)
    return file_loc
end


"""
    calc_dir_save!(file_loc::FileLocations, filter, mcmc_sampler)

For a specific filter and mcmc-sampler calculate correct sub-directory for saving result. 

Initalisation of file-locations creates the main-directory to save pilot-run data 
and inference result. Based on filter, mcmc-sampler this function creates the 
correct sub-directory for saving the result. User does not access this function. 

See also: [`FileLocations`, `init_file_loc`](@ref)
"""
function calc_dir_save!(file_loc::FileLocations, filter, mcmc_sampler; mult_ind=false)
    if filter.rho != 0
        tag_corr = "/Correlated"
    elseif filter.rho == 0
        tag_corr = "/Not_correlated"
    end

    if mcmc_sampler.name_sampler == "Gen_am_sampler"
        tag_samp = "/Gen_am_sampler"
    elseif mcmc_sampler.name_sampler == "Rand_walk_sampler"
        tag_samp = "/Rand_walk_sampler"
    elseif mcmc_sampler.name_sampler == "Am_sampler"
        tag_samp = "/Am_sampler"
    elseif mcmc_sampler.name_sampler == "Ram_sampler"
        tag_samp = "/Ram_sampler"
    end

    if mult_ind == true
        tag_mult = "/Multiple"
    else
        tag_mult = ""
    end

    # File-locations is mutable, this must be changed!
    file_loc.dir_save = file_loc.dir_save * tag_mult * tag_corr * tag_samp * "/"
end


"""
    calc_param_init_val(init_param, prior_list)::Array{FLOAT, 1}

Initial value for parameter-vector using mean, mode, median or random (init_param) on prior_list. 

Only works for case sensitive mean, mode, median or random for init_param. For cachy-distributions 
mean and mode are changed to median. 
"""
function calc_param_init_val(init_param::String, prior_list)::Array{FLOAT, 1}
    
    # If multivariate prior provided adapt length. Adapt for empty-prior-list
    len_prior = length(prior_list)
    if len_prior > 0
        if length(prior_list[1]) > 1
            len_prior = length(prior_list[1])
        end
    end

    if !(init_param == "mean" || init_param == "mode" || init_param == "random" || init_param == "median")
        @printf("Error: If init_param is a string it should be either ")
        @printf("mean, mode or random. Provided: %s\n", init_param)
        return 1 
    end

    # Change to median if Cauchy-distribution is provided
    for i in 1:length(prior_list)
        change_val = false
        if typeof(prior_list[i]) <: Truncated{<:Cauchy{<:AbstractFloat}, Continuous, <:AbstractFloat}
            change_val = true 
        elseif typeof(prior_list[i]) <: Cauchy{<:AbstractFloat}
            change_val = true 
        end

        if change_val == true && !(init_param == "median" || init_param == "random")
            @printf("As cauchy distribution is used will change to median for init-param\n")
            init_param = "median"
        end
    end

    # Calculate the number of parameters (account for multivariate distribution)
    if length(prior_list) == 1
        n_prior_dists = 1
        n_param = length(prior_list[1])
    else
        n_prior_dists = length(prior_list)
        n_param = n_prior_dists
    end

    param_init::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_param)

    # In the case either mean, mode or random is provided 
    if init_param == "mean"
        param_init_tmp = mean.(prior_list)
    elseif init_param == "mode"
        param_init_tmp = mode.(prior_list)
    elseif init_param == "median"
        param_init_tmp = median.(prior_list)
    elseif init_param == "random"
        param_init_tmp = rand.(prior_list)
    end

    # Handle the case if multivariate distribution is sent in 
    if typeof(param_init_tmp) <: Array{<:Array{<:AbstractFloat, 1}, 1}
        param_init .= param_init_tmp[1]
    else
        param_init .= param_init_tmp
    end
    
    return param_init
end
"""
    calc_param_init_val(init_param::T, prior_list)::Array{FLOAT, 1} where T<:Array{<:AbstractFloat, 1}

Initial value for parameter-vector using provided array of equal length to prior-list. 
"""
function calc_param_init_val(init_param::T, prior_list)::Array{FLOAT, 1} where T<:Array{<:AbstractFloat, 1}

    # If multivariate prior provided adapt length. Adapt for empty-prior-list
    len_prior = length(prior_list)
    if len_prior > 0
        if length(prior_list[1]) > 1
            len_prior = length(prior_list[1])
        end
    end

    if length(init_param) != len_prior
        @printf("If providing start-guesses, the number of start-guesses ")
        @printf("must match the number of priors\n")
        return 1 
    end

    param_init::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, len_prior)

    for i in 1:len_prior
        param_init[i] = convert(FLOAT, init_param[i])
    end

    return param_init
end


"""
    calc_bool_array(bool_val::Array{Bool, 1}, n_param::T1)::Array{Bool, 1} where T1 <: Signed

Check that bool-array has length n_param and return bool_val upon success. 
"""
function calc_bool_array(bool_val::Array{Bool, 1}, n_param::T1)::Array{Bool, 1} where T1 <: Signed
    
    # Sanity check input 
    if length(bool_val) != n_param
        @printf("Error: When creating bool-array from array it must ")
        @printf("match the number of parameters\n")
        return 1 
    end
   
    return bool_val

end
"""
    calc_bool_array(bool_val::Bool, n_param::T1)::Array{Bool, 1} where T1 <: Signed

Return bool-array of length n_param consting of bool-val. 
"""
function calc_bool_array(bool_val::Bool, n_param::T1)::Array{Bool, 1} where T1<:Signed

    bool_array::Array{Bool, 1} = Array{Bool, 1}(undef, n_param)
    bool_array .= bool_val

    return bool_array
end


"""
    empty_dist()::Array{Normal{Float64}, 1}

Intialise empty array of <:Array{Distribution, 1}. 
"""
function empty_dist()::Array{Normal{Float64}, 1}
    dist_use::Array{Normal{Float64}, 1} = deleteat!([Normal()], 1)
    return dist_use
end

