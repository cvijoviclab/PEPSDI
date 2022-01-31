"""
    init_pop_param_info(prior_pop_mean::Array{<:T, 1}, 
                        prior_pop_scale::Array{<:T, 1}, 
                        prior_pop_sigma::Array{<:T, 1};
                        prior_pop_kappa::Array{<:T, 1}=empty_dist(), 
                        prior_pop_corr=I, 
                        init_pop_mean="mean", 
                        init_pop_scale="mean", 
                        init_pop_kappa="mean", 
                        init_pop_sigma="mean",
                        pos_pop_kappa=true,
                        pos_pop_sigma=true, 
                        log_pop_kappa=false, 
                        log_pop_sigma=false, 
                        precision_scale::Bool=false)::ParamInfoPop where T<:Distribution

Initalise ParamInfoPop-struct (parameter information for population parameters). 

Priors, initial-values for population parameters η (correlation, scale and mean). 
For error-parameters and к transformation information (estimated on log-scale and 
if positive) can also be specified. This because error-parameters and к are updated 
via random-walk proposal, in contrast to η which is updated via NUTS-sampler. 

# Args
- `prior_pop_mean`: array of (possibly) different prior-distribution for mean in η
- `prior_pop_scale`: see prior_pop_mean
- `prior_pop_sigma`: see prior_pop_mean
- `prior_pop_kappa`: see prior_pop_mean
- `prior_pop_corr`: matrix-correlation distribution (e.g LKJ), if not specified no correlation is assumed. 
- `init_pop_mean`: initial value for mean in η. If median, mode, random or mean is initalised via prior_pop_mean. 
    If array of values of length prior_pop_mean is initalised as provided array (see [`calc_param_init_val`](@ref))
- `init_pop_scale`: see init_pop_mean
- `init_pop_kappa`: see init_pop_mean
- `init_pop_sigma`: see init_pop_mean
- `pos_pop_kappa`: bool, or bool-array, whether or not kappa should be positive and thus proposed via 
    exponential transformation (see [`calc_bool_array`](@ref))
- `pos_pop_sigma`: see pos_pop_kappa
- `log_pop_kappa`: bool, or bool-array, whether or not kappa is estimated on the log-scale. 
- `log_pop_sigma`: bool, or bool-array, whether or not kappa is estimated on the log-scale. 
- `pos_pop_sigma`: see pos_pop_kappa
- `precision_scale`: whether or not provided scale should be treated as precision vector. 

See also: [`init_pop_sampler_opt`, `calc_param_init_val`, `calc_bool_array`](@ref)
"""
function init_pop_param_info(prior_pop_mean::Array{<:T, 1}, 
                             prior_pop_scale::Array{<:T, 1}, 
                             prior_pop_sigma::Array{<:T, 1};
                             prior_pop_kappa=empty_dist(), 
                             prior_pop_corr=I, 
                             init_pop_mean="mean", 
                             init_pop_scale="mean", 
                             init_pop_kappa="mean", 
                             init_pop_sigma="mean",
                             pos_pop_kappa=true,
                             pos_pop_sigma=true, 
                             log_pop_kappa=false, 
                             log_pop_sigma=false, 
                             precision_scale::Bool=false)::ParamInfoPop where T<:Distribution

    n_mu::Int64 = length(prior_pop_mean)
    n_sigma::Int64 = length(prior_pop_sigma) 
    n_kappa::Int64 = length(prior_pop_kappa)

    # Sanity check input 
    if n_mu != length(prior_pop_scale)
        @printf("Error: As many scale-parameters must be provided as ")
        @printf("mu-parameters for the population parameters\n")
        return 1 
    end

    prior_pop_corr_use::LKJ{Float64, Int64} = LKJ(1, 1.0)
    if typeof(prior_pop_corr) <: LKJ{<:Any, <:Any}
        prior_pop_corr_use = prior_pop_corr
        if prior_pop_corr.d != n_mu
            @printf("Error: Dimension of prior covariance does not match")
            @printf(" the number of mu-parameters\n")
        end
    elseif prior_pop_corr == I
        # The correlation matrix will not be sampled 
        prior_pop_corr_use = LKJ(n_mu, 1.0)
    else
        @printf("Error: Prior-corr must be either a LKJ distribution or")
        @printf(" I (identity matrix -> no correlation in the model). \n")
    end

    # Calculate intial values 
    init_val_mean::Array{FLOAT, 1} = calc_param_init_val(init_pop_mean, prior_pop_mean)
    init_val_scale::Array{FLOAT, 1} = calc_param_init_val(init_pop_scale, prior_pop_scale)
    init_val_sigma::Array{FLOAT, 1} = calc_param_init_val(init_pop_sigma, prior_pop_sigma)
    init_val_kappa::Array{FLOAT, 1} = calc_param_init_val(init_pop_kappa, prior_pop_kappa)
    init_val_corr::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (prior_pop_corr_use.d, prior_pop_corr_use.d))
    if typeof(prior_pop_corr) <: LKJ{<:Any, <:Any}
        init_val_corr .= convert(Array{FLOAT, 2}, rand(prior_pop_corr, 1)[1])
    else
        init_val_corr .= Array{FLOAT, 2}(I, (n_mu, n_mu))
    end

    # Positivity and log-scale for sigma and kappa 
    pos_val_kappa::Array{Bool, 1} = calc_bool_array(pos_pop_kappa, n_kappa)
    pos_val_sigma::Array{Bool, 1} = calc_bool_array(pos_pop_sigma, n_sigma)
    log_val_kappa::Array{Bool, 1} = calc_bool_array(log_pop_kappa, n_kappa)
    log_val_sigma::Array{Bool, 1} = calc_bool_array(log_pop_sigma, n_sigma)

    param_info_pop = ParamInfoPop(prior_pop_mean, 
                                  prior_pop_scale, 
                                  prior_pop_corr_use,
                                  prior_pop_sigma, 
                                  prior_pop_kappa, 
                                  init_val_mean, 
                                  init_val_scale, 
                                  init_val_corr, 
                                  init_val_sigma, 
                                  init_val_kappa,  
                                  pos_val_sigma, 
                                  pos_val_kappa, 
                                  log_val_sigma, 
                                  log_val_kappa, 
                                  n_mu, 
                                  n_kappa, 
                                  n_sigma, 
                                  precision_scale)
                                  
    return param_info_pop
end


"""
    init_ind_param_info(init_ind, 
                        n_param::Int64;
                        dist::String="log_normal", 
                        log_scale::Bool=false, 
                        pos_param::Bool=true)::ParamInfoIndPre

Initalise ParamInfoIndPre-struct (parameter information for individual parameters). 

Set initial values as mean, mode, random or median from populaiton distribution.
Further, provide if parameters are estimated on log-scale and/or are enforced to be positive. 
"""
function init_ind_param_info(init_ind::String, 
                             n_param::Int64;
                             log_scale::Bool=false, 
                             pos_param::Bool=true)::ParamInfoIndPre

    # Sanity check input 
    if !(init_ind == "mean" || init_ind == "mode" || init_ind == "random" || init_ind == "median")
        @printf("Error: Individual parameters can only be initalised as 
            random, mean, mode or an array with values \n")
    end

    # Return output 
    init_ind_use::String = init_ind
    param_info_ind = ParamInfoIndPre(init_ind_use, pos_param, log_scale, n_param)

    return param_info_ind
end
"""
    init_ind_param_info(init_ind::T, 
                        n_param::Int64;
                        log_scale::Bool=false, 
                        pos_param::Bool=true)::ParamInfoIndPre where T<:Array{<:AbstractFloat, 1}

Provide initial individual parameters as an array. 
"""
function init_ind_param_info(init_ind::T, 
                             n_param::Int64;
                             log_scale::Bool=false, 
                             pos_param::Bool=true)::ParamInfoIndPre where T<:Array{<:AbstractFloat, 1}

    # Sanity check input 
    if length(init_ind) != n_param
        @printf("Error: If initial individual values for the parameters are provided the length must match")
        @printf("the length of the number of parameters.")
    end

    if log_scale == true && pos_param == true
        @printf("Warning: Parameters are enforced to be positive and estimated on log-scale\n")
    end

    # Return output 
    init_ind_use::Array{FLOAT, 1} = convert(Array{FLOAT, 1}, init_ind)
    param_info_ind = ParamInfoIndPre(init_ind_use, pos_param, log_scale, n_param)

    return param_info_ind
end
"""
    init_ind_param_info(init_ind::T, 
                        n_param::Int64;
                        log_scale::Bool=false, 
                        pos_param::Bool=true)::ParamInfoIndPre where T<:Array{<:AbstractFloat, 2}

Provide initial individual parameters as a matrix 
"""
function init_ind_param_info(init_ind::T, 
                             n_param::Int64;
                             log_scale::Bool=false, 
                             pos_param::Bool=true)::ParamInfoIndPre where T<:Array{<:AbstractFloat, 2}

    # Sanity check input 
    if size(init_ind)[2] != n_param
        @printf("Error: If initial individual values for the parameters are provided the length must match")
        @printf("the length of the number of parameters.")
    end

    if log_scale == true && pos_param == true
        @printf("Warning: Parameters are enforced to be positive and estimated on log-scale\n")
    end

    # Return output 
    init_ind_use::Array{FLOAT, 2} = convert(Array{FLOAT, 2}, init_ind)
    param_info_ind = ParamInfoIndPre(init_ind_use, pos_param, log_scale, n_param)

    return param_info_ind
end