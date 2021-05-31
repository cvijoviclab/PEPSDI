"""
    FileLocations

Directories of observed data, directory to save result, and name of model. 

Initalised by init_file_loc

See also: [`init_file_loc`](@ref)
"""
mutable struct FileLocations{T1<:Array{<:String, 1}, T2<:Array{<:AbstractFloat, 1}, T3<:Array{<:Integer, 1}}
    path_data::String
    model_name::String 
    dir_save::String 
    cov_name::T1
    cov_val::T2
    dist_id::T3
end


"""
    InitParameterInfo

Priors, initial-values, and parameter-characterics (log-scale and/or postive). 

For single-individual inference. 

Initialised by init_param. For both individual and error-parameters only 
a subset might be estimated on log-scale or be enforced as positive. The latter 
parameters are proposed via an exponential-transformation. 

See also: [`init_param`](@ref)
"""
struct InitParameterInfo{T1<:Array{<:Distribution, 1}, 
                         T2<:Array{<:Distribution, 1}, 
                         T3<:Array{<:AbstractFloat, 1}, 
                         T4<:Array{<:AbstractFloat, 1}, 
                         T5<:Signed}
                         
    prior_ind_param::T1
    prior_error_param::T2
    init_ind_param::T3
    init_error_param::T4
    n_param::T5
    ind_param_pos::Array{Bool, 1}
    error_param_pos::Array{Bool, 1}
    ind_param_log::Array{Bool, 1}
    error_param_log::Array{Bool, 1}
end


"""
    PopParam

Struct storing the current population parameters η = μ (mean), τ (scale) and (Ω) correlation. 

Penalty vec is potential future addition. 
"""
struct PopParam{T1<:Array{<:AbstractFloat, 1}, 
                T2<:Array{<:AbstractFloat, 1}, 
                T3<:Array{<:AbstractFloat, 1}, 
                T4<:Array{<:AbstractFloat, 2}}

    mean_vec::T1
    scale_vec::T2
    penalty_vec::T3
    corr_mat::T4
end


"""
    ChainsMixed

Struct storing the MCMC-chain when running the gibbs-samplers. 

Stores individual-parameters (c), population parameters (η = µ, τ, Ω), error-parameters (ξ) 
and cell-constant parameters. Further stores indiviudal likelihood. 

"""
struct ChainsMixed{T1<:Array{<:AbstractFloat, 2}, 
                   T2<:Array{<:AbstractFloat, 2}, 
                   T3<:Array{<:AbstractFloat, 3}, 
                   T4<:Array{<:AbstractFloat, 2}, 
                   T5<:Array{<:Array{<:AbstractFloat, 2}, 1}, 
                   T6<:Array{<:AbstractFloat, 2}, 
                   T7<:Array{<:AbstractFloat, 2}, 
                   T8<:Array{<:AbstractFloat, 1}}

    mean::T1 # µ
    scale::T2 # τ
    corr::T3 # Ω
    kappa_sigma::T4 # (ĸ, ξ)
    ind_param::T5
    penalty_par::T6
    log_lik_ind::T7
    log_lik_kappa_sigma::T8
end


"""
    KappaSigmaNormalSamplerOpt

Options (acceptance probability, warm-up samples, and variances) for (ĸ_pop, ξ_pop)-sampler used in PEPSDI opt 2. 

Intitialised by init_pop_sampler_opt. The variances = ε dictate how much the parameters 
vary between cells (ĸ, ξ) ~ N((ĸ_pop, ξ_pop), εI). 

See also: [`init_pop_sampler_opt`](@ref)
"""
struct KappaSigmaNormalSamplerOpt{T1<:AbstractFloat, 
                                  T2<:Signed, 
                                  T3<:Any}

    acc_prop::T1
    n_warm_up::T2
    variances::T3
end


"""
    KappaSigmaNormalSampler

Options and required data for (ĸ_pop, ξ_pop)-sampler used in PEPSDI opt 2. 
    
Intitialised by init_kappa_sigma_sampler. 

See also: [`init_kappa_sigma_sampler`](@ref)
"""
struct KappaSigmaNormalSampler{T1<:AbstractFloat, 
                               T2<:Signed, 
                               T3<:Array{<:AbstractFloat, 1}}
                           
    acc_prop::T1
    n_warm_up::T2
    variances::T3
    n_kappa_sigma::T2
end
struct KappaSigmaNormal
end


"""
    ParamInfoPop

Struct storing inference information for η, ĸ, and ξ parameters. 

Stores means, initial values, if positive or, if on log-scale or not, dimension, 
for the η, ĸ, and ξ parameters.
"""
struct ParamInfoPop{T1<:Array{<:Distribution, 1},
                    T2<:Array{<:Distribution, 1}, 
                    T3<:Distribution, 
                    T4<:Array{<:Distribution, 1}, 
                    T5<:Array{<:Distribution, 1},
                    T6<:Array{<:AbstractFloat, 1}, 
                    T7<:Array{<:AbstractFloat, 2},
                    T8<:Signed}

    prior_pop_param_mean::T1
    prior_pop_param_scale::T2
    prior_pop_param_corr::T3
    prior_pop_param_sigma::T4
    prior_pop_param_kappa::T5

    init_pop_param_mean::T6
    init_pop_param_scale::T6
    init_pop_param_corr::T7
    init_pop_param_sigma::T6
    init_pop_param_kappa::T6

    pos_pop_param_sigma::Array{Bool, 1}
    pos_pop_param_kappa::Array{Bool, 1}

    log_pop_param_sigma::Array{Bool, 1}
    log_pop_param_kappa::Array{Bool, 1}

    dim_eta::T8
    dim_kappa::T8
    dim_sigma::T8

    precision_scale::Bool
end


"""
    ParamInfoIndPre

Struct storing inference information for c_i the individual parameters. 

init_ind_param can be a matrix with values for each individual, array for value 
to use for all individuals, or (mean, median, random) to sample from the priors. 
"""
struct ParamInfoIndPre{T<:Any}
    init_ind_param::T
    pos_ind_param::Bool
    log_ind_param::Bool
    n_param::Int64
end


"""
    ParamInfoInd

Same as ParamInfoIndPre, except init_ind_param now has the stored inital values.  
"""
struct ParamInfoInd{T1<:Array{<:AbstractFloat, 1}, 
                    T2<:Signed}

    init_ind_param::T1
    pos_ind_param::Array{Bool, 1}
    log_ind_param::Array{Bool, 1}
    n_param::T2
end