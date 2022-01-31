"""
    McmcSamplerRandWalk

Essential parameters (Σ, λ) for Random-Walk sampler. 

Initalised by init_mcmc. When proposing, the step-length is multiplied to 
to the covariance matrix. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerRandWalk{T1<:Array{<:AbstractFloat, 2}, 
                           T2<:AbstractFloat, 
                           T3<:Signed, 
                           T4<:Array{<:AbstractFloat, 2}, 
                           T5<:Array{<:AbstractFloat, 1}}

    cov_mat::T1
    step_length::T2
    dim::T3
    name_sampler::String
    cov_mat_old::T4
    mean::T5
end
struct RandomWalk
end


"""
    McmcSamplerAM

Essential parameters (μ, Σ, γ0) for AM-sampler

Initalised by init_mcmc. When proposing, γ = γ0 / (iteration^alpha_power). 
μ, Σ, are updated after step_before_update steps, and are then updated 
every 30:th iteration. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerAM{T1<:Array{<:AbstractFloat, 2}, 
                     T2<:Array{<:AbstractFloat, 1}, 
                     T3<:AbstractFloat, 
                     T4<:Signed}
    
    cov_mat::T1
    # For tuning covariance matrix 
    mu::T2
    gamma0::T3
    alpha_power::T3
    lambda::T3
    steps_before_update::T4
    dim::T4
    name_sampler::String
    update_it::T4
end
struct AmSampler
end


"""
    McmcSamplerGenAM

Essential parameters (μ, Σ, log(λ), α∗, γ0) for General AM-sampler

Initalised by init_mcmc. When updating, γ = γ0 / (iteration^alpha_power). 
μ, Σ, log(λ) are updated after step_before_update steps. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerGenAM{T1<:Array{<:AbstractFloat, 2}, 
                        T2<:Array{<:AbstractFloat, 1},
                        T3<:Array{<:AbstractFloat, 1},
                        T4<:AbstractFloat, 
                        T5<:Signed}

    cov_mat::T1
    # Parameters for tuning covariance matrix 
    mu::T2
    log_lambda::T3
    alpha_target::T4
    gamma0::T4
    alpha_power::T4
    steps_before_update::T5
    dim::T5
    name_sampler::String
    update_it::T5
end
struct GenAmSampler
end


"""
    McmcSamplerRam

Essential parameters (Σ, α∗, γ0) for RAM-sampler

Initalised by init_mcmc. When updating, γ = γ0 / (iteration^alpha_power). 
Σ is updated after step_before_update steps. q_vec is the random normal 
numbers used in proposing new-parameters. 

See also: [`init_mcmc`](@ref)
"""
struct McmcSamplerRam{T1<:Array{<:AbstractFloat, 2}, 
                      T2<:Array{<:AbstractFloat, 1},
                      T3<:AbstractFloat, 
                      T4<:Signed}
    cov_mat::T1
    q_vec::T2
    alpha_target::T3
    gamma0::T3
    alpha_power::T3
    steps_before_update::T4
    dim::T4
    name_sampler::String
    update_it::T4
end
struct RamSampler
end


"""
    PopSamplerOrnsteinOpt

Options (acceptance probability and number of warm-up samples) for Ornstein η-sampler. 

Intitialised by init_pop_sampler_opt. 

See also: [`init_pop_sampler_opt`](@ref)
"""
struct PopSamplerOrnsteinOpt{T1<:AbstractFloat, T2<:Signed}
    acc_prop::T1
    n_warm_up::T2
end


"""
    PopSamplerOrnstein

Options and required data for Ornstein η-sampler used in Gibbs-sampler.
    
Intitialised by init_pop_sampler. For speed (avoid looking up) number of 
individuals and number of parameters in addition to acceptance-probability
and number of warm-up samples are stored in the population sampler-struct. 

See also: [`init_pop_sampler`](@ref)
"""
struct PopSamplerOrnstein{T1<:AbstractFloat, T2<:Signed}
    acc_prop::T1
    n_warm_up::T2
    n_individuals::T2
    n_param::T2
end
struct PopOrnstein
end

struct PopSamplerNormalLjkTwo{T1<:AbstractFloat, T2<:Signed, T3<:Array{<:Integer, 1}, T4<:Array{<:Integer, 1}}
    acc_prop::T1
    n_warm_up::T2
    index_one::T3
    index_two::T3
    tag_one::T4
    tag_two::T4
    n_individuals::T2
    n_param::T2
    n_param_ind::T2
end
struct PopNormalLjkTwo
end


"""
    PopSamplerNormalLjkOpt

Options (acceptance probability and number of warm-up samples) for Normal-LKJ η-sampler. 

Intitialised by init_pop_sampler_opt. 

See also: [`init_pop_sampler_opt`](@ref)
"""
struct PopSamplerNormalLjkOpt{T1<:AbstractFloat, T2<:Signed}
    acc_prop::T1
    n_warm_up::T2
end


struct PopSamplerNormalLjkOptTwo{T1<:AbstractFloat, T2<:Signed, T3<:Array{<:Integer, 1}, T4<:Array{<:Integer, 1}}
    acc_prop::T1
    n_warm_up::T2
    index_one::T3
    index_two::T3
    tag_one::T4
    tag_two::T4
    n_param_ind::T2
end


"""
    PopSamplerNormalLjk

Options and required data for Normal-LKJ η-sampler used in Gibbs-sampler.
    
Intitialised by init_pop_sampler. For speed (avoid looking up) number of 
individuals and number of parameters in addition to acceptance-probability
and number of warm-up samples are stored in the population sampler-struct. 

See also: [`init_pop_sampler`](@ref)
"""
struct PopSamplerNormalLjk{T1<:AbstractFloat, T2<:Signed}
    acc_prop::T1
    n_warm_up::T2
    n_individuals::T2
    n_param::T2
end
struct PopNormalLKJ
end


"""
    PopSamplerNormalDiagOpt

Options (acceptance probability and number of warm-up samples) for Normal-diag η-sampler. 

Intitialised by init_pop_sampler_opt. 

See also: [`init_pop_sampler_opt`](@ref)
"""
struct PopSamplerNormalDiagOpt{T1<:AbstractFloat, T2<:Signed}
    acc_prop::T1
    n_warm_up::T2
end


"""
    PopSamplerNormalDiag

Options and required data for Normal-diag η-sampler used in Gibbs-sampler.
    
Intitialised by init_pop_sampler. For speed (avoid looking up) number of 
individuals and number of parameters in addition to acceptance-probability
and number of warm-up samples are stored in the population sampler-struct. 

See also: [`init_pop_sampler`](@ref)
"""
struct PopSamplerNormalDiag{T1<:AbstractFloat, T2<:Signed}
    acc_prop::T1
    n_warm_up::T2
    n_individuals::T2
    n_param::T2
end
struct PopNormalDiag
end