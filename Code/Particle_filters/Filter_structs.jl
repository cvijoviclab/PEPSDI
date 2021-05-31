"""
    DynModInput

Struct storing model-quantity values (c, ĸ, covariates) for an individual. 
"""
struct DynModInput{T1<:Array{<:AbstractFloat, 1}, 
                   T2<:Array{<:AbstractFloat, 1}, 
                   T3<:Array{<:AbstractFloat, 1}}

    c::T1
    kappa::T2
    covariates::T3
end


"""
    TimeStepInfo

Time-stepping data for SDE and Poison (tau-leaping) particle propagators. 
"""
struct TimeStepInfo{T1<:AbstractFloat, T2<:Signed}
    t_start::T1
    t_end::T1
    n_step::T2
end


"""
    IndData

Observed data (measurement y_mat and time t_vec) for individual i.

Initalised by init_ind_data. For a SDE-model n_step contains the number of 
time-steps to perform betwen t_vec[i-1] and t_vec[i]. For a SSA/Extrande-model it 
is empty. In the case of multiple observed time-series, each row of y-mat 
corresponds to one measured specie. Cov_val corresponds to potential covariate 
values (left empty if there aren't any covariates)

See also: [`init_ind_data`](@ref)
"""
struct IndData{T1<:Array{<:AbstractFloat, 1}, 
               T2<:Array{<:AbstractFloat, 2}, 
               T3<:Array{<:Signed, 1}, 
               T4<:Array{<:AbstractFloat, 1}}

    t_vec::T1
    y_mat::T2
    n_step::T3
    cov_val::T4
end


"""
    RandomNumbers

Random-numbers for propegation and resampling SDE/Poisson-model particle-filters.

u_prop[i] contains random number to propegate between t_vec[i] to t_vec[i-1], 
and each entry can vary in size depending on spacing between observed 
time-points. 

See also: [`create_rand_num`](@ref)
"""
struct RandomNumbers{T1<:Array{<:Array{<:AbstractFloat, 2}, 1}, 
                     T2<:Array{<:AbstractFloat, 1}}

    u_prop::T1
    u_resamp::T2
end


"""
    RandomNumbersSsa

Random-numbers for resampling SSA/Extrande-model particle-filters.
"""
struct RandomNumbersSsa{T1<:Array{<:AbstractFloat, 1}}
    u_resamp::T1
end


"""
    ModelParameters

Rates, error-parameters, initial-values and covariates for individual i. 

If there are not any parameters, covariates is an empty vector. 

See also: [`init_model_parameters`](@ref)
"""
struct ModelParameters{T1<:Array{<:AbstractFloat, 1}, 
                       T2<:DynModInput, 
                       T3<:Array{<:Real, 1}}

    individual_parameters::T2
    x0::T3
    error_parameters::T1
    covariates::T1
end


"""
    BootstrapFilterPois

Options: time-step size (dt), number of particles, correlation level for tau-leaping (Poisson) bootstrap filter. 

If rho ∈ [0.0, 1.0) equals 0.0 the particles are uncorrelated. 
"""
struct BootstrapFilterPois{T1<:AbstractFloat, T2<:Signed}
    delta_t::T1
    n_particles::T2
    rho::T1    
end
struct BootstrapPois
end


"""
    BootstrapFilterEm

Options: time-step size (dt), number of particles, correlation level for Euler-Maruyama SDE bootstrap filter. 

If rho ∈ [0.0, 1.0) equals 0.0 the particles are uncorrelated. 
"""
struct BootstrapFilterEm{T1<:AbstractFloat, T2<:Signed}
    delta_t::T1
    n_particles::T2
    rho::T1    
end
struct BootstrapEm
end


"""
    BootstrapFilterSsa

Options: number of particles, for SSA (Gillespie) bootstrap filter. 

Particles cannot be correlated if the SSA algorithm is used to propegate particles. 
"""
struct BootstrapFilterSsa{T1<:AbstractFloat, T2<:Signed}
    n_particles::T2
    rho::T1    
end
struct BootstrapSsa
end


"""
    BootstrapFilterExtrand

Options: number of particles, for Extrande bootstrap filter. 

Particles cannot be correlated if the Extrande algorithm is used to propegate particles. 
"""
struct BootstrapFilterExtrand{T1<:AbstractFloat, T2<:Signed}
    n_particles::T2
    rho::T1    
end
struct BootstrapExtrand
end


"""
    ModDiffusionFilter

Options: time-step size (dt), number of particles, correlation level for modified diffusion bridge SDE bootstrap filter. 

If rho ∈ [0.0, 1.0) equals 0.0 the particles are uncorrelated. 
"""
struct ModDiffusionFilter{T1<:AbstractFloat, T2<:Signed}
    delta_t::T1
    n_particles::T2
    rho::T1    
end
struct ModDiffusion
end
