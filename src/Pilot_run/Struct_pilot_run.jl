"""
    TuneParticlesIndividual

Options when tuning particles for individual infernece. 

Intitialised by init_pilot_run. 

See also: [`init_pilot_run`](@ref)
"""
struct TuneParticlesIndividual{T1<:Signed,
                               T2<:Signed,
                               T3<:Signed,
                               T4<:Array{<:Signed, 1}, 
                               T5<:Array{FLOAT, 1}} 
                                
    n_particles_pilot::T1
    n_samples::T2
    n_particles_investigate::T4
    init_ind_param
    init_error
    n_times_run_filter::T3
    rho_list::T5
end


"""
    TuneParticlesMixed

Struct storing options for pilot run. Initalised by [`init_pilot_run_info`](@ref)

Stored options are number of particles for pilot, number of samples pilot, 
initial values for η, ĸ, ξ parameters, number of particles to test in the tuning, 
number of times to run filter when testing particles and correlation levels to 
tune for. 
"""
struct TuneParticlesMixed{T1<:Signed,
                          T2<:Array{<:AbstractFloat, 1}, 
                          T3<:Array{<:AbstractFloat, 1}, 
                          T4<:Array{<:AbstractFloat, 1},
                          T5<:Array{<:AbstractFloat, 1}, 
                          T6<:Array{<:AbstractFloat, 1}} 
                                
    n_particles_pilot::T1
    n_samples_pilot::T1
    init_mean::T2
    init_scale::T3
    init_kappa::T4
    init_sigma::T5
    n_particles_try::T1
    n_times_run_filter::T1
    rho_list::T6
end