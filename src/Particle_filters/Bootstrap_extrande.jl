"""
    propegate_extrand_bootstrap!(x_curr::Array{UInt16, 2}, 
                                 t_curr::Array{FLOAT, 1},
                                 p::DynModInput, 
                                 model::ExtrandModel,
                                 h_vec::Array{FLOAT, 1},
                                 t_end::T1,
                                 n_particles::T2) where {T1<:AbstractFloat, T2<:Signed}
    
Propegate n-particles (x) in the bootstrap filter for a Extrande stepper.  

Propegates n-particles for an individual with parameters p from t_curr to t_end. Old particle values x are 
overwritten for memory efficiency. The propensity vector is pre-allocated for efficiency. 
"""
function propegate_extrand_bootstrap!(x_curr::Array{UInt16, 2}, 
                                      t_curr::Array{FLOAT, 1},
                                      p::DynModInput, 
                                      model::ExtrandModel,
                                      h_vec::Array{FLOAT, 1},
                                      t_end::T1,
                                      n_particles::T2) where {T1<:AbstractFloat, T2<:Signed}
    
    # Update each particle (note x is overwritten)
    rng::MersenneTwister = MersenneTwister()
    n_states = 1:model.dim
    iter::Int64 = 0
    @inbounds for i in 1:n_particles
        x_ind = @view x_curr[:, i]
        t_start::FLOAT = t_curr[i]
        t_curr[i] = step_extrande_method!(x_ind, t_start, t_end, model, h_vec, p, rng) 
    end
end


"""
    run_filter(filt_opt::BootstrapFilterExtrand,
               model_parameters::ModelParameters, 
               random_numbers::RandomNumbers, 
               sde_mod::SdeModel, 
               individual_data::IndData)::FLOAT

Run bootstrap filter for Extrande stepper to obtain unbiased likelihood estimate. 

Each filter takes the input filt_opt, model-parameter, random-numbers, model-struct and 
individual_data. The filter is optmised to be fast and memory efficient on a single-core. 

# Args
- `filt_opt`: filter options (BootstrapFilterExtrand-struct)
- `model_parameters`: none-transfmored unknown model-parameters (ModelParameters)
- `random_numbers`: auxillerary variables, random-numbers, used for resampling to estimate the likelihood (RandomNumbersSsa-struct)
- `model`: underlaying SSA-model for calculating likelihood (ExtrandModel struct)
- `individual_data`: observed data, and number of time-steps to perform between data-points (IndData-struct)

See also: [`BootstrapFilterEm`, `ModelParameters`, `RandomNumbersSsa`, `SdeModel`, `IndData`]
"""
function run_filter(filter_opt::BootstrapFilterExtrand,
                    model_parameters::ModelParameters, 
                    random_numbers::RandomNumbersSsa, 
                    model::ExtrandModel, 
                    individual_data::IndData)::FLOAT

    # Nested function that updates the weights (normalised and non-normalised)
    # for the bootstrap filter. (Nested function typically do
    # not decrease performance in Julia)
    @inline function calc_weights!(w_unormalised::Array{FLOAT, 1}, 
                                   w_normalised::Array{FLOAT, 1}, 
                                   i_t_vec::Int64, 
                                   x_curr::Array{UInt16, 2})::FLOAT

        y_obs_sub = @view y_mat[:, i_t_vec]
        @inbounds for i in 1:n_particles
            x_curr_sub = @view x_curr[:, i]
            model.calc_obs(y_mod_vec, x_curr_sub, p, t_vec[i_t_vec])
            w_unormalised[i] = model.calc_prob_obs(y_obs_sub, 
                y_mod_vec, error_param, t_vec[i_t_vec], model.dim_obs)
        end
        new_sum_w_unormalised::FLOAT = sum(w_unormalised)
        w_normalised .= w_unormalised ./ new_sum_w_unormalised
        return new_sum_w_unormalised
    end

    # Get input 
    n_particles::Int64 = convert(Int64, filter_opt.n_particles)
    p::DynModInput{Array{Float64,1},Array{Float64,1}} = model_parameters.individual_parameters
    error_param::Array{FLOAT, 1} = model_parameters.error_parameters

    # Indiviudal observed data
    t_vec::Array{FLOAT, 1} = individual_data.t_vec
    y_mat::Array{FLOAT, 2} = individual_data.y_mat
    len_t_vec::Int64 = length(t_vec)

    # For propegating over individuals 
    x0_mat::Array{UInt16, 2} = reshape(repeat(model_parameters.x0, n_particles), (model.dim, n_particles))
    x_curr::Array{UInt16, 2} = deepcopy(x0_mat)
    w_unormalised::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    w_normalised::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    y_mod_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, model.dim_obs)
    i_dim_obs = 1:model.dim_obs
    i_dim_mod = 1:model.dim
    n_particles_inv::FLOAT = convert(FLOAT, 1 / n_particles)

    # h-vector for Gillespie
    h_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, model.n_reactions)
    t_curr_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    t_curr_vec .= 0.0
    
    log_lik::FLOAT = 0.0

    # Random numbers resampling 
    u_resamp_vec::Array{FLOAT, 1} = deepcopy(random_numbers.u_resamp)

    # Special case where t = 0 is not observed 
    t_end::FLOAT = 0.0
    iter::Int64 = 0
    if t_vec[1] > 0.0
        # Extract random numbers for propegation. SubArray avoid allocating # new copies 
        t_curr_vec .= 0.0
        t_end = t_vec[1]

        propegate_extrand_bootstrap!(x_curr, t_curr_vec, p, model, h_vec, t_end, n_particles)
    end

    # Update likelihood first time
    sum_w_unormalised::FLOAT = calc_weights!(w_unormalised, w_normalised, 1, x_curr)
    log_lik += log(sum_w_unormalised * n_particles_inv)

    # Indices for resampling 
    i_resamp::Array{UInt32, 1} = Array{UInt32, 1}(undef, n_particles)

    # Propegate over remaning time-steps 
    for i_t_vec in 2:1:len_t_vec    

        u_resample = u_resamp_vec[i_t_vec-1]
        systematic_resampling!(i_resamp, w_normalised, n_particles, u_resample)
        x_curr = x_curr[:, i_resamp]
        t_curr_vec = t_curr_vec[i_resamp]

        # Variables for propeating correct particles  
        t_end = t_vec[i_t_vec]
        propegate_extrand_bootstrap!(x_curr, t_curr_vec, p, model, h_vec, t_end, n_particles)

        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights!(w_unormalised, w_normalised, i_t_vec, x_curr)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik
end
