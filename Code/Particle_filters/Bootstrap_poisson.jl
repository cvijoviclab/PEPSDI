"""
    stand_normal_to_stand_uniform!(u::Array{FLOAT, 1})

Converts array of standard-normal to standard uniform numbers via CDF-transform. 

Used to convert auxillerary-variables to standard normal in order to obtain Poisson-numbers. 
"""
function stand_normal_to_stand_uniform!(u::Array{FLOAT, 1})
    const_val::FLOAT = 0.707106781186
    @inbounds for i in eachindex(u)
        u[i] = 0.5 * (1.0 + erf(u[i] * const_val))
    end
end


"""
    propegate_poisson!(x::Array{Int64, 2}, 
                       p::DynModInput, 
                       t_step_info::TimeStepInfo, 
                       model::PoisonModel, 
                       n_particles::T1, 
                       u::Array{FLOAT, 2}) where {T1<:Signed}
    

Propegate n-particles (x) in the bootstrap filter for a tau-leaping stepper.  

Propegates n-particles for an individual with parameters p between time-points t_step_info.t_start 
and t_step_info.t_end using t_step_info.n_step. Old particle values x are overwritten for memory 
efficiency. Negative values are set to 0 to avoid negative square-roots. The auxillerary variables 
contain random normal numbers used to propegate the particles. 
"""
function propegate_poisson!(x::Array{Int64, 2}, 
                            p::DynModInput, 
                            t_step_info::TimeStepInfo, 
                            model::PoisonModel, 
                            n_particles::T1, 
                            u::Array{FLOAT, 2}) where {T1<:Signed}
    
    # Stepping options for the EM-stepper
    delta_t::FLOAT = (t_step_info.t_end - t_step_info.t_start) / t_step_info.n_step
    t_vec = t_step_info.t_start:delta_t:t_step_info.t_end
    
    # Update each particle (note x is overwritten)
    n_reactions = model.n_reactions
    h_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_reactions)
    u_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_reactions * t_step_info.n_step)

    @inbounds for i in 1:n_particles
        
        x_curr = @view x[:, i]
        # Copy as the random numbers will be transformed to standard uniform 
        u_vec .= u[:, i]
        stand_normal_to_stand_uniform!(u_vec)

        # To acces random numbers correctly 
        i_acc = 1:model.n_reactions
        
        @inbounds for j in 1:t_step_info.n_step
            
            rand_vec = @view u_vec[i_acc]
            perform_poisson_step!(x_curr, model, h_vec, delta_t, p, t_vec[j], rand_vec)

            # Ensure only >= 0 states 
            map_to_zero_poison!(x_curr, model.dim)
            i_acc = i_acc .+ model.n_reactions
        end
    end

end


"""
    run_filter(filt_opt::BootstrapFilterPois,
               model_parameters::ModelParameters, 
               random_numbers::RandomNumbers, 
               sde_mod::SdeModel, 
               individual_data::IndData)::FLOAT

Run bootstrap filter for tau-lepaing (Poisson)-stepper to obtain unbiased likelihood estimate. 

Each filter takes the input filt_opt, model-parameter, random-numbers, model-struct and 
individual_data. The filter is optmised to be fast and memory efficient on a single-core. 

# Args
- `filt_opt`: filter options (BootstrapFilterEm-struct)
- `model_parameters`: none-transfmored unknown model-parameters (ModelParameters)
- `random_numbers`: auxillerary variables, random-numbers, used to estimate the likelihood (RandomNumbers-struct)
- `model`: underlaying Poisson-model for calculating likelihood (PoisonModel struct)
- `individual_data`: observed data, and number of time-steps to perform between data-points (IndData-struct)

See also: [`BootstrapFilterEm`, `ModelParameters`, `RandomNumbers`, `PoisonModel`, `IndData`]
"""
function run_filter(filter_opt::BootstrapFilterPois,
                    model_parameters::ModelParameters, 
                    random_numbers::RandomNumbers, 
                    model::PoisonModel, 
                    individual_data::IndData)::FLOAT

    # Nested function that updates the weights (normalised and non-normalised)
    # for the bootstrap filter. (Nested function typically do
    # not decrease performance in Julia)
    @inline function calc_weights!(w_unormalised, w_normalised, i_t_vec)::FLOAT

        y_obs_sub = SubArray(y_mat, (i_dim_obs, i_t_vec))
        @inbounds for i in 1:n_particles
            x_curr_sub = @view x_curr[:, i] 
            model.calc_obs(y_mod_vec, x_curr_sub, p, t_vec[i_t_vec])

            w_unormalised[i] = model.calc_prob_obs(y_obs_sub, y_mod_vec, error_param, t_vec[i_t_vec], model.dim_obs)
        end
        sum_w_unormalised_ret::FLOAT = sum(w_unormalised)
        w_normalised .= w_unormalised ./ sum_w_unormalised_ret

        return sum_w_unormalised_ret
    end

    # Extract individual parameters for propegation 
    n_particles::Int64 = filter_opt.n_particles
    p::DynModInput = model_parameters.individual_parameters
    error_param::Array{FLOAT, 1} = model_parameters.error_parameters

    # Extract individual data and discretization level between time-points 
    t_vec::Array{FLOAT, 1} = individual_data.t_vec
    y_mat::Array{FLOAT, 2} = individual_data.y_mat
    n_step_vec::Array{Int16, 1} = individual_data.n_step
    len_t_vec::Int64 = length(t_vec)
    
    # Pre-allocated variables required for looping in the filter 
    x0_mat::Array{Int64, 2} = reshape(repeat(model_parameters.x0, n_particles), (model.dim, n_particles))
    x_curr::Array{Int64, 2} = deepcopy(x0_mat)
    w_unormalised::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    w_normalised::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    y_mod_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, model.dim_obs)
    i_dim_obs = 1:model.dim_obs
    i_dim_mod = 1:model.dim
    n_particles_inv::FLOAT = convert(FLOAT, 1 / n_particles)

    log_lik::FLOAT = 0.0

    # If correlated-filter, convert standard-normal resampling numbers to 
    # standard uniform 
    if filter_opt.rho != 0
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
        u_resamp_vec_tmp = cdf(Normal(), u_resamp_vec_tmp)
    else
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
    end
    u_resamp_vec::Array{FLOAT, 1} = u_resamp_vec_tmp

    # Propegate particles for t1 
    i_u_prop::Int64 = 1  # Which discretization level to access 
    i_col_u_mat = 1:n_particles  # Which random numbers to use for propegation 

    if t_vec[1] > 0.0
        t_step_info = TimeStepInfo(0.0, t_vec[1], n_step_vec[i_u_prop])
        try 
            propegate_poisson!(x_curr, p, t_step_info, model, n_particles, random_numbers.u_prop[i_u_prop])
        catch 
            return -Inf 
        end

        i_u_prop += 1
    end

    # Update likelihood first time
    sum_w_unormalised::FLOAT = calc_weights!(w_unormalised, w_normalised, 1)
    log_lik += log(sum_w_unormalised * n_particles_inv)

    # Indices for resampling 
    i_resamp::Array{UInt32, 1} = Array{UInt32, 1}(undef, n_particles)

    # Propegate over remaning time-steps 
    for i_t_vec in 2:1:len_t_vec    
        
        # If correlated, sort x_curr
        if filter_opt.rho != 0
            data_sort = sum(x_curr.^2, dims=1)[1, :]
            i_sort = sortperm(data_sort)
            x_curr = x_curr[:, i_sort]
            w_normalised = w_normalised[i_sort]
        end

        u_resample = u_resamp_vec[i_t_vec-1]
        systematic_resampling!(i_resamp, w_normalised, n_particles, u_resample)
        x_curr = x_curr[:, i_resamp]
        
        # Variables for propeating correct particles  
        t_step_info = TimeStepInfo(t_vec[i_t_vec-1], t_vec[i_t_vec], n_step_vec[i_u_prop])
        
        #propegate_poisson!(x_curr, p, t_step_info, model, n_particles, random_numbers.u_prop[i_u_prop])    
        try 
            propegate_poisson!(x_curr, p, t_step_info, model, n_particles, random_numbers.u_prop[i_u_prop])    
        catch 
            return -Inf 
        end
        i_u_prop += 1
        
        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights!(w_unormalised, w_normalised, i_t_vec)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik
end

