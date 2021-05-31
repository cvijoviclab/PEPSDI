"""
    DiffBridgeSolverObj

Drift, diffusion, observation, and state-arrays + step-length when propegating via the modifed diffusion bridge. 

Pre-allocated for computational efficiency. 
"""
struct DiffBridgeSolverObj{T1<:MArray,
                           T2<:MArray,
                           T3<:MArray,
                           T4<:SArray,
                           T5<:SArray,
                           T6<:MArray,
                           T7<:Array{<:AbstractFloat, 1}}

    mean_vec::T1
    alpha_vec::T1
    cov_mat::T2
    beta_mat::T2
    sigma_mat::T3
    P_mat::T4
    P_mat_t::T5
    x_vec::T6
    delta_t::T7
    sqrt_delta_t::T7
end


"""
    calc_log_det(X::T1, dim_X::T2) where {T1<:MArray{<:Tuple, FLOAT, 2}, T2<:Signed}

Calculate log-determinant for a semite positive definite matrix X of dimension dim_X. 
"""
@inline function calc_log_det(X::T1, dim_X::T2) where {T1<:MArray{<:Tuple, FLOAT, 2}, T2<:Signed}
    log_det::FLOAT = 0.0
    @inbounds @simd for i in 1:dim_X
        log_det += log(X[i, i])
    end
    log_det *= 2
    return log_det
end


"""
    calc_log_pdf_mvn(x_curr::T1, 
                     mean_vec::T1, 
                     chol_cov_mat::T2, 
                     dim_mod::T3)::FLOAT where{T1<:MArray{<:Tuple, FLOAT, 1}, T2<:MArray{<:Tuple, FLOAT, 2}, T3<:Signed}

Calculate log-pdf for a multivariate normal distribution at x_curr. 

Since already computed the cholesky decomposition of the covariance matrix is employed. 
"""
function calc_log_pdf_mvn(x_curr::T1, 
                          mean_vec::T1, 
                          chol_cov_mat::T2, 
                          dim_mod::T3)::FLOAT where{T1<:MArray{<:Tuple, FLOAT, 1}, T2<:MArray{<:Tuple, FLOAT, 2}, T3<:Signed}

    # log(2pi)
    const_term::FLOAT = 1.837877066

    # (x - mu)' * inv_∑ * (x - mu)
    MD2 = chol_cov_mat \ (x_curr - mean_vec)
    MD2 = dot(MD2, MD2)

    log_pdf::FLOAT = -0.5 * (const_term*dim_mod + calc_log_det(chol_cov_mat, dim_mod) + MD2)

    return log_pdf
end


"""
    init_sol_object(::Val{dim_model}, ::Val{dim_model_obs}, sde_mod::SdeModel)::DiffBridgeSolverObj where {dim_model, dim_model_obs}

Initialise solution-struct (DiffBridgeSolverObj) to pre-allocate matrices and vectors for propgating particles. 

Pre-allocates drift-vector (for both EM and bridge), diffusion-matrix (both EM and bridge), current particle-values at time, 
and step-length to propegate the particles in a memory efficient manner. As StaticArrays-are employed Val-input required to 
help compiler. 
"""
function init_sol_object(::Val{dim_model}, ::Val{dim_model_obs}, sde_mod::SdeModel)::DiffBridgeSolverObj where {dim_model, dim_model_obs}
    
    mean_vec = zeros(MVector{dim_model, FLOAT})
    alpha_vec = zeros(MVector{dim_model, FLOAT})
    
    beta_mat = zeros(MMatrix{dim_model, dim_model, FLOAT})
    cov_mat = zeros(MMatrix{dim_model, dim_model, FLOAT})

    x_curr = zeros(MVector{dim_model, FLOAT})

    sigma_mat = zeros(MMatrix{dim_model_obs, dim_model_obs, FLOAT})

    P_mat = deepcopy(sde_mod.P_mat)
    P_mat_t = deepcopy(sde_mod.P_mat')

    delta_t = Array{FLOAT, 1}(undef, 1)
    sqrt_delta_t = Array{FLOAT, 1}(undef, 1)

    solver_obj = DiffBridgeSolverObj(mean_vec,
                                     alpha_vec,
                                     cov_mat,
                                     beta_mat,
                                     sigma_mat,
                                     P_mat,
                                     P_mat_t,
                                     x_curr,
                                     delta_t,
                                     sqrt_delta_t)

    return solver_obj
end


"""
    modified_diffusion_calc_arrays!(p::DynModInput,                                           
                                    sde_mod::SdeModel, 
                                    so::DiffBridgeSolverObj, 
                                    delta_k::FLOAT,
                                    y_vec,
                                    t::FLOAT) 

Calculate arrays (drift- and diffusion for both EM and bridge) to propegate via modified diffusion bridge 
"""
function modified_diffusion_calc_arrays!(p::DynModInput,                                           
                                         sde_mod::SdeModel, 
                                         so::DiffBridgeSolverObj, 
                                         delta_k::FLOAT,
                                         y_vec,
                                         t::FLOAT) 
    
    # Calculate beta and alpha arrays 
    sde_mod.calc_alpha(so.alpha_vec, so.x_vec, p, t)
    sde_mod.calc_beta(so.beta_mat, so.x_vec, p, t)

    # Structs for calculating mean and covariance
    delta_t::FLOAT = so.delta_t[1]
    sqrt_delta_t::FLOAT = so.sqrt_delta_t[1]

    # Calculate new mean and covariance values 
    inv_term = so.beta_mat*so.P_mat*inv(so.P_mat_t*so.beta_mat*so.P_mat*delta_k + so.sigma_mat)

    so.mean_vec .= so.alpha_vec .+ inv_term * (y_vec - so.P_mat_t * (so.x_vec + so.alpha_vec * delta_k))
    
    so.cov_mat .= so.beta_mat .- inv_term*so.P_mat_t*so.beta_mat*delta_t

end


"""
    modified_diffusion_propegate!(prob_em::Array{FLOAT, 1}, 
                                  prob_bridge::Array{FLOAT, 1}, 
                                  so::DiffBridgeSolverObj, 
                                  u_vec, 
                                  i_particle::T1, 
                                  sde_mod::SdeModel) where T1<:Signed

Propegate the particles one time-step for the modified diffusion bridge. 
"""
function modified_diffusion_propegate!(prob_em::Array{FLOAT, 1}, 
                                       prob_bridge::Array{FLOAT, 1}, 
                                       so::DiffBridgeSolverObj, 
                                       u_vec, 
                                       i_particle::T1, 
                                       sde_mod::SdeModel) where T1<:Signed
     
    delta_t::FLOAT = so.delta_t[1]
    sqrt_delta_t::FLOAT = so.sqrt_delta_t[1]

    # Must calculate mean-vectors before propegating (when calculcating logpdf)
    mean_vec_bridge = so.x_vec + so.mean_vec*delta_t
    mean_vec_em = so.x_vec + so.alpha_vec*delta_t

    # Note, for propegation and then log-pdf cholesky decompositon is required for cov-mat and beta 
    calc_cholesky!(so.cov_mat, sde_mod.dim)
    calc_cholesky!(so.beta_mat, sde_mod.dim)

    # Propegate
    so.x_vec .+= so.mean_vec*delta_t + so.cov_mat*u_vec * sqrt_delta_t
    map_to_zero!(so.x_vec, sde_mod.dim)
    
    # Update probabilities, cov_mat and beta-mat are both lower-triangular cholesky
    prob_bridge[i_particle] += calc_log_pdf_mvn(so.x_vec, mean_vec_bridge, so.cov_mat*sqrt_delta_t, sde_mod.dim)
    prob_em[i_particle] += calc_log_pdf_mvn(so.x_vec, mean_vec_em, so.beta_mat*sqrt_delta_t, sde_mod.dim)
    
end


"""
    propegate_modified_diffusion!(x, c, t_step_info, sde_mod, n_particles, u)

Propegate n-particles in the modified diffusion bridge filter for a SDE-model. 

Propegates n-particles for an individual with parameter vector c between 
time-points t_step_info[1] and t_step_info[2] using t_step_info[3] steps 
betweeen the time-points. Old particle values x are overwritten for memory 
efficiency. Negative values are set to 0 to avoid negative square-roots. 
The auxillerary variables contain random normal numbers used to propegate, 
and the solver_obj contains pre-allocated matrices and vectors. 
"""
function propegate_modified_diffusion!(x::Array{FLOAT, 2}, 
                                       mod_param::ModelParameters,
                                       t_step_info::TimeStepInfo, 
                                       sde_mod::SdeModel, 
                                       n_particles::T1, 
                                       u::Array{FLOAT, 2}, 
                                       y_vec_obs, 
                                       prob_bridge_log::Array{FLOAT, 1}, 
                                       prob_em_log::Array{FLOAT, 1}, 
                                       solver_obj::DiffBridgeSolverObj) where {T1<:Signed}
    
    p::DynModInput = mod_param.individual_parameters
    error_parameters::Array{FLOAT, 1} = mod_param.error_parameters
    # Stepping options for the EM-stepper
    delta_t::FLOAT = (t_step_info.t_end - t_step_info.t_start) / t_step_info.n_step
    t_end::FLOAT = t_step_info.t_end
    t_vec = t_step_info.t_start:delta_t:t_step_info.t_end
    solver_obj.delta_t[1] = delta_t
    solver_obj.sqrt_delta_t[1] = sqrt(delta_t)
    solver_obj.sigma_mat[diagind(solver_obj.sigma_mat)] .= error_parameters.^2

    # Update each particle (note x is overwritten)
    n_states = 1:sde_mod.dim
    @inbounds for i in 1:n_particles
        i_acc = 1:sde_mod.dim
        prob_em_log[i] = 0.0
        prob_bridge_log[i] = 0.0
        solver_obj.x_vec .= x[:, i]
        @inbounds for j in 1:t_step_info.n_step     
            u_vec = @view u[i_acc, i] 

            t::FLOAT = t_vec[j]
            delta_k::FLOAT = t_end - t

            # Update vectors for propegation 
            modified_diffusion_calc_arrays!(p, sde_mod, solver_obj, delta_k, y_vec_obs, t) 

            # Propegate and update probability vectors 
            modified_diffusion_propegate!(prob_em_log, prob_bridge_log, solver_obj, u_vec, i, sde_mod)
            i_acc = i_acc .+ sde_mod.dim
            
        end
        x[:, i] .= solver_obj.x_vec
    end
end


"""
    run_filter(filt_opt::ModDiffusionFilter,
               model_parameters::ModelParameters, 
               random_numbers::RandomNumbers, 
               sde_mod::SdeModel, 
               individual_data::IndData)::FLOAT

Run bootstrap filter for modified diffusion bridge SDE-stepper to obtain unbiased likelihood estimate. 

Each filter takes the input filt_opt, model-parameter, random-numbers, model-struct and 
individual_data. The filter is optmised to be fast and memory efficient on a single-core. 

# Args
- `filt_opt`: filter options (ModDiffusionFilter-struct)
- `model_parameters`: none-transfmored unknown model-parameters (ModelParameters)
- `random_numbers`: auxillerary variables, random-numbers, used to estimate the likelihood (RandomNumbers-struct)
- `sde_mod`: underlaying SDE-model for calculating likelihood (SdeModel struct)
- `individual_data`: observed data, and number of time-steps to perform between data-points (IndData-struct)

See also: [`ModDiffusionFilter`, `ModelParameters`, `RandomNumbers`, `SdeModel`, `IndData`]
"""
function run_filter(filt_opt::ModDiffusionFilter,
                    model_parameters::ModelParameters, 
                    random_numbers::RandomNumbers, 
                    sde_mod::SdeModel, 
                    individual_data::IndData)::FLOAT

    # Nested function that updates the weights (normalised and non-normalised)
    # for the modified diffusion bridge filter. Note, weights are calculated on 
    # log-scale for keeping precision. 
    @inline function calc_weights!(i_t_vec)::FLOAT

        y_obs_sub = SubArray(y_mat, (i_dim_obs, i_t_vec))

        @inbounds for ix in 1:n_particles
            x_curr_sub = @view x_curr[:, ix] 
            sde_mod.calc_obs(y_mod_vec, x_curr_sub, c, t_vec[i_t_vec])
            prob_obs_log::FLOAT = log(sde_mod.calc_prob_obs(y_obs_sub, y_mod_vec, error_param, t_vec[i_t_vec], sde_mod.dim_obs))

            log_w_unormalised = (prob_em_log[ix] + prob_obs_log) - prob_bridge_log[ix]
            w_unormalised[ix] = exp(log_w_unormalised)
        end

        sum_w_unormalised_ret::FLOAT = sum(w_unormalised)
        w_normalised .= w_unormalised ./ sum_w_unormalised_ret
        return sum_w_unormalised_ret
    end

    # Extract individual parameters for propegation 
    n_particles::Int64 = filt_opt.n_particles
    c::DynModInput = model_parameters.individual_parameters
    error_param::Array{FLOAT, 1} = model_parameters.error_parameters

    # Extract individual data and discretization level between time-points 
    t_vec::Array{FLOAT, 1} = individual_data.t_vec
    y_mat::Array{FLOAT, 2} = individual_data.y_mat
    n_step_vec::Array{Int16, 1} = individual_data.n_step
    len_t_vec::Int64 = length(t_vec)
        
    # Pre-allocated variables required for looping in the filter
    x0_mat::Array{FLOAT, 2} = reshape(repeat(model_parameters.x0, n_particles), (sde_mod.dim, n_particles))
    x_curr::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (sde_mod.dim, n_particles))
    x_curr .= x0_mat
    w_unormalised::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    w_normalised::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_particles)
    prob_bridge_log::Array{FLOAT, 1} = zeros(FLOAT, n_particles) 
    prob_em_log::Array{FLOAT, 1} = zeros(FLOAT, n_particles) 
    y_mod_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, sde_mod.dim_obs)
    i_dim_obs = 1:sde_mod.dim_obs
    i_dim_mod = 1:sde_mod.dim
    n_particles_inv::FLOAT = convert(FLOAT, 1 / n_particles)

    log_lik::FLOAT = 0.0

    # If correlated-filter, convert standard-normal resampling numbers to 
    # standard uniform 
    if filt_opt.rho != 0
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
        u_resamp_vec_tmp = cdf(Normal(), u_resamp_vec_tmp)
    else
        u_resamp_vec_tmp = deepcopy(random_numbers.u_resamp)
    end
    u_resamp_vec::Array{FLOAT, 1} = u_resamp_vec_tmp

    # Propegate particles for t1 
    i_u_prop::Int64 = 1  # Which discretization level to access 
    i_col_u_mat = 1:n_particles  # Which random numbers to use for propegation 
        
    # Solver object for diffusion bridge (avoid allocaiton 
    solver_obj::DiffBridgeSolverObj = init_sol_object(Val(sde_mod.dim), Val(sde_mod.dim_obs), sde_mod)
    
    # Special case where t = 0 is not observed 
    y_vec_obs = zeros(MVector{sde_mod.dim_obs, FLOAT})
    if t_vec[1] > 0.0
        # Extract random numbers for propegation. 
        t_step_info = TimeStepInfo(0.0, t_vec[1], n_step_vec[i_u_prop])
        y_vec_obs .= y_mat[:, 1]
        
        try
            propegate_modified_diffusion!(x_curr, 
                                    model_parameters,
                                    t_step_info, 
                                    sde_mod, 
                                    n_particles, 
                                    random_numbers.u_prop[i_u_prop], 
                                    y_vec_obs, 
                                    prob_bridge_log, 
                                    prob_em_log, 
                                    solver_obj)                                       
        catch 
            return -Inf 
        end

        i_u_prop += 1
    end


    # Note, particles can be updated normally. If propagation did not occur 
    # then prob_bridge_log and prob_em_log are zero arrays 
    sum_w_unormalised::FLOAT = calc_weights!(1)
    log_lik += log(sum_w_unormalised * n_particles_inv)

    # Indices for resampling 
    i_resamp::Array{UInt32, 1} = Array{UInt32, 1}(undef, n_particles)

    # Propegate over remaning time-steps 
    for i_t_vec in 2:1:len_t_vec    

        # If correlated, sort x_curr
        if filt_opt.rho != 0
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
        y_vec_obs .= y_mat[:, i_t_vec]
                                     
        try 
            propegate_modified_diffusion!(x_curr, 
                                    model_parameters,
                                    t_step_info, 
                                    sde_mod, 
                                    n_particles, 
                                    random_numbers.u_prop[i_u_prop], 
                                    y_vec_obs, 
                                    prob_bridge_log, 
                                    prob_em_log, 
                                    solver_obj)  
        catch 
            return -Inf 
        end
        i_u_prop += 1

        # Update weights and calculate likelihood
        sum_w_unormalised = calc_weights!(i_t_vec)
        log_lik += log(sum_w_unormalised * n_particles_inv)
    end

    return log_lik

end
