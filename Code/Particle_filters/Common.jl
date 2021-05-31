#= 
    Common functions used by all filters. 

    Create ind-data struct 
    Create and update random numbers 
    Create model-parameters struct 
    Init filter struct 
    Change option for filter struct 
=# 


"""
    create_n_step_vec(t_vec, delta_t)

Calculate number of time-steps for EM-solver between each observed time-point 

The discretization level (delta_t) is provided by the user, and should be 
small enough to ensure accuracy, while keeping computational effiency. 
"""
function create_n_step_vec(t_vec, delta_t)
    # Adapt length of n_step if t[1] != 0
    len_step = 0
    zero_in_t = false
    if t_vec[1] == 0
        len_step = length(t_vec) - 1
        zero_in_t = true
    else
        len_step = length(t_vec)
    end
    n_step::Array{Int16, 1} = Array{Int16, 1}(undef, len_step)
    i_n_step = 1

    # Special case where t = 0 is not observed
    if !zero_in_t
        n_step[1] = convert(Int16, round((t_vec[1] - 0.0) / delta_t))
        i_n_step += 1
    end

    # Fill the step-vector
    for i_t_vec in 2:length(t_vec)
        n_steps = round((t_vec[i_t_vec] - t_vec[i_t_vec-1]) / delta_t)
        n_step[i_n_step] = convert(Int16, n_steps)
        i_n_step += 1
    end

    # Ensure correct type 
    n_step = convert(Array{Int16, 1}, n_step)

    return n_step
end


"""
    init_ind_data(path::String, delta_t::FLOAT)

Create IndData-struct using provided path to data-file and time-discretization. 

The data-file in path must be a csv-file in tidy-format. 

See also: [`IndData`](@ref)
"""
function init_ind_data(data_obs::DataFrame, filter_opt; cov_name::T1=[""]) where T1<:Array{<:String, 1}

    # Avoid duplicated time-points 
    if typeof(filter_opt) <: BootstrapFilterSsa || typeof(filter_opt) <: BootstrapFilterExtrand
        delta_t = 1.0
    else
        delta_t = filter_opt.delta_t
    end
    t_vec::Array{FLOAT, 1} = deepcopy(convert(Array{FLOAT, 1}, data_obs[!, "time"]))
    unique!(t_vec); sort!(t_vec)

    # General parmeters for the data set 
    n_data_points = length(t_vec)
    len_dim_obs_model = length(unique(data_obs[!, "obs_id"]))
    y_mat::Array{FLOAT, 2} = fill(convert(FLOAT, NaN), (len_dim_obs_model, n_data_points))

    i_obs = unique(data_obs[!, "obs_id"])

    # Fill y_mat for each obs id 
    for i in 1:len_dim_obs_model
        i_data_obs_i = (i_obs[i] .== data_obs[!, "obs_id"])
        t_vec_obs = data_obs[!, "time"]
        y_vec_obs = data_obs[!, "obs"]

        y_vec_obs = y_vec_obs[i_data_obs_i]
        t_vec_obs = t_vec_obs[i_data_obs_i]

        # Populate observation matrix 
        for j in 1:length(t_vec_obs)
            i_t = findall(x->x==t_vec_obs[j], t_vec)[1]
            y_mat[i, i_t] = y_vec_obs[j]
        end

    end

    if cov_name[1] != ""
        cov_vals = Array{FLOAT, 1}(undef, length(cov_name))
        for i in 1:length(cov_name)
            cov_vals[i] = data_obs[1, cov_name[i]]        
        end
    else
        cov_vals = Array{FLOAT, 1}(undef, 0)
    end 

    ind_data = IndData(t_vec, y_mat, create_n_step_vec(t_vec, delta_t), cov_vals)

    return ind_data
end


"""
    create_rand_num(ind_data::IndData, sde_mod::SdeModel, filter; rng::MersenneTwister=MersenneTwister())

Allocate and initalise RandomNumbers-struct for a SDE-model particle filter. 

Ind_data object provides the number of time-step between each observed 
time-point. If non-correlated particles are used, resampling numbers 
are standard uniform, else standard normal. 

See also: [`RandomNumbers`](@ref)
"""
function create_rand_num(ind_data::IndData, sde_mod::SdeModel, filter; rng::MersenneTwister=MersenneTwister())
    # Allocate propegating particles
    n_particles = filter.n_particles
    dist_prop = Normal()
    len_u = length(ind_data.n_step)
    u_prop::Array{Array{FLOAT, 2}, 1} = Array{Array{FLOAT, 2}, 1}(undef, len_u)
    for i in 1:len_u
        n_row = convert(Int64, sde_mod.dim*ind_data.n_step[i])
        n_col = n_particles
        u_mat = randn(rng, Float64, (n_row, n_col))
        
        u_prop[i] = u_mat
    end

    # Numbers used for systematic resampling
    if filter.rho == 0
        u_sys_res = rand(rng, length(ind_data.t_vec))
    elseif filter.rho != 0
        u_sys_res = randn(rng, Float64, length(ind_data.t_vec))
    end
    
    return RandomNumbers(u_prop, u_sys_res)
end
"""
    create_rand_num(ind_data::IndData, model::PoisonModel, filter; rng::MersenneTwister=MersenneTwister())

Allocate and initalise RandomNumbers-struct for a Poisson(tau-lepaing)-model particle filter. 

Only the number of random-numbers required differ from SDE-models. 
"""
function create_rand_num(ind_data::IndData, model::PoisonModel, filter; rng::MersenneTwister=MersenneTwister())
    # Allocate propegating particles
    n_particles = filter.n_particles
    dist_prop = Normal()
    len_u = length(ind_data.n_step)
    u_prop::Array{Array{FLOAT, 2}, 1} = Array{Array{FLOAT, 2}, 1}(undef, len_u)
    for i in 1:len_u
        n_row = convert(Int64, model.n_reactions * ind_data.n_step[i])
        n_col = n_particles
        u_mat = randn(rng, Float64, (n_row, n_col))
        
        u_prop[i] = u_mat
    end

    # Numbers used for systematic resampling
    if filter.rho == 0
        u_sys_res = rand(rng, length(ind_data.t_vec))
    elseif filter.rho != 0
        u_sys_res = randn(rng, Float64, length(ind_data.t_vec))
    end
    
    return RandomNumbers(u_prop, u_sys_res)
end
"""
    create_rand_num(ind_data::IndData, model::SsaModel, filter; rng::MersenneTwister=MersenneTwister())::RandomNumbersSsa

Create random-numbers for SSA (Gillespie) model. Only numbers to resample in the filter are created. 
"""
function create_rand_num(ind_data::IndData, model::SsaModel, filter; rng::MersenneTwister=MersenneTwister())::RandomNumbersSsa
    u_sys_res = rand(rng, length(ind_data.t_vec))
    return RandomNumbersSsa(u_sys_res)
end
"""
    create_rand_num(ind_data::IndData, model::SsaModel, filter; rng::MersenneTwister=MersenneTwister())::RandomNumbersSsa

Create random-numbers for Extrande model. Only numbers to resample in the filter are created. 
"""
function create_rand_num(ind_data::IndData, model::ExtrandModel, filter; rng::MersenneTwister=MersenneTwister())
    u_sys_res = rand(rng, length(ind_data.t_vec))
    return RandomNumbersSsa(u_sys_res)
end


"""
    update_random_numbers!(rand_num::RandomNumbers, 
                           filter::BootstrapFilterEm;
                           rng::MersenneTwister=MersenneTwister())

    
Update random numbers, auxillerary variables, (propegation and resampling) for the Boostrap EM-filter. 

If the particles are correlated particles u are updated via a Crank-Nichelson scheme. 

Overwrites the rand_num with the new random-numbers. 
"""
function update_random_numbers!(rand_num::RandomNumbers, 
                                filter::BootstrapFilterEm;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::FLOAT = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            dim = size(rand_num.u_prop[i])
            rand_num.u_prop[i] .= (filter.rho .* rand_num.u_prop[i]) + randn(rng, FLOAT, dim) *std 
        end

        dim = size(rand_num.u_resamp)
        rand_num.u_resamp .= (filter.rho .* rand_num.u_resamp) + randn(rng, FLOAT, dim) * std
    end
end
"""
    update_random_numbers!(rand_num_new::RandomNumbers, 
                           rand_num_old::RandomNumbers, 
                           filter::BootstrapFilterEm;
                           rng::MersenneTwister=MersenneTwister())

Using rand_num_old the new-random numbers are stored in rand_num_new.
"""
function update_random_numbers!(rand_num_new::RandomNumbers, 
                                rand_num_old::RandomNumbers, 
                                filter::BootstrapFilterEm;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num_new.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num_new.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::FLOAT = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            randn!(rand_num_new.u_prop[i])
            rand_num_new.u_prop[i] .*= std
            rand_num_new.u_prop[i] .+= filter.rho .* rand_num_old.u_prop[i]
        end

        randn!(rand_num_new.u_resamp)
        rand_num_new.u_resamp .*= std
        rand_num_new.u_resamp .+= filter.rho .* rand_num_old.u_resamp
    end
end
"""
    update_random_numbers!(rand_num::RandomNumbers, 
                           filter::ModDiffusionFilter;
                           rng::MersenneTwister=MersenneTwister())

    
Update random numbers, auxillerary variables, (propegation and resampling) for the modified diffusion bridge 

Same approach as bootstrap EM. 
"""
function update_random_numbers!(rand_num::RandomNumbers, 
                                filter::ModDiffusionFilter, 
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::FLOAT = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            dim = size(rand_num.u_prop[i])
            rand_num.u_prop[i] .= (filter.rho .* rand_num.u_prop[i]) + randn(rng, FLOAT, dim) *std 
        end

        dim = size(rand_num.u_resamp)
        rand_num.u_resamp .= (filter.rho .* rand_num.u_resamp) + randn(rng, FLOAT, dim) * std
    end
end
"""
    update_random_numbers!(rand_num_new::RandomNumbers, 
                           rand_num_old::RandomNumbers, 
                           filter::ModDiffusionFilter;
                           rng::MersenneTwister=MersenneTwister())
"""
function update_random_numbers!(rand_num_new::RandomNumbers, 
                                rand_num_old::RandomNumbers, 
                                filter::ModDiffusionFilter;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num_new.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num_new.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::FLOAT = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
            rand_num_new.u_prop[i] .*= std
            rand_num_new.u_prop[i] .+= filter.rho .* rand_num_old.u_prop[i]
        end

        randn!(rng, rand_num_new.u_resamp)
        rand_num_new.u_resamp .*= std
        rand_num_new.u_resamp .+= filter.rho .* rand_num_old.u_resamp
    end
end
"""
    update_random_numbers!(rand_num::RandomNumbers, 
                           filter::BootstrapFilterPois;
                           rng::MersenneTwister=MersenneTwister())

Update using the same-scheme as for SDE-model, but for a Poisson model where more 
random numbers are required. 
"""
function update_random_numbers!(rand_num::RandomNumbers, 
                                filter::BootstrapFilterPois;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::FLOAT = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            dim = size(rand_num.u_prop[i])
            rand_num.u_prop[i] .= (filter.rho .* rand_num.u_prop[i]) + randn(rng, FLOAT, dim) *std 
        end

        dim = size(rand_num.u_resamp)
        rand_num.u_resamp .= (filter.rho .* rand_num.u_resamp) + randn(rng, FLOAT, dim) * std
    end
end
"""
    update_random_numbers!(rand_num_new::RandomNumbers, 
                           rand_num_old::RandomNumbers, 
                           filter::BootstrapFilterPois;
                           rng::MersenneTwister=MersenneTwister())

Update using the same-scheme as for SDE-model, but for a Poisson model where more 
random numbers are required. 
"""
function update_random_numbers!(rand_num_new::RandomNumbers, 
                                rand_num_old::RandomNumbers, 
                                filter::BootstrapFilterPois;
                                rng::MersenneTwister=MersenneTwister())

    n_time_step_updates = length(rand_num_new.u_prop)
    # Update u-propegation in case of no correlation 
    if filter.rho == 0
        for i in 1:n_time_step_updates
            randn!(rng, rand_num_new.u_prop[i])
        end

        # Resampling random-numbers 
        rand!(rng, rand_num_new.u_resamp)

    elseif filter.rho != 0
    # Note, propegation numbers become normal if using correlation 
        std::FLOAT = sqrt(1 - filter.rho^2)
        for i in 1:n_time_step_updates
            randn!(rand_num_new.u_prop[i])
            rand_num_new.u_prop[i] .*= std
            rand_num_new.u_prop[i] .+= filter.rho .* rand_num_old.u_prop[i]
        end

        randn!(rand_num_new.u_resamp)
        rand_num_new.u_resamp .*= std
        rand_num_new.u_resamp .+= filter.rho .* rand_num_old.u_resamp
    end
end
"""
    update_random_numbers!(rand_num::RandomNumbersSsa, 
                           filter::BootstrapFilterPois;
                           rng::MersenneTwister=MersenneTwister())

Update random-numbers (auxillerary-variables) for SSA- and extrande filter. Here, 
only the resampling numbers are updated. 
"""
function update_random_numbers!(rand_num::RandomNumbersSsa, 
                                filter::BootstrapFilterSsa)
    rand!(Uniform(), rand_num.u_resamp)
end
function update_random_numbers!(rand_num::RandomNumbersSsa, 
                                rand_num_old::RandomNumbersSsa,
                                filter::BootstrapFilterSsa)
    rand!(Uniform(), rand_num.u_resamp)
end
"""
    update_random_numbers!(rand_num::RandomNumbersSsa, 
                           filter::BootstrapFilterPois;
                           rng::MersenneTwister=MersenneTwister())

Update random-numbers (auxillerary-variables) for SSA- and extrande filter. Here, 
only the resampling numbers are updated. 
"""
function update_random_numbers!(rand_num::RandomNumbersSsa, 
                                filter::BootstrapFilterExtrand)
    rand!(Uniform(), rand_num.u_resamp)
end
function update_random_numbers!(rand_num::RandomNumbersSsa, 
                                rand_num_old::RandomNumbersSsa,
                                filter::BootstrapFilterExtrand)
    rand!(Uniform(), rand_num.u_resamp)
end


"""
    init_model_parameters(ind_param, error_param, model::SdeModel; covariates=false, kappa=false)::ModelParameters

Allocate and initalise ModelParameters-struct for sde-model (initial values Float64)

Individual parameters correspond to rate-constants and/or initial values. 
Error_param is assumed to be an array.

See also: [`ModelParameters`](@ref)
""" 
function init_model_parameters(ind_param, error_param, model::SdeModel; covariates=false, kappa=false)::ModelParameters
    if covariates == false
        covariates = Array{FLOAT, 1}(undef, 0)
    end

    if kappa == false
        kappa = Array{FLOAT, 1}(undef, 0)
        ind_param = DynModInput(ind_param, kappa, covariates)
    else
        ind_param = DynModInput(ind_param, kappa, covariates)
    end

    x0 = Array{FLOAT, 1}(undef, model.dim)
    model.calc_x0!(x0, ind_param)

    # Struct object 
    return ModelParameters(ind_param, x0, error_param, covariates)
end
"""
    init_model_parameters(ind_param, error_param, model::SsaModel; covariates=false, kappa=false)::ModelParameters

Allocate and initalise ModelParameters-struct for ssa-model (initial values UInt16)
"""
function init_model_parameters(ind_param, error_param, model::SsaModel; covariates=false, kappa=false)::ModelParameters
    if covariates == false
        covariates = Array{FLOAT, 1}(undef, 0)
    end
    x0::Array{UInt16} = Array{UInt16, 1}(undef, model.dim)
    model.calc_x0!(x0, ind_param)

    if kappa == false
        kappa = Array{FLOAT, 1}(undef, 0)
        ind_param = DynModInput(ind_param, kappa, covariates)
    else
        ind_param = DynModInput(ind_param, kappa, covariates)
    end

    # Struct object 
    return ModelParameters(ind_param, x0, error_param, covariates)
end
"""
    init_model_parameters(ind_param, error_param, model::ExtrandModel; covariates=false, kappa=false)::ModelParameters

Allocate and initalise ModelParameters-struct for Extrande-model (initial values UInt16)
"""
function init_model_parameters(ind_param, error_param, model::ExtrandModel; covariates=false, kappa=false)::ModelParameters
    if covariates == false
        covariates = Array{FLOAT, 1}(undef, 0)
    end
    x0::Array{UInt16} = Array{UInt16, 1}(undef, model.dim)
    model.calc_x0!(x0, ind_param)

    if kappa == false
        kappa = Array{FLOAT, 1}(undef, 0)
        ind_param = DynModInput(ind_param, kappa, covariates)
    else
        ind_param = DynModInput(ind_param, kappa, covariates)
    end

    # Struct object 
    return ModelParameters(ind_param, x0, error_param, covariates)
end
"""
    init_model_parameters(ind_param, error_param, model::PoisonModel; covariates=false, kappa=false)::ModelParameters

Allocate and initalise ModelParameters-struct for Poisson-model (initial values Int32)
"""
function init_model_parameters(ind_param, error_param, model::PoisonModel; covariates=false, kappa=false)::ModelParameters
    if covariates == false
        covariates = Array{FLOAT, 1}(undef, 0)
    end

    if kappa == false
        kappa = Array{FLOAT, 1}(undef, 0)
        ind_param = DynModInput(ind_param, kappa, covariates)
    else
        ind_param = DynModInput(ind_param, kappa, covariates)
    end

    x0::Array{Int32} = Array{Int32, 1}(undef, model.dim)
    model.calc_x0!(x0, ind_param)

    # Struct object 
    return ModelParameters(ind_param, x0, error_param, covariates)
end


"""
    systematic_resampling!(index_sampling, weights, n_samples, u)

Calculate indices from systematic resampling of non-normalised weights. 

u must be standard uniform. 
"""
function systematic_resampling!(index_sampling::Array{UInt32, 1}, weights::Array{FLOAT, 1}, n_samples::T1, u::FLOAT) where T1<:Signed

    # For resampling u ~ U(0, 1/N)
    u /= n_samples
    # Step-length when traversing the cdf
    delta_u::FLOAT = 1.0 / n_samples
    # For speed (avoid division)
    sum_weights_inv::FLOAT = 1.0 / sum(weights)

    # Sample from the cdf starting from the random number
    k::UInt32 = 1
    sum_cum::FLOAT = weights[k] * sum_weights_inv
    @inbounds for i in 1:n_samples
        # See which part of the distribution u is at
        while sum_cum < u
            k += 1
            sum_cum += weights[k] * sum_weights_inv
        end
        index_sampling[i] = k
        u += delta_u
    end
end


"""
    init_filter(filter::BootstrapEm, dt; n_particles=1000, rho=0.0) 

Initialise bootstrap particle filter struct for a SDE-model using Euler-Maruyama-stepper. 

By default a non-correlated particle filter is used (rho = 0.0). Step-length for 
Euler-Maruyama should be as large as possible, while still ensuring numerical accuracy. 

# Args
- `filter`: particle filter to use, BootstrapEm() = Bootstrap filter with Euler-Maruyama stepper
- `dt`: Euler-Maruyama step-length 
- `n_particles`: number of particles to use when estimating the likelihood
- `rho`: correlation level. rho ∈ [0.0, 1.0) and if rho = 0.0 the particles are uncorrelated. 
"""
function init_filter(filter::BootstrapEm,
                     dt;
                     n_particles=1000,
                     rho=0.0) 
    
    # Ensure correct type in calculations 
    dt = convert(FLOAT, dt)
    rho = convert(FLOAT, rho)
    filter_opt = BootstrapFilterEm(dt, n_particles, rho)
    return filter_opt
end
"""
    init_filter(filter::ModDiffusion, dt; n_particles=1000, rho=0.0) 

Initialise modified diffusion bridge filter for a SDE-model. 

By default a non-correlated particle filter is used (rho = 0.0). Step-length for the bridge
should be as large as possible, while still ensuring numerical accuracy. 

# Args
- `filter`: particle filter to use, ModDiffusion() = Modified diffusion bridge for SDE-models 
- `dt`: Modified diffusion bridge step-length 
- `n_particles`: number of particles to use when estimating the likelihood
- `rho`: correlation level. rho ∈ [0.0, 1.0) and if rho = 0.0 the particles are uncorrelated. 
"""
function init_filter(filter::ModDiffusion,
                     dt::FLOAT;
                     n_particles=1000,
                     rho=0.0) 
    
    # Ensure correct type in calculations 
    dt::FLOAT = convert(FLOAT, dt)
    rho::FLOAT = convert(FLOAT, rho)
    filter_opt = ModDiffusionFilter(dt, n_particles, rho)
    return filter_opt
end
"""
    init_filter(filter::BootstrapEm, dt; n_particles=1000, rho=0.0) 

Initialise bootstrap particle filter struct for a PoisonModel-model (tau-leaping) with fixed step-length.

By default a non-correlated particle filter is used (rho = 0.0). Step-length for 
tau-lepaing should be as large as possible, while still ensuring numerical accuracy. 

# Args
- `filter`: particle filter to use, BootstrapPois() = tau-lepaing (Poisson) filter with fixed step-length
- `dt`: tau-leaping step-length 
- `n_particles`: number of particles to use when estimating the likelihood
- `rho`: correlation level. rho ∈ [0.0, 1.0) and if rho = 0.0 the particles are uncorrelated. 
"""
function init_filter(filter::BootstrapPois,
                     dt;
                     n_particles=1000,
                     rho=0.0) 
    
    # Ensure correct type in calculations 
    dt = convert(FLOAT, dt)
    rho = convert(FLOAT, rho)
    filter_opt = BootstrapFilterPois(dt, n_particles, rho)
    return filter_opt
end
"""
    init_filter(filter::BootstrapSsa; n_particles=1000, rho=0.0) 

Initialise bootstrap particle filter struct for a SSA-model (Gillespie) 

Non-correlated particle filter is the only choice (rho = 0.0). 

# Args
- `filter`: particle filter to use, BootstrapSsa() = SSA (Gillespie) filter  
- `n_particles`: number of particles to use when estimating the likelihood
- `rho`: correlation level. rho = 0.0 since the particles cannot be correlated for SSA. 
"""
function init_filter(filter::BootstrapSsa;
                     n_particles::T1=1000,
                     rho::FLOAT=0.0)  where T1<:Signed
    
    # Ensure correct type in calculations 
    filter_opt = BootstrapFilterSsa(n_particles, rho)
    return filter_opt
end
"""
    init_filter(filter::BootstrapExtrand; n_particles=1000)

Initialise bootstrap particle filter struct for a Extrande-model

Non-correlated particle filter is the only choice (rho = 0.0). 

# Args
- `filter`: particle filter to use, BootstrapSsa() = SSA (Gillespie) filter  
- `n_particles`: number of particles to use when estimating the likelihood
"""
function init_filter(filter::BootstrapExtrand;
                     n_particles::T1=1000)  where T1<:Signed
    
    # Ensure correct type in calculations 
    filter_opt = BootstrapFilterExtrand(n_particles, 0.0)
    return filter_opt
end


"""
    change_filter_opt(filter::BootstrapFilterEm, n_particles, rho)

Create a deepcopy of a Bootstrap EM-filter with new number of particles and correlation-level rho.  

Used by pilot-run functions to investigate different particles. 
"""
function change_filter_opt(filter::BootstrapFilterEm, n_particles, rho)

    new_filter = BootstrapFilterEm(    
        filter.delta_t,
        n_particles,
        rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end
"""
    change_filter_opt(filter::ModDiffusionFilter, n_particles::T1, rho::FLOAT) where T1<:Signed

Create a deepcopy of a modified-diffusion bridge filter with new number of particles and correlation-level rho.  
"""
function change_filter_opt(filter::ModDiffusionFilter, n_particles::T1, rho::FLOAT) where T1<:Signed

    new_filter = ModDiffusionFilter(    
        filter.delta_t,
        n_particles,
        rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end
"""
    change_filter_opt(filter::BootstrapFilterPois, n_particles, rho)

Create a deepcopy of a Bootstrap Poisson-filter with new number of particles and correlation-level rho.  
"""
function change_filter_opt(filter::BootstrapFilterPois, n_particles, rho)

    new_filter = BootstrapFilterPois(    
        filter.delta_t,
        n_particles,
        rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end
"""
    change_filter_opt(filter::BootstrapFilterSsa, n_particles)

Create a copy of a Bootstrap SSA-filter with new number of particles. 
"""
function change_filter_opt(filter::BootstrapFilterSsa, n_particles, rho)

    new_filter = BootstrapFilterSsa(    
        n_particles,
        filter.rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end
"""
    change_filter_opt(filter::BootstrapFilterExtrand, n_particles)

Create a copy of a Bootstrap Extrande-filter with new number of particles. 
"""
function change_filter_opt(filter::BootstrapFilterExtrand, n_particles, rho)

    new_filter = BootstrapFilterExtrand(    
        n_particles,
        filter.rho)
    new_filter = deepcopy(new_filter)

    return new_filter
end
