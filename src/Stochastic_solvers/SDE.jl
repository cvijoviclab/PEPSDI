"""
    SdeModel

Dimensions and drift, diffusion, and observation functions for a SDE-models.

Drift, diffusion, and observation functions all follow a specific arg-format
(see notebook). Calc_x0 calculates initial values using the individual parameters 
P_mat can be provided if the observation model is on the format y = P*X + ε, ε ~ N(0, σ^2) 
"""
struct SdeModel{F1<:Function,
                F2<:Function,   
                F3<:Function, 
                F4<:Function, 
                F5<:Function, 
                T1<:Signed, 
                T2<:SArray,
                T3<:SArray}
            
    calc_alpha::F1
    calc_beta::F2
    calc_x0!::F3
    dim::T1
    calc_obs::F4
    calc_prob_obs::F5
    dim_obs::T1
    P_mat::T2
    P_mat_trans::T3
end


"""
    init_sde_model(calc_alpha, calc_beta, calc_x0, calc_obs, calc_prob_obs, dim_mod::T1, dim_obs::T1)::SdeModel where T1<:Signed

Initialise a sde-model struct for SDE-based inference. 

Important that functions calc_alpha, calc_beta, calc_x0, calc_obs, calc_prob_obs all follow a specific format, 
for details see notebook. Option where P-matrix is not provided and thus assumed that observation model 
is not on the format y = P*X + ε, ε ~ N(0, σ^2).

# Args
- `calc_alpha`: function for calculating the drift-vector (see notebook for format)
- `calc_beta`: function for calculating the drift-vector (see notebook for format)
- `calc_x0`: function for calculating initial values (see notebook for format)
- `calc_obs`: function for calculating model observations y (see notebook for format)
- `calc_prob`: function for calculating probability to observe observation y (see notebook for format)
- `dim_mod`: model dimension (number of stochastic differential equations)
- `dim_obs`: dimension of observation model y (length of the vector y)
"""
function init_sde_model(calc_alpha, 
                        calc_beta, 
                        calc_x0, 
                        calc_obs, 
                        calc_prob_obs, 
                        dim_mod::T1, 
                        dim_obs::T1)::SdeModel where T1<:Signed
    
    P_mat = SMatrix{1, 1, FLOAT}(0.0)
    sde_mod::SdeModel = SdeModel(calc_alpha,
                                 calc_beta, 
                                 calc_x0,
                                 dim_mod,
                                 calc_obs,
                                 calc_prob_obs,
                                 dim_obs, 
                                 P_mat, 
                                 P_mat)

    return sde_mod
end
"""
    init_sde_model(calc_alpha, calc_beta, calc_x0, calc_obs, calc_prob_obs, dim_mod::T1, dim_obs::T1, P_mat::T2)::SdeModel where {T1<:Signed, T2<:Array{Int64}}

Option where P-matrix is provided and thus assumed that observation model 
is not on the format y = P*X + ε, ε ~ N(0, σ^2). This makes a SDE-model compatible with 
the modified-diffusion bridge filter. 
"""
function init_sde_model(calc_alpha, 
                        calc_beta, 
                        calc_x0, 
                        calc_obs, 
                        calc_prob_obs, 
                        dim_mod::T1, 
                        dim_obs::T1, 
                        P_mat::T2)::SdeModel where {T1<:Signed, T2<:Array{Int64}}
                                                        
    if typeof(P_mat) <: Array{<:Signed, 1}
        n_col = 1
        n_row = size(P_mat)[1]
        P_mat_use_tmp = Array{FLOAT, 2}(undef, (n_row, n_col))
        P_mat_use_tmp .= P_mat
    else
        P_mat_use_tmp = convert(Array{FLOAT, 2}, P_mat)
    end
    dim1, dim2 = size(P_mat_use_tmp)
    P_mat_use = convert(SMatrix{dim_mod, dim2, FLOAT}, P_mat_use_tmp)
    P_mat_use_transpose = convert(SMatrix{dim2, dim_mod, FLOAT}, P_mat_use_tmp')

    sde_mod::SdeModel = SdeModel(calc_alpha,
                                 calc_beta, 
                                 calc_x0,
                                 dim_mod,
                                 calc_obs,
                                 calc_prob_obs,
                                 dim_obs, 
                                 P_mat_use, 
                                 P_mat_use_transpose)

    return sde_mod
end


"""
    EmSolverObj

Arrays and scalars for efficiently solving SDE-systems using Euler-Maruyama.

When performing EM-steps the alpha_vec and beta_mat are not reallocated. 
"""
struct EmSolverObj{T1<:Array{<:AbstractFloat, 1}, 
                   T2<:Array{<:AbstractFloat, 2}, 
                   T3<:AbstractFloat}
    alpha_vec::T1
    beta_mat::T2
    dt::T3
    sqrt_dt::T3
end


"""
    create_dir_for_path(path::String)

Create if not existing (nested) directories for a file-path or directory-path. 
"""
function create_dir_for_path(path::String)
    i_dir = 0
    for i in 1:length(path)
        index = length(path) - (i - 1)
        if path[index] == '/'
            i_dir = index
            break
        end
    end
    mkpath(path[1:i_dir])
end


"""
    calc_cholesky!(X::Array{Float64, 2}, dim_X::T1) where T1<:Signed

Calculate lower triangular cholesky decomposition of X by overwriting X. 
"""
@inline function calc_cholesky!(X::Array{Float64, 2}, dim_X::T1) where T1<:Signed
    
    @inbounds for j in 1:dim_X
        @inbounds for i in 1:(j-1)
            # Calculating inner sum 
            inner_sum1::Float64 = 0.0
            @inbounds for k in 1:(i-1)
                inner_sum1 += X[k, i] * X[k, j]
            end
            X[i, j] = (X[i, j] - inner_sum1) / X[i, i] 
        end
        inner_sum2::Float32 = 0.0
        @inbounds for k in 1:(j-1)
            inner_sum2 += X[k, j]*X[k, j]
        end
        X[j, j] = sqrt(X[j, j] - inner_sum2)

    end

    # Zero upper-diagonal and make lower diagonal matrix 
    @inbounds for i in 1:dim_X
        @inbounds for j in 1:(i - 1)
            X[i, j] = X[j, i]
            X[j, i] = 0.0
        end
    end
end
@inline function calc_cholesky!(X::T1, dim_X::T2) where {T1<:MArray, T2<:Signed}
    
    @inbounds for j in 1:dim_X
        @inbounds for i in 1:(j-1)
            # Calculating inner sum 
            inner_sum1::Float64 = 0.0
            @inbounds for k in 1:(i-1)
                inner_sum1 += X[k, i] * X[k, j]
            end
            X[i, j] = (X[i, j] - inner_sum1) / X[i, i]
        end
        inner_sum2::Float64 = 0.0
        @inbounds for k in 1:(j-1)
            inner_sum2 += X[k, j]*X[k, j]
        end
        X[j, j] = sqrt(X[j, j] - inner_sum2)
    end

    # Zero upper-diagonal and make lower diagonal matrix 
    @inbounds for i in 1:dim_X
        @inbounds for j in 1:(i - 1)
            X[i, j] = X[j, i]
            X[j, i] = 0.0
        end
    end
end


"""
    map_to_zero!(x_data, n_elements::Int64)

Convert negative element in x_data to 0 if x becomes negative in a time-leap.
"""
function map_to_zero!(x_data, n_elements::Int64)

    @inbounds @simd for i in 1:n_elements
        if x_data[i] < 0 
            x_data[i] = 0.1
        end
    end
end


"""
    perform_step_em!(u_new, u_old, c, rand_vec, sde_mod, em_obj; t=1.0)

Perform a step using Euler-Maruyama algorithm and store result in u_new.

The output is written to u_new to avoid allocating multiple copies. 
# Args
- `u_new`: (output) updated state-vector.
- `u_old`: state vector a time t.
- `c`: parameter vector (or struct).
- `rand_vec`: vector of random N(0, 1) values of dim u for propegation.
- `sde_mod`: model dimension and sde-model function.
- `em_obj`: arrays and scalars for time-stepping.
- `t`: time-value for u_old.

See also: [`SdeMod`, `EmSolverObj`](@ref)
"""
@inline function perform_step_em!(u_new, 
                                  u_old, 
                                  c, 
                                  rand_vec, 
                                  sde_mod::SdeModel, 
                                  em_obj::EmSolverObj;
                                  t=1.0)
                                  
    # Update drift and diffusion (memory efficiently)
    sde_mod.calc_alpha(em_obj.alpha_vec, u_old, c, t)
    sde_mod.calc_beta(em_obj.beta_mat, u_old, c, t)

    @views u_new .= (u_old .+ em_obj.alpha_vec .* em_obj.dt +
        cholesky(em_obj.beta_mat).L*rand_vec .*em_obj.sqrt_dt)
end


"""
    solve_sde_em(sde_mod, time_span, u0, c, dt; stand_step=true)

Solve sde-system using Euler-Maruyama over specific time-span. 

Solve SDE-system sde_mod using EM-stepper using intitial values u0 and 
parameters c and step-size dt. Returns a time-vector and matrix with 
state-values for each time-step. 

If stand_step=false results in boostrap-filter 
propegation function being used to step. This for testing. 

See also: [`SdeMod`]
"""
function solve_sde_em(sde_mod, time_span, u0, c, dt; stand_step=true)

    norm_dist = Normal()
    em_obj = EmSolverObj(zeros(sde_mod.dim), zeros(sde_mod.dim, sde_mod.dim),
                        dt, sqrt(dt))

    # Allocate the time-span vector, solution vector and random numbers
    t_vec = range(time_span[1], time_span[2], step=dt)
    n_data_points = length(t_vec)
    u_vec = zeros(sde_mod.dim, n_data_points)
    u_vec[:, 1] = u0
    rand_mat = rand(norm_dist, (sde_mod.dim, n_data_points))

    n_states = 1:sde_mod.dim
    if stand_step
        for i in 1:n_data_points-1
            u_prop = @view u_vec[:, i+1]
            perform_step_em!(u_prop,
                SubArray(u_vec, (n_states, i)), c,
                SubArray(rand_mat, (n_states, i)),
                sde_mod, em_obj; t = t_vec[i])

            map_to_zero!(u_prop, sde_mod.dim)
        end
    else
        n_particles = 1
        for i in 1:n_data_points-1
            t_step_info = (t_vec[i], t_vec[i+1], t_vec[i+1]-t_vec[i])
            x = u_vec[n_states, i]
            u = SubArray(u_vec, (n_states, i))
            propegate_em!(x, c, t_step_info, sde_mod, n_particles, u)
            u_vec[n_states, i+1] = x
        end
    end

    return t_vec, u_vec
end


function solve_sde_model_n_times(model, time_span, p, dt; times_solve=1000)

    # For storing the result
    t_vec = range(time_span[1], time_span[2], step=dt)
    n_col_sol = length(t_vec)
    n_col_sol = length(t_vec)
    sol_mat::Array{Float64, 2} = Array{Float64, 2}(undef, (model.dim*times_solve, n_col_sol))

    # Solve the SDE:s multiple times 
    for i in 1:times_solve
        x_current = Array{Float64, 1}(undef, model.dim)
        model.calc_x0!(x_current, p)
        t_vec1, u_mat1 = solve_sde_em(model, time_span, x_current, p, dt)

        i_low = model.dim * i - (model.dim - 1)
        i_upp = model.dim * i
        @views sol_mat[i_low:i_upp, :] = u_mat1
    end

    # Aggregate the median and mean values, note the first
    # three columns are the first state, the second pair is the
    # second state etc.
    quant = zeros(model.dim*3, n_col_sol)
    for i in 1:model.dim

        i_rows = i:model.dim:(times_solve*model.dim)
        start_index = (i - 1) *3 + 1

        quant[start_index, :] = median(sol_mat[i_rows, :], dims=1)

        # Fixing the quantiles
        for j in 1:n_col_sol
            i_low = start_index+1
            i_upp = start_index+2
            quant[i_low:i_upp, j] = quantile(sol_mat[i_rows, j], [0.05, 0.95])
        end
    end

    return t_vec, quant 

end


"""
    simulate_data_sde(sde_mod, error_dist, t_vec, c, x0; dt=1e-5)

Simulate data for a sde-model corruputed by an error-distribution. 

Simulate data for data-points t_vec, and parameter vector c and starting 
values x0. SDE-system is solved using EM-stepper with time-step dt. Change dt
if looking at very small time-interval (t_vec[1], t_vec[end]). 

See also: [`SdeModel`], [`solve_sde_em`](@ref)
"""
function simulate_data_sde(sde_mod, error_dist, t_vec, c, x0; dt=1e-5)

    # Allocate the solution vector (each row is observeble for a state)
    y_vec = zeros(sde_mod.dim_obs, length(t_vec))
    y_mod = zeros(sde_mod.dim_obs)

    # Case when obs1 is observed at t0
    start_index = 1
    if t_vec[1] == 0
        sde_mod.calc_obs(y_mod, x0, c, 0.0)
        error_vec = rand(error_dist, sde_mod.dim_obs)
        y_vec[:, 1] = y_mod + error_vec
        start_index = 2
    end

    # Generate simulations
    x_curr = x0
    for i in start_index:length(t_vec)
        # Handle special case when t = 0.0 is not observed
        if i == 1
            t_int = (0.0, t_vec[1])
        else
            t_int = (t_vec[i-1], t_vec[i])
        end

        # Propegate sde-system system
        t, u = solve_sde_em(sde_mod, t_int, x_curr, c, dt)
        # Update y-vector
        x_curr .= u[:, end]
        sde_mod.calc_obs(y_mod, x_curr, c, t)
        if length(error_dist) == 1
            error_vec = rand(error_dist, sde_mod.dim_obs)
        else
            error_vec = rand(error_dist)
        end
        y_vec[:, i] = y_mod + error_vec
    end

    t_vec = convert(Array{Float64}, t_vec)
    return t_vec, y_vec
end

