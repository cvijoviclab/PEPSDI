"""
    SsaModel

Propensity vector, stoichiometry matrix and observation functions for a SSA (Gillespie)-models.

Propensity and observation functions all follow a specific arg-format (details in notebook). 
Calc_x0 calculates initial values using the individual parameters. Assumed that initial values 
do not execed Int16 
"""
struct SsaModel{F1<:Function, 
                F2<:Function, 
                F3<:Function, 
                F4<:Function, 
                T1<:UInt16,
                T2<:Array{<:Int16, 2}}
    calc_h_vec!::F1
    calc_x0!::F2
    calc_obs::F3
    calc_prob_obs::F4
    dim::T1
    dim_obs::T1
    n_reactions::T1
    S_mat::T2
end


"""
    step_direct_method!(x_current, 
                        t_start::T1, 
                        t_end::T1, 
                        model::SsaModel, 
                        h_vec::T2, 
                        p::DynModInput) where {T1<:AbstractFloat, T2<:Array{<:AbstractFloat}}

Propegate x_current from t_start to t_end using Gillespie's direct method. 

The propensity-vector (h_vec) is updated using the current state-value 
x_current and the model parameters p. Outputs t_curr (current time)
and number of iterations to step from t_start to t_end. 
"""
function step_direct_method!(x_current, 
                             t_start::T1, 
                             t_end::T1, 
                             model::SsaModel, 
                             h_vec::T2, 
                             p::DynModInput) where {T1<:AbstractFloat, T2<:Array{<:AbstractFloat}}

    t_curr::FLOAT = t_start
    transition_rate::FLOAT = 0.0
    delta_t::FLOAT = 0.0
    next_reaction::Int64 = 1
    selector::FLOAT = 0.0
        
    iter::Int64 = 0
    while t_curr < t_end && iter < 1e8
        model.calc_h_vec!(x_current, h_vec, p, t_curr)
        transition_rate = sum(h_vec)

        # Expontial random numbers 
        delta_t = -log(rand()) / transition_rate 
        t_curr += delta_t
        
        # Weighted sampling of next reaction 
        selector = transition_rate * rand()
        @inbounds for i in 1:model.n_reactions
            selector -= h_vec[i]
            if selector < 0
                next_reaction = i
                break
            end
        end

        @views x_current .-= model.S_mat[next_reaction, :]
        iter += 1
    end

    return tuple(t_curr, iter)
end


"""
    solve_ssa_model(model::SsaModel, t_vec, p::DynModInput)

Simulate model over t_vec[1] -> t_vec[end] using the Gillespie's direct method. 

Simulate a SSA-model using model-parameters p from t_vec[1] to t_vec[end]
and save the state-values at the time-points in t_vec. 

See also: [`SsaModel`]
"""
function solve_ssa_model(model::SsaModel, t_vec, p::DynModInput)
    n_data_points = length(t_vec)
    n_states = model.dim
    n_data_points = length(t_vec)
    u_vec::Array{Int64, 2} = Array{Int64, 2}(undef, (n_states, n_data_points))

    x_current = Array{UInt16, 1}(undef, n_states)
    model.calc_x0!(x_current, p)
    @views u_vec[:, 1] .= x_current

    h_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, model.n_reactions)
    t_curr::FLOAT = t_vec[1]
    iter::Int64 = 0

    for i in 1:(n_data_points-1)
        t_curr, iter = step_direct_method!(x_current, t_curr, t_vec[i+1], model, h_vec, p)
        @views u_vec[:, i+1] .= x_current
    end

    return t_vec, u_vec
end


"""
    solve_ssa_model_n_times(model::SsaModel, t_vec, p::DynModInput; times_solve=1000)

Use [`solve_ssa_model`] to solve a SSA-model n-times and returns the (0.05, 0.5, 0.95) quantiles. 

See also: [`solve_ssa_model`]
"""
function solve_ssa_model_n_times(model::SsaModel, t_vec, p::DynModInput; times_solve=1000)

    # For storing the result
    n_col_sol = length(t_vec)
    sol_mat::Array{Int64, 2} = Array{Int64, 2}(undef, (model.dim*times_solve, n_col_sol))

    # Solve the SDE:s multiple times using both methods
    for i in 1:times_solve
        x_current = Array{UInt16, 1}(undef, model.dim)
        model.calc_x0!(x_current, p)
        t_vec1, u_mat1 = solve_ssa_model(model, t_vec, p)

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
    simulate_data_ssa(ssa_mod::SsaModel, error_dist, t_vec, p)

Simulate data for a SSA-model corruputed by an error-distribution. 

Simulate data for data-points t_vec, and parameter vector p. The model is simulated 
using the Gillespie's direct method. 
"""
function simulate_data_ssa(ssa_mod::SsaModel, error_dist, t_vec, p)

    # Allocate the solution vector (each row is observeble for a state)
    y_vec = zeros(ssa_mod.dim_obs, length(t_vec))
    y_mod = zeros(ssa_mod.dim_obs)

    x0 = Array{UInt16, 1}(undef, ssa_mod.dim)
    ssa_mod.calc_x0!(x0, p)

    # Case when obs1 is observed at t0
    start_index = 1
    if t_vec[1] == 0
        ssa_mod.calc_obs(y_mod, x0, p, 0.0)
        error_vec = rand(error_dist, ssa_mod.dim_obs)
        y_vec[:, 1] = y_mod + error_vec
        t_vec_use = t_vec
    else
        start_index = 2
        t_vec_use = vcat(0, t_vec) 
    end
    t_vec_use = convert(Array{FLOAT, 1}, t_vec_use)

    t_vec2, u_mat = solve_ssa_model(ssa_mod, t_vec_use, p)

    #  Map simulations to observations 
    i_data = start_index
    for i in 1:length(t_vec)
        t = t_vec[i]
        ssa_mod.calc_obs(y_mod, u_mat[:, i_data], p, t)
        if length(error_dist) == 1
            error_vec = rand(error_dist, ssa_mod.dim_obs)
        else
            error_vec = rand(error_dist)
        end
        y_vec[:, i] = y_mod + error_vec
        i_data += 1
    end
    
    t_vec = convert(Array{Float64}, t_vec)
    return t_vec, y_vec
end

