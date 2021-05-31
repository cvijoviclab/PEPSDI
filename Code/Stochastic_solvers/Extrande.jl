"""
    ExtrandModel

Propensity vector, maximum propensity-vector, stoichiometry matrix and observation functions for a Extrand-models.

Propensity and observation functions all follow a specific arg-format (details in notebook). 
Propensity max-function computes the maximum potential propensity between t -> t + tau. This 
is required by the extrande-algorithm to properly handle time-variant propensities 
(see: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004923)
Calc_x0 calculates initial values using the individual parameters. Assumed that initial values 
do not execed Int16 
"""
struct ExtrandModel{F1<:Function, 
                    F2<:Function, 
                    F3<:Function, 
                    F4<:Function, 
                    F5<:Function, 
                    T1<:Signed,
                    T2<:Array{<:Int16, 2}}
    calc_h_vec!::F1
    calc_max_h_vec!::F2
    calc_x0!::F3
    calc_obs::F4
    calc_prob_obs::F5
    dim::T1
    dim_obs::T1
    n_reactions::T1
    S_mat::T2
end


"""
    log_rand(rng::MersenneTwister)::FLOAT

Fast computations of the logarithm for a random uniform number. 
"""
@inline function log_rand(rng::MersenneTwister)::FLOAT
    return log(rand(rng))
end


"""
    step_extrande_method!(x_current, 
                          t_start::T1, 
                          t_end::T1, 
                          model::ExtrandModel, 
                          h_vec::T2, 
                          p::DynModInput, 
                          rng::MersenneTwister)::FLOAT where {T1<:AbstractFloat, T2<:Array{<:AbstractFloat}}

Propegate x_current from t_start to t_end using the extrande-method for model models with time-invariant propensities. 

The propensity-vector (h_vec) is updated using the current state-value 
x_current and the model parameters p. Outputs t_curr (current time).
"""
function step_extrande_method!(x_current, 
                               t_start::T1, 
                               t_end::T1, 
                               model::ExtrandModel, 
                               h_vec::T2, 
                               p::DynModInput, 
                               rng::MersenneTwister)::FLOAT where {T1<:AbstractFloat, T2<:Array{<:AbstractFloat}}

    t_curr::FLOAT = t_start
    transition_rate::FLOAT = 0.0
    delta_t::FLOAT = 0.0
    next_reaction::Int64 = 1
    selector::FLOAT = 0.0

    B::FLOAT = 0.0
    L::FLOAT = 0.0
    tau::FLOAT = 0.0
    rand_bound::FLOAT = 0.0
        
    iter::Int64 = 0
    while t_curr < t_end && iter < 1e8

        model.calc_max_h_vec!(x_current, h_vec, p, t_start, t_end)
        
        # Set the maximum limits 
        B = sum(h_vec)
        L = t_end - t_curr

        # Check if should fire reaction channels 
        tau = -log_rand(rng) / B
        t_curr += tau

        if tau < L
            
            model.calc_h_vec!(x_current, h_vec, p, t_curr)
            transition_rate = sum(h_vec)
            
            rand_bound = B * rand(rng)
            if transition_rate > rand_bound
                selector = 0.0
                @inbounds for i in 1:model.n_reactions
                    selector += h_vec[i]
                    if selector > rand_bound
                        next_reaction = i
                        break
                    end
                end
                
                @views x_current .-= model.S_mat[next_reaction, :]
            end
        end

    end
        
    return t_curr
end


"""
    solve_extrande_model(model::ExtrandModel, t_vec, p::DynModInput; rng=MersenneTwister())

Simulate model over t_vec[1] -> t_vec[end] using the extrande-method 

Simulate an Extrande-model using model-parameters p from t_vec[1] to t_vec[end]
and save the state-values at the time-points in t_vec. Can provide rng for 
reproducibillity.  

See also: [`ExtrandModel`]
"""
function solve_extrande_model(model::ExtrandModel, t_vec, p::DynModInput; rng=MersenneTwister())
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
        t_curr = step_extrande_method!(x_current, t_curr, t_vec[i+1], model, h_vec, p, rng)
        @views u_vec[:, i+1] .= x_current
    end

    return t_vec, u_vec
end


"""
    solve_extrande_model_n_times(model::ExtrandModel, t_vec, p::DynModInput; times_solve=5000)

Use [`solve_extrande_model`] to solve an Extrande-model n-times and returns the (0.05, 0.5, 0.95) quantiles. 

See also: [`solve_extrande_model`]
"""
function solve_extrande_model_n_times(model::ExtrandModel, t_vec, p::DynModInput; times_solve=5000)

    # For storing the result
    n_col_sol = length(t_vec)
    sol_mat::Array{Int64, 2} = Array{Int64, 2}(undef, (model.dim*times_solve, n_col_sol))

    # Solve the SDE:s multiple times using both methods
    for i in 1:times_solve
        x_current = Array{UInt16, 1}(undef, model.dim)
        model.calc_x0!(x_current, p)
        t_vec1, u_mat1 = solve_extrande_model(model, t_vec, p)

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
    simulate_data_extrande(model::ExtrandModel, error_dist, t_vec, p)

Simulate data for a Extrande-model corruputed by an error-distribution. 

Simulate data for data-points t_vec, and parameter vector p. The model is simulated 
using the Extrande-method. 
"""
function simulate_data_extrande(model::ExtrandModel, error_dist, t_vec, p)

    # Allocate the solution vector (each row is observeble for a state)
    y_vec = zeros(model.dim_obs, length(t_vec))
    y_mod = zeros(model.dim_obs)

    x0 = Array{UInt16, 1}(undef, model.dim)
    model.calc_x0!(x0, p)

    seed = rand(DiscreteUniform(1, 10000))
    rng = MersenneTwister(seed)

    # Case when obs1 is observed at t0
    start_index = 1
    if t_vec[1] == 0
        model.calc_obs(y_mod, x0, p, 0.0)
        error_vec = rand(error_dist, model.dim_obs)
        y_vec[:, 1] = y_mod + error_vec
        t_vec_use = t_vec
    else
        start_index = 2
        t_vec_use = vcat(0, t_vec) 
    end
    t_vec_use = convert(Array{FLOAT, 1}, t_vec_use)

    t_vec2, u_mat = solve_extrande_model(model, t_vec_use, p, rng=rng)

    #  Map simulations to observations 
    i_data = start_index
    for i in 1:length(t_vec)
        t = t_vec[i]
        model.calc_obs(y_mod, u_mat[:, i_data], p, t)
        if length(error_dist) == 1
            error_vec = rand(error_dist, model.dim_obs)
        else
            error_vec = rand(error_dist)
        end
        y_vec[:, i] = y_mod + error_vec
        i_data += 1
    end
    
    t_vec = convert(Array{Float64}, t_vec)
    return t_vec, y_vec
end
