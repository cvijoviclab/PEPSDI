"""
    PoisonModel

Propensity vector, stoichiometry matrix and observation functions for a Poisson-model.

Propensity and observation functions all follow a specific arg-format
(details in notebook). Calc_x0 calculates initial values using the 
individual parameters. Assumed that initial values do not execed Int16 
"""
struct PoisonModel{F1<:Function, 
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
    map_to_zero_poison!(x_data, n_elements::T1) where T1<:Integer

Convert negative element in x_data to 0 if x becomes negative in a time-leap.  
"""
@inline function map_to_zero_poison!(x_data, n_elements::T1) where T1<:Integer

    @inbounds @simd for i in 1:n_elements
        if x_data[i] < 0 
            x_data[i] = 0
        end
    end
end


# Constants for computing Poisson-pdf function 
const PI2 = 6.283185307179586476925286
const S0 = 0.083333333333333333333 # 1/12
const S1 = 0.00277777777777777777778 # 1/360 
const S2 = 0.00079365079365079365079365 # 1/1260 
const S3 = 0.000595238095238095238095238 # 1/1680 
const S4 = 0.0008417508417508417508417508 # 1/1188 
const sfe = [0, 0.081061466795327258219670264,
    0.041340695955409294093822081, 0.0276779256849983391487892927,
    0.020790672103765093111522771, 0.0166446911898211921631948653,
    0.013876128823070747998745727, 0.0118967099458917700950557241,
    0.010411265261972096497478567, 0.0092554621827127329177286366,
    0.008330563433362871256469318, 0.0075736754879518407949720242,
    0.006942840107209529865664152, 0.0064089941880042070684396310,
    0.005951370112758847735624416, 0.0055547335519628013710386899]


"""
    stirlerr(n::Int64)::Float64

Stirling error-approximation for factorial calculation: stirlerr(n) = log(n!) - log( sqrt(2*pi*n)*(n/e)^n )

Part of evaluating Poisson-pdf. 
"""
function stirlerr(n::Int64)::Float64
    
    if n<16 
        return sfe[n]
    end

    nn::Float64 = convert(Float64, n)
    nn = nn * nn

    if n > 500 
        return (S0-S1/nn)/n 
    end

    if n > 80
        return (S0-(S1-S2/nn)/nn)/n
    end

    if n > 35
        return (S0-(S1-(S2-S3/nn)/nn)/nn)/n
    end

    return (S0-(S1-(S2-(S3-S4/nn)/nn)/nn)/nn)/n
end


"""
    bd0(x::Int64, np::Float64)::Float64

Deviance term bd0(x,np) = x log(x/np) + np - x when computing Poisson-pdf 
"""
function bd0(x::Int64, np::Float64)::Float64

    local ej::Float64 
    local s::Float64 
    local s1::Float64
    local v::Float64

    local j::Int64 = 1

    if abs(x-np) < 0.1*(x+np)
        
        s = (x-np)*(x-np)/(x+np)
        v = (x-np)/(x+np)
        ej = 2*x*v

        while true 
            ej *= v*v
            s1 = s+ej/(2*j+1)
            if s1==s 
                return s1
            end
            s = s1;
            j += 1
        end
    end

    return x*log(x/np)+np-x;
end


"""
    pdf_poisson(x::Int64, λ::Float64)::Float64

Compute pdf for integer x for a Poisson(λ) distribution. 

A Julia implementation of the R implementation: https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf
"""
function pdf_poisson(x::Int64, λ::Float64)::Float64

    if λ == 0 
        return (x==0) ? 1.0 : 0.0
    end

    if x == 0 
        return exp(-λ)
    end

    return exp(-stirlerr(x) - bd0(x,λ))/sqrt(PI2*x)

end


"""
    poisson_rand_small(lambda::FLOAT, u::FLOAT)::Int64

Generate a poisson-random number for small λ using a single uniform number u. 

Calculates which integer of the CDF the random number u corresponds to and returns an integer. 
"""
@inline function poisson_rand_small(lambda::FLOAT, u::FLOAT)::Int64
    
    # Quantile based method 
    log_lambda::FLOAT = log(lambda)
    log_f::FLOAT = -lambda
    F::FLOAT = exp(log_f)
    
    k::Int64 = 1
    while F < u

        log_f += log_lambda - log(k)
        F += exp(log_f)
        k += 1
    end

    return k - 1
end

"""
    calc_stand_normal_quantile(u::FLOAT)::FLOAT

Calculate infervse CDF for standard normal distribution. 
"""
@inline function calc_stand_normal_quantile(u::FLOAT)::FLOAT
    const_val::FLOAT = 1.4142135623730

    return erfinv(2.0*u - 1.0) * const_val

end


"""
    poisson_rand_large(lambda::FLOAT, u::FLOAT)::Int64

Generate a poisson-random number for large λ using a single uniform number u. 

Calculates which integer of the CDF the random number u corresponds to and returns corresponding 
interger. To start the CDF-based search a normal-approximation of the Poisson-distribution is employed. 
"""
@inline function poisson_rand_large(lambda::FLOAT, u::FLOAT)::Int64

    # Calculate the gaussian-quantile for u 
    q::FLOAT = calc_stand_normal_quantile(u)

    # Approximate poisson quantile 
    sqrt_lambda::FLOAT = sqrt(lambda)
    log_lambda::FLOAT = log(lambda)
    bar_q::FLOAT = q * sqrt_lambda + (q^2 - 1) / 6 + (q - q^3) / (72 * sqrt_lambda) + lambda
    bar_q_int::Int64 = convert(Int64, max(0.0, round(bar_q)))

    # Calculate F 
    F = cdf(Poisson(lambda), bar_q_int)

    # Large case
    if F < u 

        while F < u
            F += pdf_poisson(bar_q_int + 1, lambda)
            bar_q_int += 1
        end

        return bar_q_int - 1
    end

    if F > u 
        while F > u && bar_q_int >= 0
            F -= pdf_poisson(bar_q_int, lambda)
            bar_q_int -= 1
        end

        return bar_q_int + 1
    end
end


"""
    poisson_rand(lambda::FLOAT, u::FLOAT)::Int64

Generate a poisson-random number for a λ using a single uniform number u. 

Uses [`poisson_rand_large`](@ref) if λ > 4.0, else uses [`poisson_rand_small`](@ref)
based on benchmarks. 

See also : [`poisson_rand_large`, `poisson_rand_small`](@ref)
"""
function poisson_rand(lambda::FLOAT, u::FLOAT)::Int64
    rand_num::Int64 = lambda > 4.0 ? poisson_rand_large(lambda, u) : poisson_rand_small(lambda, u)

    return rand_num
end


"""
    perform_poisson_step!(x_current, 
                          model::PoisonModel,
                          h_vec::Array{FLOAT, 1}, 
                          delta_t::FLOAT, 
                          p::DynModInput,
                          t::FLOAT, 
                          u_vec::Array{FLOAT, 1})

Perform a Poisson-leap to propegate x_current from t to t + delta_t. 

The propensity-vector (h_vec) is updated using the current state-value 
x_current and the model parameters p. u_vec contains random-numbers used to 
generate the Poisson-variables. 
"""
function perform_poisson_step!(x_current, 
                               model::PoisonModel,
                               h_vec::Array{FLOAT, 1}, 
                               delta_t::FLOAT, 
                               p::DynModInput,
                               t::FLOAT, 
                               u_vec)

    # Calculate h-vec and multiply with delta-t to get exepcted number of reactions 
    model.calc_h_vec!(x_current, h_vec, p, t)
    h_vec .*= delta_t

    # Update x-current 
    @inbounds for i in 1:model.n_reactions
        @views x_current .-= (model.S_mat[i, :] * poisson_rand(h_vec[i], u_vec[i]))
    end

end


"""
    solve_poisson_model(model::PoisonModel, time_span, p, dt::FLOAT; x_current = false)

Simulate model over time-span using the Poisson-leap method with step-length dt. 

Simulate a Poisson-model using model-parameters p from time_span[1] to time_span[2]
using a step-length dt. Returns time-vector and matrix with state-values. 

If x_current = false calculates initial values using model.calc_x0!. Else use 
values provided in x_current. 

See also: [`PoisonModel`]
"""
function solve_poisson_model(model::PoisonModel, time_span, p, dt::FLOAT; x_current = false)

    n_states = model.dim

    # Based on dt decide intervall for solving 
    t_vec = range(time_span[1], time_span[end], step=dt)
    n_data_points = length(t_vec)
    x_out::Array{Int64, 2} = Array{Int64, 2}(undef, (n_states, n_data_points))

    # Calculate initial value 
    if x_current == false
        x_current = Array{Int32, 1}(undef, n_states)
        model.calc_x0!(x_current, p)
    end
    @views x_out[:, 1] .= x_current

    h_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, model.n_reactions)

    # Random numbers for propegation 
    rand_mat = rand(model.n_reactions, n_data_points)

    # Propegate 
    for i in 1:n_data_points-1
        rand_vec = @view rand_mat[:, i]
        perform_poisson_step!(x_current, model, h_vec, dt, p, t_vec[i], rand_vec)
        map_to_zero_poison!(x_current, model.dim)
        x_out[:, i+1] .= x_current

    end

    return t_vec, x_out
end


"""
    simulate_data_poisson(model::PoisonModel, error_dist, t_vec, p; dt=1e-5)

Simulate data for a Poisson-model corruputed by an error-distribution. 

Simulate data for data-points t_vec, and parameter vector p. The model is simulated 
using the Poisson-method with a step-length dt.

See also: [`PoisonModel`], [`solve_poisson_model`](@ref)
"""
function simulate_data_poisson(model::PoisonModel, error_dist, t_vec, p; dt=1e-5)

    # Allocate the solution vector (each row is observeble for a state)
    y_vec = zeros(model.dim_obs, length(t_vec))
    y_mod = zeros(model.dim_obs)
    n_states = model.dim

    x_curr = Array{Int32, 1}(undef, n_states)
    model.calc_x0!(x_curr, p)

    # Case when obs1 is observed at t0
    start_index = 1
    if t_vec[1] == 0
        model.calc_obs(y_mod, x_curr, p, 0.0)
        error_vec = rand(error_dist, model.dim_obs)
        y_vec[:, 1] = y_mod + error_vec
        start_index = 2
    end

    # Generate simulations
    for i in start_index:length(t_vec)
        # Handle special case when t = 0.0 is not observed
        if i == 1
            t_int = (0.0, t_vec[1])
        else
            t_int = (t_vec[i-1], t_vec[i])
        end

        # Propegate poisson-system system
        t, u = solve_poisson_model(model, t_int, p, dt, x_current=x_curr)
        # Update y-vector
        x_curr .= u[:, end]
        model.calc_obs(y_mod, x_curr, p, t)
        if length(error_dist) == 1
            error_vec = rand(error_dist, model.dim_obs)
        else
            error_vec = rand(error_dist)
        end
        y_vec[:, i] = y_mod + error_vec
    end

    t_vec = convert(Array{Float64}, t_vec)
    return t_vec, y_vec
end
