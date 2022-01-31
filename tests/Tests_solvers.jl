#=
    Tests for the stochastic simulators. Currently the file checks that 
    all simulator functions can be run without crashing. In the future tests 
    will be added to assure that all simulators are accurately implmented. 
=#


using Printf
using Random
using LinearAlgebra
using Distributions
using Plots 
using DifferentialEquations
using Catalyst 
using Revise
tmp = push!(LOAD_PATH, pwd() * "/src")
using PEPSDI 


include(pwd() * "/Paper/Models/Ornstein.jl")


function sir_h_vec!(u, h_vec, p, t)
    c = p.c
    h_vec[1] = exp(log(c[1]) + log(u[1]) + log(u[2]))
    h_vec[2] = exp(log(c[2]) + log(u[2]))
    
end
function sir_h_vec_max!(u, h_vec, p, t_start, t_end)
    c = p.c
    h_vec[1] = exp(log(c[1]) + log(u[1]) + log(u[2]))
    h_vec[2] = exp(log(c[2]) + log(u[2]))
    
end
function sir_x0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 999
    x0[2] = 1
    x0[3] = 0
end
function sir_x0!(x0::T1, p) where T1<:Array{<:Int32, 1}
    x0[1] = 999
    x0[2] = 1
    x0[3] = 0
end
function sir_obs(y, u, p, t)
    y[1] = u[3]
end
sir_model = @reaction_network begin
    β, S + I --> 2I
    ν, I --> R
end β ν



"""
    run_solver_functions()

Tests if all simulator functions can be run by running the simulate data function 
    
The simulator function utlises all functions for a stochastic solver. 
"""

function run_solver_functions()

    @printf("Testing if all functions can be run\n")

    @printf("Testing extrande ...")
   
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model = ExtrandModel(clock_h_vec!, clock_h_vec_max!, clock_x0!, clock_obs, clock_prob_obs, 2, 1, 4, S_left - S_right)
    error_dist = Normal(0, 2.0)
    p = DynModInput([4.0, 25.0, 2.0, 1.0], [0.0], [0.0])
    t_vec, y_vec = simulate_data_extrande(my_model, error_dist, 0.5:0.75:48, p)

    @printf("done\n\n")

    
    @printf("Testing SSA (Gillespie) ...")

    S_left = convert(Array{Int16, 2}, [0 0; 0 0; 1 0; 0 1; 1 1])
    S_right = convert(Array{Int16, 2}, [1 0; 0 1; 0 0; 0 0; 0 2])
    my_model = SsaModel(auto_h_vec!, 
                        auto_x0!, 
                        auto_obs, 
                        auto_prob_obs, 
                        UInt16(2), UInt16(1), UInt16(5), S_left - S_right)

    error_dist = Normal(0, 0.2)
    p = DynModInput([10, 0.1, 0.1, 0.7, 0.008], [0.0], [0.0])
    t_vec, y_vec = simulate_data_ssa(my_model, error_dist, 1.0:2:100, p)

    @printf("done\n\n")


    @printf("Testing SDE ...")
    
    sde_mod = init_sde_model(alpha_ornstein_full, 
                             beta_ornstein_full, 
                             calc_x0_ornstein!, 
                             ornstein_obs, 
                             prob_ornstein_full,
                             1, 1)

    error_dist = Normal(0.0, 0.3)
    c = exp.([-0.7, 2.3, -0.9])
    p = DynModInput(c, [0.0], [0.0])
    x0 = [0.0]; t_vec = 0.25:0.25:10
    t_vec, y_vec = simulate_data_sde(sde_mod, error_dist, t_vec, p, x0, dt=1e-5)

    @printf("done\n\n")


    @printf("Testing Poisson ...")

    S_left = convert(Array{Int16, 2}, [0 0; 0 0; 1 0; 0 1; 1 1])
    S_right = convert(Array{Int16, 2}, [1 0; 0 1; 0 0; 0 0; 0 2])
    my_model = PoisonModel(auto_h_vec!, 
                           auto_x0!, 
                           auto_obs, 
                           auto_prob_obs, 
                           UInt16(2), UInt16(1), UInt16(5), S_left - S_right)

    error_dist = Normal(0, 0.2)
    p = DynModInput([10, 0.1, 0.1, 0.7, 0.008], [0.0], [0.0])
    t_vec, y_vec = simulate_data_poisson(my_model, error_dist, 1.0:2:100, p, dt=1e-3)

    @printf("done\n\n")

    @printf("Done testing if all solver can run\n")
    @printf("All solvers can run \n")

end


function check_sde_solver(times_run)

    sde_mod = init_sde_model(alpha_ornstein_full, 
                            beta_ornstein_full, 
                            calc_x0_ornstein!, 
                            ornstein_obs, 
                            prob_ornstein_full,
                            1, 1)
    time_span = (0.0, 15.0)
    c = exp.([-0.7, 2.3, -0.9])
    p = DynModInput(c, [0.0], [0.0])
    t_vec, quant = solve_sde_model_n_times(sde_mod, time_span, p, 1e-2; times_solve=times_run)

    i_max = length(0.0:1e-2:15.0)
    data_save = zeros(Float64, i_max, times_run)
    for i in 1:times_run
        sde_prob = SDEProblem(alpha_ornstein_full, beta_ornstein_full_2, [0.0], (0.0,15.0))
        sol = solve(sde_prob, EM(), dt=1e-2, p = p)
        for j in 1:i_max
            data_save[j, i] = sol.u[j][1]
        end
    end

    quant_val = zeros(Float64, 3, i_max)
    quant_val[1, :] .= [median(data_save[i, :]) for i in 1:i_max]
    quant_val[2, :] .= [quantile(data_save[i, :], 0.05) for i in 1:1501]
    quant_val[3, :] .= [quantile(data_save[i, :], 0.95) for i in 1:1501]

    p = plot(t_vec, quant[1, :], legend = :outertopright, color="#E69F00", linewidth=2.0, label = "PEPSDI")
    p = plot!(t_vec, quant_val[1, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label = "SciML")
    p = plot!(t_vec, quant[2, :], color="#E69F00", linewidth=2.0, label=false)
    p = plot!(t_vec, quant_val[2, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label=false)
    p = plot!(t_vec, quant[3, :], color="#E69F00", linewidth=2.0, label=false)
    p = plot!(t_vec, quant_val[3, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label=false)
    title!("OU model 0.05, 0.5, 0.95 quantiles: Euler-Maruyama")

    return p
end


function check_ssa_solver(times_run; return_plot=true)

    S_left = convert(Array{Int16, 2}, [1 1 0; 0 1 0])
    S_right = convert(Array{Int16, 2}, [0 2 0; 0 0 1])
    model = SsaModel(sir_h_vec!, 
                    sir_x0!, 
                    sir_obs, 
                    empty_func, 
                    UInt16(3), UInt16(1), UInt16(2), S_left - S_right)
    t_vec = 0.0:1.0:250.0
    c = [0.1 / 1000, 0.01]
    p = DynModInput(c, [0.0], [0.0])
    t_vec, quant = solve_ssa_model_n_times(model, t_vec, p; times_solve=times_run)
    quant = quant[4:6, :]

    p = [0.1 / 1000, 0.01]
    u0 = [999, 1, 0]
    prob = DiscreteProblem(sir_model, u0, (0.0, 250.0), p)
    jump_prob = JumpProblem(sir_model, prob, Direct())

    i_max = length(t_vec)
    data_save = zeros(Float64, i_max, times_run)
    for i in 1:times_run
        prob = DiscreteProblem(sir_model, u0, (0.0, 250.0), p)
        jump_prob = JumpProblem(sir_model, prob, Direct())
        sol = solve(jump_prob, SSAStepper())
        for j in 1:i_max
            data_save[j, i] = sol(t_vec[j])[2]
        end
    end

    quant_val = zeros(Float64, 3, i_max)
    quant_val[1, :] .= [median(data_save[i, :]) for i in 1:i_max]
    quant_val[2, :] .= [quantile(data_save[i, :], 0.05) for i in 1:i_max]
    quant_val[3, :] .= [quantile(data_save[i, :], 0.95) for i in 1:i_max]

    p = plot(t_vec, quant[1, :], legend = :outertopright, color="#E69F00", linewidth=2.0, label = "PEPSDI")
    p = plot!(t_vec, quant_val[1, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label = "SmcML")
    p = plot!(t_vec, quant[2, :], color="#E69F00", linewidth=2.0, label = false)
    p = plot!(t_vec, quant_val[2, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label = false)
    p = plot!(t_vec, quant[3, :], color="#E69F00", linewidth=2.0, label = false)
    p = plot!(t_vec, quant_val[3, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label = false)
    title!("SIR model 0.05, 0.5, 0.95 quantiles : SSA")

    if return_plot
        return p
    else
        return t_vec, quant_val
    end
end


function check_tau_solver(times_run)

    S_left = convert(Array{Int16, 2}, [1 1 0; 0 1 0])
    S_right = convert(Array{Int16, 2}, [0 2 0; 0 0 1])
    c = [0.1 / 1000, 0.01]
    p = DynModInput(c, [0.0], [0.0])
    model = PoisonModel(sir_h_vec!, 
                        sir_x0!, 
                        sir_obs, 
                        empty_func, 
                        UInt16(3), UInt16(1), UInt16(2), S_left - S_right)

    t_vec = 0.0:1e-2:250.0
    i_max =length(t_vec)
    data_save = zeros(Float64, i_max, times_run)
    for i in 1:times_run
        t_vec_tmp, u = solve_poisson_model(model, (0.0, 250.0), p, 1e-2)
        data_save[:, i] .= u[2, :]
    end

    quant = zeros(Float64, 3, i_max)
    quant[1, :] .= [median(data_save[i, :]) for i in 1:i_max]
    quant[2, :] .= [quantile(data_save[i, :], 0.05) for i in 1:i_max]
    quant[3, :] .= [quantile(data_save[i, :], 0.95) for i in 1:i_max]

    t_vec2, quant_val = check_ssa_solver(times_run; return_plot=false)
    p = plot(t_vec, quant[1, :], legend = :outertopright, color="#E69F00", linewidth=2.0, label = "PEPSDI")
    p = plot!(t_vec2, quant_val[1, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label = "SmcML")
    p = plot!(t_vec, quant[2, :], color="#E69F00", linewidth=2.0, label=false)
    p = plot!(t_vec2, quant_val[2, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label=false)
    p = plot!(t_vec, quant[3, :], color="#E69F00", linewidth=2.0, label=false)
    p = plot!(t_vec2, quant_val[3, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label=false)

    title!("SIR-model 0.05, 0.5, 0.95 quantiles : Tau-leaping")

    return p
end


function check_extrand_solver(times_run)

    S_left = convert(Array{Int16, 2}, [1 1 0; 0 1 0])
    S_right = convert(Array{Int16, 2}, [0 2 0; 0 0 1])
    model = ExtrandModel(sir_h_vec!, 
                        sir_h_vec_max!,
                        sir_x0!, 
                        sir_obs, 
                        empty_func, 
                        3, 1, 2, S_left - S_right)
    t_vec = 0.0:1.0:250.0
    c = [0.1 / 1000, 0.01]
    p = DynModInput(c, [0.0], [0.0])
    i_max =length(t_vec)
    data_save = zeros(Float64, i_max, times_run)
    for i in 1:times_run
        t_vec, u = solve_extrande_model(model, t_vec, p)
        data_save[:, i] .= u[2, :]
    end
    quant = zeros(Float64, 3, i_max)
    quant[1, :] .= [median(data_save[i, :]) for i in 1:i_max]
    quant[2, :] .= [quantile(data_save[i, :], 0.05) for i in 1:i_max]
    quant[3, :] .= [quantile(data_save[i, :], 0.95) for i in 1:i_max]

    p = [0.1 / 1000, 0.01]
    u0 = [999, 1, 0]
    prob = DiscreteProblem(sir_model, u0, (0.0, 250.0), p)
    jump_prob = JumpProblem(sir_model, prob, Direct())

    i_max = length(t_vec)
    data_save = zeros(Float64, i_max, times_run)
    for i in 1:times_run
        prob = DiscreteProblem(sir_model, u0, (0.0, 250.0), p)
        jump_prob = JumpProblem(sir_model, prob, Direct())
        sol = solve(jump_prob, SSAStepper())
        for j in 1:i_max
            data_save[j, i] = sol(t_vec[j])[2]
        end
    end

    quant_val = zeros(Float64, 3, i_max)
    quant_val[1, :] .= [median(data_save[i, :]) for i in 1:i_max]
    quant_val[2, :] .= [quantile(data_save[i, :], 0.05) for i in 1:i_max]
    quant_val[3, :] .= [quantile(data_save[i, :], 0.95) for i in 1:i_max]

    p = plot(t_vec, quant[1, :], legend = :outertopright, color="#E69F00", linewidth=2.0)
    p = plot!(t_vec, quant_val[1, :], linestyle=:dash, color="#56B4E9", linewidth=2.0)
    p = plot!(t_vec, quant[2, :], color="#E69F00", linewidth=2.0, label=false)
    p = plot!(t_vec, quant_val[2, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label=false)
    p = plot!(t_vec, quant[3, :], color="#E69F00", linewidth=2.0, label=false)
    p = plot!(t_vec, quant_val[3, :], linestyle=:dash, color="#56B4E9", linewidth=2.0, label=false)
    title!("SIR-model 0.05, 0.5, 0.95 quantiles : Extrand")

    return p
end

dir_save = pwd() * "/tests/Stochastic_solvers/"
if !isdir(dir_save)
    mkpath(dir_save)
end

println("Running tests stochastis solvers")
p1 = check_tau_solver(20000)
p2 = check_ssa_solver(40000)
p3 = check_sde_solver(10000)
p4 = check_extrand_solver(40000)
savefig(p1, dir_save * "Tau_leaping.png")
savefig(p2, dir_save * "SSA.png")
savefig(p3, dir_save * "SDE.png")
savefig(p4, dir_save * "Extrand.png")