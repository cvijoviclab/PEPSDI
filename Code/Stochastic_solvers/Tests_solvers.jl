#=
    Tests for the stochastic simulators. Currently the file checks that 
    all simulator functions can be run without crashing. In the future tests 
    will be added to assure that all simulators are accurately implmented. 
=#


using Printf
using Random
using LinearAlgebra
using Distributions
tmp = push!(LOAD_PATH, pwd() * "/Code")
using MyPkg


include(pwd() * "/Code/Models/Ornstein.jl")
include(pwd() * "/Code/Models/Ornstein_mixed.jl")
include(pwd() * "/Code/Models/Schlogl_ssa.jl")
include(pwd() * "/Code/Models/Clock_model.jl")
include(pwd() * "/Code/Models/Model_auto.jl")



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

end


run_solver_functions()
