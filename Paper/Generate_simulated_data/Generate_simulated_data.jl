using Distributions
using CSV
using DataFrames
using Printf
using Random
using LinearAlgebra

tmp = push!(LOAD_PATH, pwd() * "/src")
using PEPSDI


include(pwd() * "/Paper/Models/Ornstein.jl")
include(pwd() * "/Paper/Models/Ornstein_mixed.jl")
include(pwd() * "/Paper/Models/Schlogl_ssa.jl")
include(pwd() * "/Paper/Models/Clock_model.jl")
include(pwd() * "/Paper/Models/Clock_fig1.jl")


# Struct contianing simulation result for an individual.
# Args:
#     t_vec, time-vector
#     y_vec, y-matrix (each row is an observation) for the observed
#       time-points
#     id, id of the simulated individual
struct SimResult
    t_vec::Array{Float64, 1}
    y_vec::Array{Float64, 2}
    id::Int64
    dim_obs::Int64
end


# Function that writes simulated data to file for one or more individuals.
# The data is stored in a tidy-format, that is each row corresponds to one
# observation for one time-point. The data is also saved with an id referring
# to the individual which the data corresponds to.
# Args:
#     sim_result, array of sim_result entries
#     dir_save, directory where to save the result
#     file_name, name of file to be saved
#     dim_obs, dimension of observeble
# Returns:
#     void
function write_simulated_data(sim_result, dir_save, file_name)

    # Sanity check input
    y_vec = sim_result[1].y_vec
    dim_obs = sim_result[1].dim_obs
    if size(y_vec)[1] != dim_obs
        print("Error, y_vec has wrong dimensions. Should have equal number
        of rows as observed outputs\n")
    end
    if !isa(sim_result, Array)
        print("Error, sim_result must be an array")
    end

    # Ensure directory exists
    mkpath(dir_save)
    file_save = dir_save * file_name
    # If file exists, delete as appending later
    if isfile(file_save)
        rm(file_save)
    end

    # Write each output variable for all individuals
    # i ensures correctly written header
    i = 1
    for sim_res in sim_result
        for j in 1:dim_obs
            df = DataFrame()
            lab = string(j)
            len = length(sim_res.t_vec)
            df[!, "time"] = sim_res.t_vec
            df[!, "obs"] = sim_res.y_vec[j, :]
            df[!, "obs_id"] = repeat([lab], len)
            df[!, "id"] = repeat([sim_res.id], len)

            if i == 1 && j == 1
                CSV.write(file_save, df, append=false)
            else
                CSV.write(file_save, df, append=true)
            end
        end
        i += 1
    end

end


# Function for simulating multiple-individuals to file for a stochastic 
# differential equation. Note, the user provides the distribution of the 
# individual parameters and whether or not they are estimated on the log-scale. 
# Args:
#     model, sde-model object with observation model 
#     t_vec, time-vector of points where to save result 
#     n_individuals, number of individuals to simulate 
#     eta_data, array with [mean, cov_mat, dist_param, bool], where bool 
#       states if parameters are on log-scale, and dist is distribution
#     kappa_data, vector with [kappa_val, bool], where bool states if log or not 
#     error_dist, error-distribution
#     model_name
# Returns:
#     void 
function simulate_mult_ind(model, t_vec, n_individuals, eta_data, kappa_data, 
    error_dist, model_name::String; model_type="SDE", dt=1e-4, dir_save="")

    # Given model-name, construct correct dir for svaing 
    if dir_save == ""
        if model_type == "SDE"
            dir_save = pwd() * "/Intermediate/Simulated_data/SDE/Multiple_ind/" * model_name * "/"
        else
            dir_save = pwd() * "/Intermediate/Simulated_data/SSA/Multiple_ind/" * model_name * "/"
        end
    end


    # Create distribution to sample parameters from 
    dist_use = eta_data[3]
    if dist_use == "normal"
        dist = MvNormal(eta_data[1], eta_data[2])
    elseif dist_use == "log_normal"
        dist = MvLogNormal(MvNormal(eta_data[1], eta_data[2]))
    else
        @printf("Error: Provide either normal or log-normal as distribution to use")
    end

    # Process up the kappa-vector 
    kappa_vec = kappa_data[1]
    if kappa_data[2] == true
        kappa_vec .= exp.(kappa_vec)
    end

    # For initial values 
    if model_type == "SDE"
        x0 = Array{FLOAT, 1}(undef, model.dim)
    else
        x0 = Array{UInt16, 1}(undef, model.dim)
    end
    c_vec_mat = Array{FLOAT, 2}(undef, (length(dist), n_individuals))

    # Simulate the individuals 
    sim_results = []
    for i in 1:n_individuals

        # Ensure correct transformation for c-vector 
        c_vec = rand(dist, 1)[:, 1]
        if eta_data[4] == true
            c_vec .= exp.(c_vec)
        end
        println(c_vec)
        c_vec_mat[:, i] .= c_vec

        @printf("i = %d", i)

        # Simulate model and add results to sim_results
        param_mod = DynModInput(c_vec, kappa_vec, zeros(0))
        model.calc_x0!(x0, param_mod)
        if model_type == "SDE"
            t_vec_save, y_vec = simulate_data_sde(model, error_dist, t_vec_sim, param_mod, x0, dt=dt)
        elseif model_type == "SSA"
            t_vec_save, y_vec = simulate_data_ssa(model, error_dist, t_vec_sim, param_mod)
        elseif model_type == "Extrande"
            t_vec_save, y_vec = simulate_data_extrande(model, error_dist, t_vec_sim, param_mod)
        end
        push!(sim_results, SimResult(t_vec_save, y_vec, i, model.dim_obs))

    end

    write_simulated_data(sim_results, dir_save, model_name*".csv")

    return c_vec_mat
end


# Different cases (user input)


if ARGS[1] == "Single_ornstein"

    println("Simulating single individual ornstein model ...")

    if length(ARGS) == 1
        Random.seed!(1234)
        tag_save = ""
    else
        Random.seed!(parse(Int64, ARGS[2]))
        tag_save = ARGS[2]
    end
    
    sde_mod =  init_sde_model(alpha_ornstein_full, 
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
    sim_result = [SimResult(t_vec, y_vec, 1, sde_mod.dim_obs)]
    # Writing result to file
    write_simulated_data(sim_result, "./Intermediate/Simulated_data/SDE/Ornstein/",
        "Ornstein" * tag_save * ".csv")

    println("done")
end


if ARGS[1] == "Single_schlogl_ssa"

    println("Simulating Schlogl-model single individual")

    if length(ARGS) == 1
        Random.seed!(12345)
        tag_save = ""
    else
        Random.seed!(parse(Int64, ARGS[2]))
        tag_save = ARGS[2]
    end
    
    S_left = convert(Array{Int16, 2}, zeros(Int16, (4, 1)) .+ [2; 3; 0; 1])
    S_right = convert(Array{Int16, 2}, zeros(Int16, (4, 1)) .+ [3; 2; 1; 0])
    my_model = SsaModel(schlogl_h_vec_full!, 
                        schlogl_x0!, 
                        schlogl_obs_ssa, 
                        schlogl_prop_obs, 
                        UInt16(1), UInt16(1), UInt16(4), S_left - S_right)

    error_dist = Normal(0, 2.0)
    p = DynModInput([1.8e-1, 2.5e-4, 2.2e3], [0.0], zeros(0))
    #t_vec_sim = 1.0:1:50
    t_vec, y_vec = simulate_data_ssa(my_model, error_dist, 1.0:0.5:50, p)
    sim_result = [SimResult(t_vec, y_vec, 1, 1)]
    # Writing result to file
    write_simulated_data(sim_result, "./Intermediate/Simulated_data/SSA/Schlogl/",
        "Schlogl" * tag_save * ".csv")
end


if ARGS[1] == "Single_clock_extrande"

    println("Single individual clock extrande")
    Random.seed!(1234)
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model = ExtrandModel(clock_h_vec!, clock_h_vec_max!, clock_x0!, clock_obs, clock_prob_obs, 2, 1, 4, S_left - S_right)

    error_dist = Normal(0, 2.0)
    p = DynModInput([4.0, 25.0, 2.0, 1.0], [0.0], [0.0])

    t_vec, y_vec = simulate_data_extrande(my_model, error_dist, 0.5:0.75:48, p)
    sim_result = [SimResult(t_vec, y_vec, 1, 1)]
    # Writing result to file
    write_simulated_data(sim_result, "./Intermediate/Simulated_data/SSA/Clock/","Clock.csv")

end


if ARGS[1] == "Multiple_schlogl_kappa_ssa"

    println("Multiple Schlogl kappa ssa")
    # Simulate biological schlogl-model 
    Random.seed!(321)
    S_left = convert(Array{Int16, 2}, zeros(Int16, (4, 1)) .+ [2; 3; 0; 1])
    S_right = convert(Array{Int16, 2}, zeros(Int16, (4, 1)) .+ [3; 2; 1; 0])
    my_model = SsaModel(schlogl_h_vec!, 
                        schlogl_x0!, 
                        schlogl_obs_sde, 
                        schlogl_prop_obs, 
                        UInt16(1), UInt16(1), UInt16(4), S_left - S_right)

    scale_vec = [0.1]
    cor_mat = rand(LKJ(1, 0.1))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[-2.17, -8.73], true]
    eta_data = [[7.2], cov_mat, "normal", true]
    error_dist = Normal(0, 2.0)
    t_vec_sim = 1.0:0.5:50

    data_param = simulate_mult_ind(my_model, t_vec_sim, 40, eta_data, kappa_data, error_dist, "schlogl", model_type="SSA")
end

if ARGS[1] == "Multiple_schlogl_kappa_ssa_rev"

    println("Multiple Schlogl kappa ssa rev")
    # Simulate biological schlogl-model 
    Random.seed!(321)
    S_left = convert(Array{Int16, 2}, zeros(Int16, (4, 1)) .+ [2; 3; 0; 1])
    S_right = convert(Array{Int16, 2}, zeros(Int16, (4, 1)) .+ [3; 2; 1; 0])
    my_model = SsaModel(schlogl_h_vec!, 
                        schlogl_x0!, 
                        schlogl_obs_sde, 
                        schlogl_prop_obs, 
                        UInt16(1), UInt16(1), UInt16(4), S_left - S_right)

    scale_vec = [0.1]
    cor_mat = rand(LKJ(1, 0.1))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[-2.17, -8.73], true]
    eta_data = [[7.2], cov_mat, "normal", true]
    error_dist = Normal(0, 2.0)
    t_vec_sim = 1.0:0.5:50

    data_param = simulate_mult_ind(my_model, t_vec_sim, 150, eta_data, kappa_data, error_dist, "schlogl_rev", model_type="SSA")
end


if ARGS[1] == "Multiple_clock_extrande"

    println("Multiple clock extrande")
    Random.seed!(123)
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model = ExtrandModel(clock_h_vec!, clock_h_vec_max!, clock_x0!, clock_obs, clock_prob_obs, 2, 1, 4, S_left - S_right)

    scale_vec = [0.4, 0.2, 0.2, 0.1]
    cor_mat = rand(LKJ(4, 3.0))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[0.0], true]
    eta_data = [log.([3.0, 30.0, 3.0, 2.0]), cov_mat, "normal", true]

    println(cor_mat)

    error_dist = Normal(0, 2)
    t_vec_sim = 0.5:0.75:48
    data_param = simulate_mult_ind(my_model, t_vec_sim, 40, eta_data, kappa_data, error_dist, "Clock", model_type="Extrande")

end


if ARGS[1] == "Multiple_clock_extrande_review"

    println("Multiple clock extrande")
    Random.seed!(1234)
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model = ExtrandModel(clock_h_vec!, clock_h_vec_max!, clock_x0!, clock_obs, clock_prob_obs, 2, 1, 4, S_left - S_right)

    scale_vec = [0.4, 0.2, 0.2, 0.1]
    cor_mat = rand(LKJ(4, 3.0))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[0.0], true]
    eta_data = [log.([3.0, 30.0, 3.0, 2.0]), cov_mat, "normal", true]

    println(cor_mat)

    error_dist = Normal(0, 2)
    t_vec_sim = 0.5:0.75:48
    data_param = simulate_mult_ind(my_model, t_vec_sim, 60, eta_data, kappa_data, error_dist, "Clock_review", model_type="Extrande")

end


if ARGS[1] == "Multiple_clock_double"

    println("Multiple clock extrande")
    Random.seed!(123)
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model1 = ExtrandModel(clock_h_vec!, clock_h_vec_max!, clock_x0!, clock_obs, clock_prob_obs, 2, 1, 4, S_left - S_right)
    my_model2 = ExtrandModel(clock_h_vec2!, clock_h_vec_max2!, clock_x0!, clock_obs, clock_prob_obs, 2, 1, 4, S_left - S_right)

    scale_vec = [0.4, 0.2, 0.2, 0.1]
    cor_mat = rand(LKJ(4, 3.0))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[0.0], true]
    eta_data = [log.([3.0, 30.0, 3.0, 2.0]), cov_mat, "normal", true]

    println(cor_mat)

    error_dist = Normal(0, 2)
    t_vec_sim = 0.5:0.75:48
    n_individuals = 5000
    data_param = simulate_mult_ind(my_model1, t_vec_sim, n_individuals, eta_data, kappa_data, error_dist, "Ex1_5000", model_type="Extrande", dir_save=dir_save)
    data_param = simulate_mult_ind(my_model2, t_vec_sim, n_individuals, eta_data, kappa_data, error_dist, "Ex2_5000", model_type="Extrande", dir_save=dir_save)
    data_param = simulate_mult_ind(my_model1, t_vec_sim, 40, eta_data, kappa_data, error_dist, "Ex1_40", model_type="Extrande", dir_save=dir_save)
    data_param = simulate_mult_ind(my_model2, t_vec_sim, 40, eta_data, kappa_data, error_dist, "Ex2_40", model_type="Extrande", dir_save=dir_save)
    data_param = simulate_mult_ind(my_model2, t_vec_sim, 100000, eta_data, kappa_data, error_dist, "Ex2_100000", model_type="Extrande", dir_save=dir_save)

end


if ARGS[1] == "Multiple_schlogl_benchmark"

    println("Multiple Schlogl kappa sde")

    my_model = init_sde_model(schlogl_alpha_kappa, 
                               schlogl_beta_kappa, 
                               schlogl_x0!, 
                               schlogl_obs_ssa, 
                               schlogl_prop_obs, 
                               1, 1)
    scale_vec = [0.1]
    cor_mat = rand(LKJ(1, 0.1))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[-2.17, -8.73], true]
    eta_data = [[7.2], cov_mat, "normal", true]
    error_dist = Normal(0, 2)
    t_vec_sim = 0.25:0.25:20.0
    dir_save = pwd() * "/Intermediate/Simulated_data/SDE/Multiple_ind/" * "Schlogl_benchmark" * "/"

    n_individuals = parse(Int64, ARGS[2])
    
    @printf("N_individuals = %d\n", n_individuals)

    Random.seed!(123)
    name_save = "N_ind" * string(n_individuals) 

    data_param = simulate_mult_ind(my_model, t_vec_sim, n_individuals, eta_data, kappa_data, 
        error_dist, name_save, model_type="SDE", dir_save = dir_save, dt=1e-3)

    # Write individual parameter to disk 
    data_save = convert(DataFrame, data_param')
    rename!(data_save, ["c1"])
    println(data_save)
    CSV.write(dir_save * "/Param" * string(n_individuals) * ".csv", data_save)
    
end 


if ARGS[1] == "Clock_fig1"

    println("Multiple clock1 extrande")
    Random.seed!(12345)
    S_left = convert(Array{Int16, 2}, [0 0; 1 0; 1 0; 0 1])
    S_right = convert(Array{Int16, 2}, [1 0; 1 1; 0 0; 0 0])
    my_model = ExtrandModel(clock_fig1_h_vec!, clock_fig1_h_vec_max!, clock_fig1_x0!, clock_fig1_obs, clock_fig1_prob_obs, 2, 1, 4, S_left - S_right)

    scale_vec = [0.4, 0.2, 0.2, 0.1]
    cor_mat = rand(LKJ(4, 3.0))
    cov_mat = Diagonal(scale_vec) * cor_mat * Diagonal(scale_vec)
    kappa_data = [[0.0], true]
    eta_data = [log.([3.0, 30.0, 3.0, 0.5]), cov_mat, "normal", true]

    println(cor_mat)

    error_dist = Normal(0, 2)
    t_vec_sim = 0.5:0.75:72
    data_param = simulate_mult_ind(my_model, t_vec_sim, 40, eta_data, kappa_data, error_dist, "Clock_fig1", model_type="Extrande")
end

