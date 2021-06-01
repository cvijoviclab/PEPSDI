#=
    File to launch benchmark of adaptive mcmc-proposals for the ornstein-model. 
    The benchmark script is launched with five args:

    ARGS[1] : Sampler (RAM, GenAM or AM). Can also be start-guess 
    ARGS[2] : Start-guess (sg1, sg2, sg3, sg4 or sg5)
    ARGS[3] : Data-set (1, 12 or 123)
    ARGS[4] : N-repititions (integer)
    ARGS[5] : Pilot (true or false)

    The benchmark results are saved in Intermediate/Single_individual. A R-script 
    processes these results to yield the result presented in the paper. 
=# 


using Distributions
using Printf
using Random
using LinearAlgebra
tmp = push!(LOAD_PATH, pwd() * "/Code") # Push PEPSDI into load-path 
using PEPSDI # Load PEPSDI 


# Load the model-equations used 
include(pwd() * "/Code/Models/Schlogl_ssa.jl")


if length(ARGS) == 5
    if ARGS[5] == "pilot"
        pilot = true
    else
        pilot = false
    end
end


function test_proposal(n_particles, sampler_use, n_samples, tag, data_set; pilot=false, update_it=1)

    if tag == "sg1"
        dir_save = "Schlogl_test_proposal/sg1_" * data_set
        init_ind_sg = [-0.03823288168007821, -5.882214238452927, 5.747266828792446]
        init_error_sg = [2.196436815922223]

    elseif tag == "sg2"
        dir_save = "Schlogl_test_proposal/sg2_" * data_set
        init_ind_sg = [-1.0830989288830426, -6.868503765536172, 6.599769357415814]
        init_error_sg = [2.7674670822646124]

    elseif tag == "sg3"
        dir_save = "Schlogl_test_proposal/sg3_" * data_set
        init_ind_sg = [0.038339912479200544, -6.326492931974268, 3.7918740672122837]
        init_error_sg = [1.6374670546184498]

    elseif tag == "sg4"
        dir_save = "Schlogl_test_proposal/sg4_" * data_set
        init_ind_sg = [-0.31507968270936304, -6.347659651282988, 5.834511739297202]
        init_error_sg = [2.01762213657855]

    elseif tag == "sg5"
        dir_save = "Schlogl_test_proposal/sg5_" * data_set
        init_ind_sg = [-0.45655799531887686, -6.142200468134376, 6.751286867082489]
        init_error_sg = [1.4973661434486383]
    end

    P_mat = [1]
    my_model = init_sde_model(schlogl_alpha, 
                             schlogl_beta, 
                             schlogl_x0!, 
                             schlogl_obs_sde, 
                             schlogl_prop_obs, 
                             1, 1, P_mat) 

    prior_ind_param = [Normal(-1.0, 10.0), 
                       Normal(-3.0, 10.0), 
                       Normal(7.0, 10.0)]
    prior_error_param = [Normal(2.0, 0.5)]
    param_info = init_param(prior_ind_param, 
                            prior_error_param, 
                            ind_param_pos=false, 
                            ind_param_log=true,
                            init_ind_param=log.([1.8e-1, 2.5e-4, 2.2e3]), 
                            error_param_pos=true)

    # Filter information 
    dt = 5e-2
    filter_opt = init_filter(ModDiffusion(), dt, n_particles=n_particles, rho=0.99)
    rho = filter_opt.rho

    # File-locations
    path_data = pwd() * "/Intermediate/Simulated_data/SSA/Schlogl/Schlogl" * data_set * ".csv"
    file_loc = init_file_loc(path_data, "Schlogl_test_proposal2")

    data_obs = CSV.read(file_loc.path_data, DataFrame)
    ind_data = init_ind_data(data_obs, filter_opt)

    # Sampler 
    cov_mat = diagm([0.1, 0.1, 0.1, 0.1])

    if sampler_use == "AM" && pilot == true 
        mcmc_sampler = init_mcmc(AmSampler(), param_info, step_before_update=100, cov_mat=cov_mat, lambda = 2.38^2 / 4)
        file_loc = init_file_loc(path_data, dir_save * "/AM")

    elseif sampler_use == "RAM" && pilot == true
        mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
        file_loc = init_file_loc(path_data, dir_save * "/RAM")

    elseif sampler_use == "RAM_alt" && pilot == true
        mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat, update_it=update_it)
        file_loc = init_file_loc(path_data, dir_save * "/RAM_alt")

    elseif sampler_use == "GenAM" && pilot == true
        mcmc_sampler = init_mcmc(GenAmSampler(), param_info, step_before_update=100, cov_mat=cov_mat, log_lambda= log(2.38^2 / 4))
        file_loc = init_file_loc(path_data, dir_save * "/GenAM")

    elseif sampler_use == "RAM" && pilot == false
        file_loc = init_file_loc(path_data, dir_save * "/RAM")
        mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
        param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, 
            sampler_name = "Ram_sampler")
        mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.rho)

        filter_opt = init_filter_pilot(filter_opt, file_loc, rho, "Ram_sampler")

    elseif sampler_use == "RAM_alt" && pilot == false
        file_loc = init_file_loc(path_data, dir_save * "/RAM_alt")
        mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat, update_it=update_it)
        param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, 
            sampler_name = "Ram_sampler")
        mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.rho)

        filter_opt = init_filter_pilot(filter_opt, file_loc, rho, "Ram_sampler")

    elseif sampler_use == "GenAM" && pilot == false
        file_loc = init_file_loc(path_data, dir_save * "/GenAM")
        mcmc_sampler = init_mcmc(GenAmSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
        param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, 
            sampler_name = "Gen_am_sampler")
        mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.rho)

        filter_opt = init_filter_pilot(filter_opt, file_loc, rho, "Gen_am_sampler")

    elseif sampler_use == "AM" && pilot == false
        file_loc = init_file_loc(path_data, dir_save * "/AM")
        mcmc_sampler = init_mcmc(AmSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
        param_info_new = change_start_val_to_pilots(param_info, file_loc, filter_opt, 
            sampler_name = "Am_sampler")
        mcmc_sampler = init_mcmc_pilot(mcmc_sampler, file_loc, filter_opt.rho)

        filter_opt = init_filter_pilot(filter_opt, file_loc, rho, "Am_sampler")
    end        

    if pilot == true
        pilot_run_info = init_pilot_run([prior_ind_param, prior_error_param], 
                        init_ind_param=init_ind_sg, 
                        init_error_param=init_error_sg,
                        n_particles_pilot=3000, 
                        n_samples_pilot=5000,
                        n_particles_investigate=[9, 50],
                        n_times_run_filter=100, 
                        rho_list=[0.99])

        tune_particles_single_individual(pilot_run_info, mcmc_sampler, param_info, 
            filter_opt, my_model, deepcopy(file_loc))
    else

        samples, log_lik_val, sampler = run_mcmc(n_samples, mcmc_sampler, param_info_new, 
                                                filter_opt, my_model, file_loc)     
                                                
        return samples, log_lik_val, sampler
    end
    
end


function find_start_guess_schlögl()
    
    prior_ind_param = [Normal(-1.0, 2.0), 
                       Normal(-3.0, 2.0), 
                       Normal(7.0, 2.0)]
    prior_error_param = [Normal(2.0, 0.5)]

    P_mat = [1]
    model = init_sde_model(schlogl_alpha, 
                             schlogl_beta, 
                             schlogl_x0!, 
                             schlogl_obs_sde, 
                             schlogl_prop_obs, 
                             1, 1, P_mat) 

    # Filter information 
    dt = 5e-2
    n_particles = 3000
    filter_opt = init_filter(ModDiffusion(), dt, n_particles=n_particles, rho=0.99)

    # File-locations
    path_data = pwd() * "/Intermediate/Simulated_data/SSA/Schlogl/Schlogl.csv"
    file_loc = init_file_loc(path_data, "Schlogl_test_proposal")

    # Compute the individual data 
    data_obs = CSV.read(file_loc.path_data, DataFrame)
    ind_data = init_ind_data(data_obs, filter_opt)
 
    param_info = init_param(prior_ind_param, 
                            prior_error_param, 
                            ind_param_pos=false, 
                            ind_param_log=true,
                            error_param_pos=true)

    n_param_infer = length(param_info.prior_ind_param) + length(param_info.prior_error_param)
    i_range_ind = 1:length(param_info.prior_ind_param)
    i_range_error = length(param_info.prior_ind_param)+1:n_param_infer


    n_start_guess_reach = 5
    n_correct = 0
    println("Finding start-guesses")
    while n_correct < n_start_guess_reach
    
        ind_param = rand.(prior_ind_param)
        error_param = rand.(prior_error_param)
        param_tot = deepcopy(vcat(ind_param, error_param))

        # Init the model parameters object
        mod_param = init_model_parameters(deepcopy(ind_param),
                                          deepcopy(error_param),
                                          model, 
                                          covariates=ind_data.cov_val)

        rand_num_old = create_rand_num(ind_data, model, filter_opt)

        map_proposal_to_model_parameters_ind!(mod_param, param_tot, param_info, i_range_ind, i_range_error)

        log_lik = run_filter(filter_opt, mod_param, rand_num_old, model, ind_data)

        if abs(log_lik) != Inf
            println("Ind_param =", ind_param, "error_param = ", error_param)
            n_correct += 1
        end

    end

end


if ARGS[1] == "Start_guess" 
    # Employed when finding the start-guesses for each pilot run. 
    find_start_guess_schlögl()
end


if ARGS[1] == "RAM" && pilot
    sg = ARGS[2]
    data_set = ARGS[3]
    test_proposal(50, "RAM", 60000, sg, data_set, pilot=true)
end
if ARGS[1] == "RAM" && !pilot
    sg = ARGS[2]
    data_set = ARGS[3]

    n_rep = parse(Int64, ARGS[4])
    for i in 1:n_rep
        test_proposal(50, "RAM", 60000, sg, data_set, pilot=false)
    end
end


if ARGS[1] == "RAM_alt" && pilot
    sg = ARGS[2]
    data_set = ARGS[3]
    test_proposal(50, "RAM_alt", 60000, sg, data_set, pilot=true, update_it=20)
end
if ARGS[1] == "RAM_alt" && !pilot
    sg = ARGS[2]
    data_set = ARGS[3]
    
    n_rep = parse(Int64, ARGS[4])
    for i in 1:n_rep
        test_proposal(50, "RAM_alt", 60000, sg, data_set, pilot=false, update_it=20)
    end
end


if ARGS[1] == "GenAM" && pilot
    sg = ARGS[2]
    data_set = ARGS[3]
    test_proposal(50, "GenAM", 60000, sg, data_set, pilot=true)
end
if ARGS[1] == "GenAM" && !pilot
    sg = ARGS[2]
    data_set = ARGS[3]
    n_rep = parse(Int64, ARGS[4])
    for i in 1:n_rep
        test_proposal(50, "GenAM", 60000, sg, data_set, pilot=false)
    end
end


if ARGS[1] == "AM" && pilot
    sg = ARGS[2]
    data_set = ARGS[3]
    test_proposal(50, "AM", 60000, sg, data_set, pilot=true)
end
if ARGS[1] == "AM" && !pilot
    sg = ARGS[2]
    data_set = ARGS[3]

    n_rep = parse(Int64, ARGS[4])
    for i in 1:n_rep
        test_proposal(50, "AM", 60000, sg, data_set, pilot=false)
    end
end
