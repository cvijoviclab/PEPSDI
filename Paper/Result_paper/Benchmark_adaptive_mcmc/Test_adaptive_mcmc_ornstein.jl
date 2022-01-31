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


# The differenttial equations for the Ornstien-model 
include(pwd() * "/Code/Models/Ornstein.jl")


if length(ARGS) == 5
    if ARGS[5] == "pilot"
        pilot = true
    else
        pilot = false
    end
end


function test_proposal(n_particles, sampler_use, n_samples, tag, data_set; pilot=false, update_it=1)

    if tag == "sg1"
        dir_save = "Ornstein_test_proposal/sg1_" * data_set
        init_ind_sg = [1.1902678809862768, 3.04817970778924, 1.142650902867199]
        init_error_sg = [0.301699480755097]

    elseif tag == "sg2"
        dir_save = "Ornstein_test_proposal/sg2_" * data_set
        init_ind_sg = [-0.396679079295223, 0.33528745480831235, 0.9809678267585334]
        init_error_sg = [0.08180866297616052]

    elseif tag == "sg3"
        dir_save = "Ornstein_test_proposal/sg3_" * data_set
        init_ind_sg = [0.27381537121215616, 0.8057709328942795, -0.33936602980781916]
        init_error_sg = [0.1938778380716992]

    elseif tag == "sg4"
        dir_save = "Ornstein_test_proposal/sg4_" * data_set
        init_ind_sg = [-0.8889357468973064, 1.3272149538714721, 0.5924032426727195]
        init_error_sg =  [0.16431870191368597]

    elseif tag == "sg5"
        dir_save = "Ornstein_test_proposal/sg5_" * data_set
        init_ind_sg = [-0.28113324410584123, 0.2651144099349385, -0.7174100165743348]
        init_error_sg = [0.48485346932991313]
    end

    prior_ind_param = [Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(0.0, 1.0)]
    prior_error_param = [Gamma(1.0, 0.4)]

    my_model = init_sde_model(alpha_ornstein_full, 
        beta_ornstein_full, 
        calc_x0_ornstein!, 
        ornstein_obs, 
        prob_ornstein_full,
        1, 1)

    param_info = init_param(prior_ind_param, 
                            prior_error_param, 
                            ind_param_pos=false, 
                            ind_param_log=true,
                            error_param_pos=true)

    # Filter information 
    dt = 1e-2
    rho = 0.99
    filter_opt = init_filter(BootstrapEm(), dt, rho=rho)

    # File-locations
    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Ornstein/Ornstein" * data_set * ".csv"
    file_loc = init_file_loc(path_data, "Ornstein_test_proposal")

    # Sampler 
    cov_mat = diagm([0.1, 0.1, 0.1, 0.1])

    if sampler_use == "AM" && pilot == true 
        mcmc_sampler = init_mcmc(AmSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
        file_loc = init_file_loc(path_data, dir_save * "/AM")

    elseif sampler_use == "RAM" && pilot == true
        mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
        file_loc = init_file_loc(path_data, dir_save * "/RAM")

    elseif sampler_use == "RAM_alt" && pilot == true
        mcmc_sampler = init_mcmc(RamSampler(), param_info, step_before_update=100, cov_mat=cov_mat, update_it=update_it)
        file_loc = init_file_loc(path_data, dir_save * "/RAM_alt")

    elseif sampler_use == "GenAM" && pilot == true
        mcmc_sampler = init_mcmc(GenAmSampler(), param_info, step_before_update=100, cov_mat=cov_mat)
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


function find_start_guess_ornstein()
    
    prior_ind_param = [Normal(0.0, 1.0), Normal(1.0, 1.0), Normal(0.0, 1.0)]
    prior_error_param = [Gamma(1.0, 0.4)]

    model = init_sde_model(alpha_ornstein_full, 
        beta_ornstein_full, 
        calc_x0_ornstein!, 
        ornstein_obs, 
        prob_ornstein_full,
        1, 1)

    # Filter information 
    dt = 1e-2
    rho = 0.99
    filter_opt = init_filter(BootstrapEm(), dt, rho=rho)

    # File-locations
    path_data = pwd() * "/Intermediate/Simulated_data/SDE/Ornstein/Ornstein1.csv"
    file_loc = init_file_loc(path_data, "Ornstein_test_proposal")

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

    Random.seed!(123)

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



if ARGS[1] == "find_start_guess"
    # Employed when finding the start-guesses for each pilot run. 
    find_start_guess_ornstein()
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

