"""
    init_pilot_run(model_prior; 
                   init_ind_param="mean", 
                   init_error_param="mean", 
                   n_pilot_runs=1, 
                   n_samples_pilot=1000, 
                   n_particles_pilot=10000, 
                   rho_list::Array{FLOAT, 1}=[0.0],
                   n_particles_investigate=[10, 50, 100, 500, 1000], 
                   n_times_run_filter=1000)

Intitialise TuneParticlesIndividual-struct for single-individual pilot-run

By choosing init_ind and/or init_error_param as arrays or random multiple 
pilot-runs are launched based on each row in the arrays, or randomly sampled 
from the prior. This results in array of multiple-structs being returned. 
Note, number of particles for the pilot should be a large value (e.g 1000), and 
the number of samples should be sufficiently large to reach the posterior. 
Number of times to run-the filter should be large enough to provide a good 
estimate of the variance. 

# Args
- `model_prior`: Entry 1: Individual-parameters priors. Entry 2: error-parameters priors 
- `init_ind_param`: Array if starting values values for inidividual parameters are 
    provided for pilot-run. Else use random, mode or mean for choosing starting via the priors. 
- `init_error_param`: See above.
- `n_pilot_runs`: If Arg2 or Arg3 equals random, decides the number of pilot-runs 
- `n_samples_pilot`: Number of mcmc-sampler steps in pilot run. 
- `n_particles_pilot`: Number of particles to use in pilot-run. 
- `rho_list`: Correlation-values to investigate when tuning filter 
    at central posterior location obtained from pilot-run. 
- `n_particles_investigate`: Array of particles to investigate when tuning filter 
    at central posterior location obtained from pilot-run. 
- `n_particles_pilot`: Number of times to run filter when tuning particles at  
    central posterior location obtained from pilot-run. 

See also: [`init_param`, `TuneParticlesIndividual`](@ref)
"""
function init_pilot_run(model_prior; 
                        init_ind_param="mean", 
                        init_error_param="mean", 
                        n_pilot_runs=1, 
                        n_samples_pilot=1000, 
                        n_particles_pilot=10000, 
                        rho_list::Array{FLOAT, 1}=[0.0],
                        n_particles_investigate=[10, 50, 100, 500, 1000], 
                        n_times_run_filter=1000)

    prior_ind_param = model_prior[1]
    prior_error_param = model_prior[2]
    n_ind_param = length(prior_ind_param)
    n_error_param = length(prior_error_param)

    T1  = Array{<:AbstractFloat}
    if typeof(init_ind_param) <: T1 && typeof(init_error_param) <: T1
        if size(init_ind_param, 2) != size(init_error_param, 2)
            print("Error: When init_ind_param and init_error_param both are") 
            print("two-dimensionsal arrays their number of rows must match")
            return 1 
        end
    end

    # Get number of pilot-runs 
    if typeof(init_ind_param) <: T1 || typeof(init_ind_param) <: T1
        if typeof(init_ind_param) <: T1
            n_pilot_runs = size(init_ind_param, 2)
        else
            n_pilot_runs = size(init_error_param, 2)
        end
    end

    # Set-up the pilot-run objects 
    pilot_run_info_array = []
    for i in 1:n_pilot_runs
        # Ensure correct input when getting parameters 
        if typeof(init_ind_param) <: T1
            ind_param = init_ind_param[:, i]
        else
            ind_param = init_ind_param
        end
        if typeof(init_error_param) <: T1
            error_param = init_error_param[:, i]
        else
            error_param = init_error_param
        end

        param_info_temp = init_param(prior_ind_param, 
                                    prior_error_param, 
                                    init_ind_param=ind_param, 
                                    init_error_param=error_param)
        ind_param = param_info_temp.init_ind_param
        error_param = param_info_temp.init_error_param

        pilot_run_info = TuneParticlesIndividual(n_particles_pilot, 
                                                 n_samples_pilot, 
                                                 n_particles_investigate, 
                                                 ind_param,
                                                 error_param, 
                                                 n_times_run_filter, 
                                                 rho_list)
        
        push!(pilot_run_info_array, pilot_run_info)
    end

    return pilot_run_info_array
end


"""
    change_param_info(param_info::InitParameterInfo, ind_param, error_param)

Create a copy of a InitParameterInfo-info struct with new individual and error-parameters. 

Functionality used when launching pilot-runs with different starting values. 
"""
function change_param_info(param_info::InitParameterInfo, ind_param, error_param)

    param_info_new = InitParameterInfo(
        param_info.prior_ind_param,
        param_info.prior_error_param,
        ind_param,
        error_param,
        param_info.n_param,
        param_info.ind_param_pos,
        param_info.error_param_pos,
        param_info.ind_param_log,
        param_info.error_param_log)
    param_info_new = deepcopy(param_info_new)

    return param_info_new
end


"""
    check_if_pilot_result_exists(file_loc, filter, n_samples, param_info)

Check if pilot-run results for specific sampler, filter and starting values. 

If pilot-run results does not exist for a specific sampler (specified in file-loc), 
filter and starting values (in param_info) false is returned and pilot-run will 
be launched. Else, true is returned together with parameter values at central 
posterior location and the experimental tag for the pilot-run. 

See also: [`run_pilot_run`](@ref)
"""
function check_if_pilot_result_exists(file_loc, filter, n_samples, param_info)

    # Initial values used/used in pilot run 
    init_value = vcat(param_info.init_ind_param, param_info.init_error_param)
    n_param = length(init_value)

    # Ensure that directory where pilot-runs are stored exists 
    dir_save = file_loc.dir_save * "Pilot_runs/"
    if !isdir(dir_save)
        mkpath(dir_save)
    end

    file_name = dir_save * "Central_pos_value.csv"
    # Return if file does not exist 
    if !isfile(file_name)
        i_exp = 0
        return false, nothing, i_exp
    end

    # Check if center position values exist 
    central_pos_val = nothing 
    data_central_pos = CSV.read(file_name, DataFrame)
    filter!(row -> row[:n_particles] == filter.n_particles, data_central_pos)
    filter!(row -> row[:n_samples] == n_samples, data_central_pos)
    filter!(row -> row[:correlation] == filter.rho, data_central_pos)

    # Case central-position values exist 
    if issubset(init_value, data_central_pos[!, "init_val"]) == true
        n_col_file = size(data_central_pos)[1]
        for i in 1:n_param:n_col_file
            i_file = i:i+(n_param-1)
            if init_value == data_central_pos[!, "init_val"][i_file] 
                central_pos_val = data_central_pos[!, "cent_pos_val"][i_file]
                i_exp = data_central_pos[!, "i_exp"][i_file[1]]
            end
        end
        # Get which experimental id in the file this corresponds to 
        return true, central_pos_val, i_exp
    else 
        i_exp = 0
        return false, central_pos_val, i_exp
    end

end


"""
    run_pilot_run(n_samples, mcmc_sampler, param_info, model, 
        file_loc, filter; percent_use=0.1)

Run pilot-run for specific sampler, initial-values, filter and model. 

Runs pilot-run for n_samples iterations using provided mcmc-sampler, starting-
values (param_info) and filter; and by using the last percent_use from running 
the mcmc-sampler computes a central-posterior location saveing the values in 
the directory provided by file-locations struct. 

See also: [`run_pilot_run`](@ref)
"""
function run_pilot_run(n_samples, mcmc_sampler, param_info, model, 
    file_loc, filter; percent_use=0.1)

    # Save the initial values 
    param_info_use = deepcopy(param_info)
    init_value = vcat(param_info.init_ind_param, param_info.init_error_param)

    @printf("Running pilot-run\n")
    # Run particle filter, ensure to keep the file-locations correct 
    file_loc_use_mcmc = deepcopy(file_loc)
    samples, log_lik_val, sampler = run_mcmc(n_samples, mcmc_sampler, param_info_use, 
                                             filter, model, file_loc_use_mcmc, pilot=true)

    # Caclulate central posterior values 
    n_param = size(samples)[1]
    i_use = convert(Int64, ceil((1-percent_use)*n_samples)):n_samples
    samples_use = samples[:, i_use]
    central_pos_val = median(samples_use, dims=2)

    # Ensure that directory where pilot-runs are stored exists 
    dir_save = file_loc.dir_save * "Pilot_runs/"
    if !isdir(dir_save)
        mkpath(dir_save)
    end
    file_name = dir_save * "Central_pos_value.csv"

    # If the file with posterior central values does not exist 
    if !isfile(file_name)
        i_exp = 1
        data_save = DataFrame(cent_pos_val = central_pos_val[:, 1], 
            init_val = init_value, 
            n_particles = repeat([filter.n_particles], n_param), 
            i_exp = repeat([i_exp], n_param), 
            n_samples = repeat([n_samples], n_param), 
            correlation = repeat([filter.rho], n_param))

        CSV.write(file_name, data_save)
    else 
        data_central_pos = CSV.read(file_name, DataFrame)
        i_exp = maximum(data_central_pos["i_exp"]) + 1
        data_save = DataFrame(cent_pos_val = central_pos_val[:, 1], 
            init_val = init_value, 
            n_particles = repeat([filter.n_particles], n_param), 
            i_exp = repeat([i_exp], n_param), 
            n_samples = repeat([n_samples], n_param), 
            correlation = repeat([filter.rho], n_param))
        CSV.write(file_name, data_save, append=true)
    end

    # Save the sampler option 
    save_mcmc_opt_pilot(sampler, file_loc, i_exp, 1)

    return central_pos_val, i_exp
end


"""
    investigate_different_particles(n_times_run_filter,
                                    filter_opt, 
                                    n_particles_try,
                                    mod_param, 
                                    file_loc, 
                                    model)

Tune the number of particles for a single-individual. 

For a specific model, filter and individual (mod_param) runs the particle filter 
and increases the number of particles until variance is below the target variance. 
If filter is correlated, target variance is calculated as 2.16²/ (1 - rho^2). 
If not-correlated, target variance equals 2, based on recomendations by Wiqvist et al. 
Returns the number of particles to use. 

See also: [`run_pilot_run`](@ref)
"""
function investigate_different_particles(n_times_run_filter,
                                         filter_opt, 
                                         mod_param, 
                                         file_loc, 
                                         model)

    data_obs = CSV.read(file_loc.path_data, DataFrame)
    ind_data = init_ind_data(data_obs, filter_opt)

    particles_use = 10
    times_correct = 0
    while particles_use < 5000

        filter_var = change_filter_opt(filter_opt, particles_use, 0.0)
        filter_use = change_filter_opt(filter_opt, particles_use, filter_opt.rho)

        j = 1
        log_lik_vals::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_times_run_filter)
        rand_num_var = create_rand_num(ind_data, model, filter_var)
        while j < (n_times_run_filter + 1)
            log_lik = run_filter(filter_var, mod_param, rand_num_var, 
                                      model, ind_data)
            if log_lik == -Inf
                @printf("Infinity")
                continue
            end
            log_lik_vals[j] = log_lik
            update_random_numbers!(rand_num_var, filter_var)
            j += 1
        end
  
        est_var = var(log_lik_vals)

        # In case of correlation, calculate target variance 
        if filter_use.rho == 0
            target_var = 2.0

        elseif filter_use.rho != 0
            log_lik_vec1 = Array{FLOAT, 1}(undef, n_times_run_filter)
            log_lik_vec2 = Array{FLOAT, 1}(undef, n_times_run_filter)
            # Run-filter to get correlation 
            j = 1
            while j < n_times_run_filter
                rand_num1 = create_rand_num(ind_data, model, filter_use)
                rand_num2 = deepcopy(rand_num1)
                update_random_numbers!(rand_num2, filter_use)

                # Important to not use variance filter here 
                log_lik1 = run_filter(filter_use, mod_param, rand_num1, 
                                           model, ind_data)
                log_lik2 = run_filter(filter_use, mod_param, rand_num2, 
                                           model, ind_data)

                if log_lik1 == -Inf || log_lik2 == -Inf
                    continue
                end

                log_lik_vec1[j] = log_lik1
                log_lik_vec2[j] = log_lik2
                j += 1
            end

            est_cor = cor(log_lik_vec1, log_lik_vec2)
            target_var = 2.16^2 / (1 - est_cor^2)
        end

        #printf("N_particles = %d, est_var = %.3f, target_var = %.3f\n", particles_use, est_var, target_var)
        if target_var < est_var
            particles_use += 10
            times_correct = 0
        else
            if times_correct > 1
                break 
            else
                times_correct += 1
            end
        end

    end

    return particles_use
end   


"""
tune_particles_single_individual(pilot_run_info, 
                                 mcmc_sampler, 
                                 param_info,
                                 filter, 
                                 model, 
                                 file_loc)

Tune the particles for a provided sampler, filter and model for single individial inference. 

Given sampler, filter, parameter-information (param_info) tunes the number 
of particles for each starting value in the pilot_run_info array by 1) Running a 
pilot-run to get central-posterior location and (2) trying out different number 
of particles at central-posterior location to get an optimal variance of the 
log-likelihood. For each starting values in pilot_run_info, different correlation 
levels can be investigated. 

See also: [`run_pilot_run`, `TuneParticlesIndividual`](@ref)
"""
function tune_particles_single_individual(pilot_run_info, 
                                          mcmc_sampler, 
                                          param_info,
                                          filter, 
                                          model, 
                                          file_loc)
    
    # Get indicies for later data 
    n_param_infer = length(param_info.prior_ind_param) + length(param_info.prior_error_param)
    i_range_ind = 1:length(param_info.prior_ind_param)
    i_range_error = length(param_info.prior_ind_param)+1:n_param_infer

    for i in 1:length(pilot_run_info)
        # Run for the different correlation levels for a specific start-guess 
        pilot_run_data = pilot_run_info[i]
        rho_list = pilot_run_data.rho_list
        n_times_run_filter = pilot_run_data.n_times_run_filter
        for j in 1:length(rho_list)

            # Ensure correct parameters for running the pilot-run 
            filter_pilot = change_filter_opt(filter, pilot_run_data.n_particles_pilot, 
                rho_list[j])
            param_info_pilot = change_param_info(param_info, 
                                                pilot_run_data.init_ind_param, 
                                                pilot_run_data.init_error)

            # Ensure correct file-locations 
            file_loc_use = deepcopy(file_loc)
            calc_dir_save!(file_loc_use, filter_pilot, mcmc_sampler)

            # Run pilot-run only if it has not been run before 
            a, b, c = check_if_pilot_result_exists(file_loc_use, filter_pilot, 
                            pilot_run_data.n_samples, param_info_pilot)
            run_exist, central_pos_values, i_exp = a, b, c
            if run_exist == false
                a, b = run_pilot_run(pilot_run_data.n_samples, mcmc_sampler, 
                            param_info_pilot, model, file_loc_use, filter_pilot) 
                central_pos_values = a
                i_exp = b
            end

            # Map central-posterior to model-parameters 
            data_obs = CSV.read(file_loc.path_data, DataFrame)
            ind_data = init_ind_data(data_obs, filter)
            param_info_find_particles = change_param_info(param_info, 
                                                        central_pos_values[i_range_ind], 
                                                        central_pos_values[i_range_error])
            mod_param = init_model_parameters(param_info_find_particles.init_ind_param, 
                                            param_info_find_particles.init_error_param, 
                                            model, 
                                            covariates=ind_data.cov_val)
            map_proposal_to_model_parameters_ind!(mod_param, central_pos_values, param_info_find_particles, 
                                                  i_range_ind, i_range_error)

            # Investigate the provided number of particles 
            @printf("\nTesting particles for experiment = %d ", i_exp)
            @printf("using a correlation level of = %.3f\n", filter_pilot.rho)

            particles_use = investigate_different_particles(n_times_run_filter, filter_pilot, mod_param, file_loc_use, model)

            @printf("For start-guess %d, particles_use = %d \n", i_exp, particles_use)
            dir_save = file_loc_use.dir_save * "Pilot_runs/Exp_tag" * string(i_exp) * "/"
            file_save = dir_save * "N_particles.tsv"
            if !isdir(dir_save)
                mkdir(dir_save)
            end
            open(file_save, "w") do io
                writedlm(io, [particles_use])
            end
        end
    end
end


"""
    change_start_val_to_pilots(param_info, file_loc, filter; sampler_name="Gen_am_sampler", exp_id=1)

For a sampler, filter and model (file_loc) change starting values to pilot-run values. 

Changes, by creating a new copy of param_info, starting values for mcmc-sampler to 
the central-posterior-location values obtained in pilot-run. Note, sampler-name 
must be Gen_am_sampler, Am_sampler or Rand_walk_sampler.

See also: [`InitParameterInfo`](@ref)
"""
function change_start_val_to_pilots(param_info, file_loc, filter;
     sampler_name="Gen_am_sampler", exp_id=1)

    rho = filter.rho 

    # Ensure that correct pilot run is searched for 
    if rho != 0
        tag_corr = "/Correlated/"
    elseif rho == 0
        tag_corr = "/Not_correlated/"
    end

    if sampler_name == "Gen_am_sampler"
        tag_sampler = "Gen_am_sampler/"
    elseif sampler_name == "Rand_walk_sampler"
        tag_sampler = "Rand_walk_sampler/"
    elseif sampler_name == "Am_sampler"
        tag_sampler = "Am_sampler/"
    elseif sampler_name == "Ram_sampler"
        tag_sampler = "Ram_sampler/"
    else
        @printf("The provided sampler does not exist\n")
        @printf("Available samplers are Gen_am_sampler and Rand_walk_sampler\n")
        return 1 
    end

    dir_file = file_loc.dir_save * tag_corr * tag_sampler * "Pilot_runs/"
    file_data = dir_file * "Central_pos_value.csv"
    if !isfile(file_data)
        @printf("Error: Pilot-run data does not exist for provided sampler and/or correlation level\n")
        return 1
    end

    data = CSV.read(file_data, DataFrame)

    # Filter out-specific correlation level 
    filter!(row -> row[:correlation] == rho, data)
    if size(data)[1] == 0
        @printf("Pilot-run data does not exist for provided rho %.3f\n", rho)
        return 1
    end
    # Filter out correct experiment 
    i_exp_list = unique(data[!, "i_exp"])
    exp_use = i_exp_list[exp_id]
    filter!(row -> row[:i_exp] == exp_use, data)
    param = data[!, "cent_pos_val"]

    # Change initial parameter values 
    n_param = length(param_info.prior_ind_param) + length(param_info.prior_error_param)
    range_ind_param = 1:length(param_info.prior_ind_param)
    range_error_param = length(param_info.prior_ind_param)+1:n_param
    param_info = change_param_info(param_info, param[range_ind_param], 
        param[range_error_param])

    return param_info
end


"""
    init_filter_pilot(filter_opt, file_loc, rho, sampler_name; exp_tag::T=1) where T<:Signed

Initialise a filter with the number of particles obtained from the tuning in the pilot-run. 
"""
function init_filter_pilot(filter_opt, file_loc, rho, sampler_name; exp_tag::T=1) where T<:Signed

    # Ensure that correct pilot run is searched for 
    if rho != 0
        tag_corr = "/Correlated/"
    elseif rho == 0
        tag_corr = "/Not_correlated/"
    end

    if sampler_name == "Gen_am_sampler"
        tag_sampler = "Gen_am_sampler/"
    elseif sampler_name == "Rand_walk_sampler"
        tag_sampler = "Rand_walk_sampler/"
    elseif sampler_name == "Am_sampler"
        tag_sampler = "Am_sampler/"
    elseif sampler_name == "Ram_sampler"
        tag_sampler = "Ram_sampler/"
    else
        @printf("The provided sampler does not exist\n")
        @printf("Available samplers are Gen_am_sampler and Rand_walk_sampler\n")
        return 1 
    end

    dir_file = file_loc.dir_save * tag_corr * tag_sampler * "Pilot_runs/" * "Exp_tag" * string(exp_tag) * "/"
    file_data = dir_file * "N_particles.tsv"
    
    n_particles_use = convert(Int64, readdlm(file_data, '\t', FLOAT, '\n')[1, 1])

    filter_use = change_filter_opt(filter_opt, n_particles_use, rho)
    return filter_use 

end