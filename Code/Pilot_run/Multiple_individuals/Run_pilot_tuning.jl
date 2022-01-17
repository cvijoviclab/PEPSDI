#=
    Function for running pilot-run and tuning particles for Gibbs option 1 and option 2  
=# 


"""
    init_pilot_run_info(pop_param_info::ParamInfoPop; 
                        init_mean="mean", 
                        init_scale="mean", 
                        init_kappa="mean",
                        init_sigma="mean",
                        n_particles_pilot::T1=1000, 
                        n_samples_pilot::T1=1000, 
                        rho_list=[0.0], 
                        n_particles_try=5, 
                        n_times_run_filter::T1=100)::TuneParticlesMixed where T1<:Signed

Initalise TuneParticlesMixed-struct which stores information for pilot-run and tuning of particles. 

# Args
- `n_samples`: number of samples to take from posterior. 
- `pop_param_info`: priors, transformations and initial values for population parameters (η, σ, κ) (see [`init_pop_param_info`](@ref))
- `init_mean`: initial value for mean in η for pilot-run. If median, mode, random or mean is initalised via prior_pop_mean. 
    If array of values of length prior_pop_mean is initalised as provided array (see [`calc_param_init_val`](@ref))
- `init_scale`: see init_mean
- `init_kappa`: see init_mean
- `init_sigma`: see init_mean
- `n_particles_pilot`: number of particles in pilot-run for filter 
- `rho_list`: correlation levels to tune particles for 
- `n_particles_try`: number of starting particles when tuning number of particles
- `n_times_run_filter`: number of times to run filter when tuning particles

See also: [`TuneParticlesMixed`](@ref)
"""

function init_pilot_run_info(pop_param_info::ParamInfoPop; 
                             init_mean="mean", 
                             init_scale="mean", 
                             init_kappa="mean",
                             init_sigma="mean",
                             n_particles_pilot::T1=1000, 
                             n_samples_pilot::T1=1000, 
                             rho_list=[0.0], 
                             n_particles_try=5, 
                             n_times_run_filter::T1=100)::TuneParticlesMixed where T1<:Signed

    init_val_mean::Array{FLOAT, 1} = calc_param_init_val(init_mean, pop_param_info.prior_pop_param_mean)
    init_val_scale::Array{FLOAT, 1} = calc_param_init_val(init_scale, pop_param_info.prior_pop_param_scale)
    init_val_kappa::Array{FLOAT, 1} = calc_param_init_val(init_kappa, pop_param_info.prior_pop_param_kappa)
    init_val_sigma::Array{FLOAT, 1} = calc_param_init_val(init_sigma, pop_param_info.prior_pop_param_sigma)

    rho_list_use::Array{FLOAT, 1} = convert(Array{FLOAT, 1}, rho_list)
    tune_part_data = TuneParticlesMixed(n_particles_pilot, n_samples_pilot, 
        init_val_mean, init_val_scale, init_val_kappa, init_val_sigma, n_particles_try, 
        n_times_run_filter, rho_list)
    
    return tune_part_data
end


"""
    save_pilot_result(res_chains::ChainsMixed, 
                      mcmc_sampler_ind_arr, 
                      mcmc_sampler_pop, 
                      file_loc::FileLocations, 
                      tune_part_data::TuneParticlesMixed)

Save pilot-run-results (central-posterior position for parameters and mcmc-samplers options) to disk. 

Each pilot-run is assigned an integer experiment id (exp_id) when written to file.
The last 20% of samples from the pilot-run are used to compute central posterior locaiton. 
Chains and mcmc-sampler (indiviudal array and kappa-sigma) are outputed by Gibbs-sampler. 
"""
function save_pilot_result(res_chains::ChainsMixed, 
                           mcmc_sampler_ind_arr, 
                           mcmc_sampler_pop, 
                           file_loc::FileLocations, 
                           tune_part_data::TuneParticlesMixed;
                           sampler::String="standard")

    # Nested function for writing result to disk 
    function write_file(data_val, col_names, file_save)
        data_write = convert(DataFrame, data_val)
        rename!(data_write, col_names)
        CSV.write(file_save, data_write)
    end

    n_ind_param = size(mcmc_sampler_ind_arr[1].cov_mat)[1]

    # Save pilot run depending on standard or alternative sampler 
    if sampler == "standard"
        dir_save_pilot = file_loc.dir_save * "/Pilot_run_data/"
    elseif sampler == "alt"
        dir_save_pilot = file_loc.dir_save * "/Pilot_run_data_alt/"
    else
        @printf("Error: Non standard solver used")
    end

    if !isdir(dir_save_pilot)
        mkpath(dir_save_pilot)
    end
    file_info = dir_save_pilot * "Pilot_info.csv"

    # Get the experimental tag for this pilot-run 
    if !isfile(file_info)
        tag_exp = 1
        append_data = false
    else
        data = CSV.read(file_info, DataFrame)
        tag_exp = maximum(data[!, "id"]) + 1
        append_data = true
    end

    # Create header-columns for the file 
    n_mean = length(tune_part_data.init_mean)
    name_mean = Array{String, 1}(undef, n_mean)
    name_scale = deepcopy(name_mean)
    name_ind = Array{String, 1}(undef, n_ind_param)
    for i in 1:n_mean
        name_mean[i] = "mean" * string(i)
        name_scale[i] = "scale" * string(i)
    end
    for i in 1:n_ind_param
        name_ind[i] = "c" * string(i)
    end
    n_kappa = length(tune_part_data.init_kappa)
    name_kappa = Array{String, 1}(undef, n_kappa)
    for i in 1:n_kappa
        name_kappa[i] = "kappa" * string(i)
    end
    n_sigma = length(tune_part_data.init_sigma)
    name_sigma = Array{String, 1}(undef, n_sigma)
    for i in 1:n_sigma
        name_sigma[i] = "sigma" * string(i)
    end
    name_col = vcat(name_mean, name_scale, name_kappa, name_sigma, ["n_samples", "n_particles"],  ["id"])

    # Write information data to disk 
    data_val = vcat(tune_part_data.init_mean, 
                    tune_part_data.init_scale, 
                    tune_part_data.init_kappa, 
                    tune_part_data.init_sigma,
                    [tune_part_data.n_samples_pilot], 
                    [tune_part_data.n_particles_pilot], 
                    [tag_exp])

    data_save = convert(DataFrame, data_val')
    rename!(data_save, name_col)
    CSV.write(file_info, data_save, append=append_data)

    # Calculate central-posterior values 
    samples_use = tune_part_data.n_samples_pilot 
    mean_cent_pos = mean(res_chains.mean[:, samples_use], dims = 2)
    scale_cent_pos = mean(res_chains.scale[:, samples_use], dims = 2)
    kappa_sigma_cent_pos = mean(res_chains.kappa_sigma[:, samples_use], dims = 2)
    corr_cent_pos = mean(res_chains.corr[:, :, samples_use], dims = 3)
    # Central posterior values for each individual
    n_individuals = length(res_chains.ind_param)
    ind_param_cent_pos = Array{FLOAT, 2}(undef, (n_individuals, n_ind_param+1))
    for i in 1:n_individuals
        cent_pos_ind_i = mean(res_chains.ind_param[i][:, samples_use], dims=2)
        ind_param_cent_pos[i, 1:end-1] = cent_pos_ind_i
        ind_param_cent_pos[i, end] = i
    end

    # Write central posterior values 
    dir_save_pilot *= "Exp_tag" * string(convert(Int64, tag_exp)) * "/"
    if !isdir(dir_save_pilot)
        mkpath(dir_save_pilot)
    end
    file_ind = dir_save_pilot * "Ind.csv"
    file_mean = dir_save_pilot * "Mean.csv"
    file_scale = dir_save_pilot * "Scale.csv"
    file_corr = dir_save_pilot * "Corr.csv"
    file_kappa_sigma = dir_save_pilot * "Kappa_sigma.csv"
    write_file(ind_param_cent_pos, vcat(name_ind, ["id"]), file_ind)
    write_file(kappa_sigma_cent_pos', vcat(name_kappa, name_sigma), file_kappa_sigma)
    write_file(scale_cent_pos', name_scale, file_scale)
    write_file(mean_cent_pos', name_mean, file_mean)
    write_file(corr_cent_pos[:, :, 1], name_ind, file_corr)

    # Write mcmc-samplers 
    dir_save_mcmc = dir_save_pilot * "Mcmc_param/"
    if !isdir(dir_save_mcmc)
        mkpath(dir_save_mcmc)
    end
    for i in 1:n_individuals
        ind_id = string(i)
        exp_tag = string(tag_exp)
        save_mcmc_opt_pilot(mcmc_sampler_ind_arr[i], file_loc, exp_tag, ind_id, dir_save=dir_save_mcmc)
    end
    # Kappa_sigma, for alternative sampler an entire array is saved. For standard gibbs single sampler is saved 
    if sampler == "standard"
        ind_id = "Kappa_sigma"
        exp_tag = string(tag_exp)
        save_mcmc_opt_pilot(mcmc_sampler_pop, file_loc, exp_tag, ind_id, dir_save=dir_save_mcmc)
    
    elseif sampler == "alt"
        for i in 1:n_individuals
            ind_id = "Kappa_sigma" * string(i)
            exp_tag = string(tag_exp)
            save_mcmc_opt_pilot(mcmc_sampler_pop[i], file_loc, exp_tag, ind_id, dir_save=dir_save_mcmc)
        end
    end


    return convert(Int64, tag_exp)
end


"""
    run_pilot_run(n_samples, 
                  pop_param_info::ParamInfoPop, 
                  ind_param_info::ParamInfoIndPre, 
                  file_loc::FileLocations, 
                  model, 
                  filter_opt, 
                  mcmc_sampler_ind, 
                  mcmc_sampler_pop, 
                  pop_sampler_opt, 
                  tune_part_data::TuneParticlesMixed)

Perform pilot-run with n_samples to obtain central-posterior location and write said location to disk with unique exp_id. 
"""
function run_pilot_run(n_samples, 
                       pop_param_info::ParamInfoPop, 
                       ind_param_info::ParamInfoIndPre, 
                       file_loc::FileLocations, 
                       model, 
                       filter_opt, 
                       mcmc_sampler_ind, 
                       mcmc_sampler_pop, 
                       pop_sampler_opt, 
                       tune_part_data::TuneParticlesMixed)


    x1, x2, x3 = run_PEPSDI_opt1(n_samples, pop_param_info, ind_param_info, file_loc, 
        model, filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, 
        pilot=true)

    res_chains = x1
    mcmc_sampler_ind_arr = x2
    mcmc_sampler_pop = x3

    exp_tag = save_pilot_result(res_chains, mcmc_sampler_ind_arr, mcmc_sampler_pop, 
        file_loc, tune_part_data)

    return exp_tag
end
"""
Use alternative sampler where kappa and sigma are treated as parameters with 
small variability. 
"""
function run_pilot_run(n_samples, 
                       pop_param_info::ParamInfoPop, 
                       ind_param_info::ParamInfoIndPre, 
                       file_loc::FileLocations, 
                       model, 
                       filter_opt, 
                       mcmc_sampler_ind, 
                       mcmc_sampler_pop, 
                       pop_sampler_opt, 
                       kappa_sigma_sampler_opt,
                       tune_part_data::TuneParticlesMixed)


    x1, x2, x3 = run_PEPSDI_opt2(n_samples, pop_param_info, ind_param_info, file_loc, model, 
                     filter_opt, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, 
                     kappa_sigma_sampler_opt, pilot=true)
    
    res_chains = x1
    mcmc_sampler_ind_arr = x2
    mcmc_sampler_pop_arr = x3

    exp_tag = save_pilot_result(res_chains, mcmc_sampler_ind_arr, mcmc_sampler_pop_arr, 
        file_loc, tune_part_data, sampler="alt")

    return exp_tag
end


"""
    check_if_pilot_exist(tune_part_data::TuneParticlesMixed, file_loc::FileLocations)

Check if pilot run (with specific number of samples, particles and initial values provided in tune_part_data) exists. 
"""
function check_if_pilot_exist(tune_part_data::TuneParticlesMixed, file_loc::FileLocations; sampler::String="standard")

    # Check if information file-exists 
    if sampler == "standard"
        dir_save_pilot = file_loc.dir_save * "/Pilot_run_data/"
        if !isdir(dir_save_pilot)
            return false, 0
        end
        file_info = dir_save_pilot * "Pilot_info.csv"
        if !isfile(file_info)
            return false, 0
        end

    elseif sampler == "alt"
        dir_save_pilot = file_loc.dir_save * "/Pilot_run_data_alt/"
        if !isdir(dir_save_pilot)
            return false, 0
        end
        file_info = dir_save_pilot * "Pilot_info.csv"
        if !isfile(file_info)
            return false, 0
        end

    else
        println("Sampler-type for pilot must be either standrad or alt")
    end

    data_info = convert(Array{Real, 2}, CSV.read(file_info, DataFrame))
    n_row_file = size(data_info)[1]
    data_ind_vec = vcat(tune_part_data.init_mean, 
                        tune_part_data.init_scale, 
                        tune_part_data.init_kappa, 
                        tune_part_data.init_sigma, 
                        [tune_part_data.n_samples_pilot], 
                        [tune_part_data.n_particles_pilot])


    for i in 1:n_row_file
        if sum(.!(data_info[i, 1:end-1] .== data_ind_vec)) == 0
            return true, convert(Int64, data_info[i, end])
        end
    end

    return false, 0
end


"""
    tune_particles_mixed(n_individuals::T,    
                         exp_id::T,
                         pop_param_info::ParamInfoPop, 
                         ind_param_info::ParamInfoIndPre, 
                         model::SdeModel, 
                         pop_sampler_opt, 
                         filter_opt,
                         file_loc::FileLocations, 
                         tune_part_data::TuneParticlesMixed) where T<:Signed

Based on pilot-run with exp_id tunes the number of particles for correlation level in tune_part_data. 

Particles are for all individual. For rho = 0 target filter variance of global likelihood is less than 2. Filter variance is 
obtained by repedetly running filter for number of times provided by tune_part_data. For rho > 0 target 
variance is 2.16^2 / (1 - est_cor^2), where estimated correlation is obtained by running two paralell filters 
launched from the same random numbers. Note, here the global likelihood (product of individual likelihoods) is used as 
tuning target. 
"""
function tune_particles_mixed(n_individuals::T,    
                              exp_id::T,
                              pop_param_info::ParamInfoPop, 
                              ind_param_info::ParamInfoIndPre, 
                              model,
                              pop_sampler_opt, 
                              filter_opt,
                              file_loc::FileLocations, 
                              tune_part_data::TuneParticlesMixed) where T<:Signed

    # Dimension required for allocating arrays, and correlation to try 
    dim_mean = length(pop_param_info.prior_pop_param_mean)
    rho = filter_opt.rho

    pop_sampler = init_pop_sampler(pop_sampler_opt, n_individuals, dim_mean)
    pop_par_tmp = init_pop_param_curr(pop_param_info)
    dist_ind = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)

    # Take initial parameter values from pilot runs 
    ind_data_arr = init_ind_data_arr(file_loc, filter_opt)
    pop_param_info_pilot, ind_param_info_arr = init_param_pilot(pop_param_info, 
        ind_param_info, file_loc, dist_ind, exp_id)
    mod_param_arr = init_model_parameters_arr(ind_param_info_arr, pop_param_info_pilot, 
        model, n_individuals, ind_data_arr)

    # Current model parameters not transformed
    ind_param_current = Array{FLOAT, 2}(undef, (dim_mean, n_individuals))
    for i in 1:n_individuals
        ind_param_current[:, i] .= ind_param_info_arr[i].init_ind_param
    end
    # Current kappa-sigma not transformed
    kappa_sigma_current = vcat(pop_param_info_pilot.init_pop_param_kappa, pop_param_info_pilot.init_pop_param_sigma)
    log_kappa::Array{Bool, 1} = pop_param_info.log_pop_param_kappa
    log_sigma::Array{Bool, 1} = pop_param_info.log_pop_param_sigma
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:length(kappa_sigma_current)]
    kappa_index = .!sigma_index

    # Ensures correct transformation
    @simd for j in 1:n_individuals
        @inbounds map_sigma_to_mod_param_mixed!(mod_param_arr[j], log_sigma, (@view kappa_sigma_current[sigma_index]))
    end
    @simd for j in 1:n_individuals
        @inbounds map_kappa_to_mod_param_mixed!(mod_param_arr[j], log_kappa, (@view kappa_sigma_current[kappa_index]))
    end
    @simd for j in 1:n_individuals
        @inbounds update_ind_param_ind!(mod_param_arr[j].individual_parameters, ind_param_current[:, j], 
             ind_param_info_arr[j].log_ind_param)
    end

    # Array with particle filters (for parallel computations)
    filter_arr = Array{typeof(filter_opt), 1}(undef, n_individuals)
    for i in 1:n_individuals
        filter_arr[i] = deepcopy(filter_opt)
    end

    # Array with model (for parallel computations)
    model_arr = Array{typeof(model), 1}(undef, n_individuals)
    for i in 1:n_individuals
        model_arr[i] = deepcopy(model)
    end

    # Pre-allocate arrays for tuning particle data 
    log_lik_var = Array{FLOAT, 1}(undef, n_individuals)
    log_lik_target = Array{FLOAT, 1}(undef, n_individuals)
    n_particles = Array{Int64, 1}(undef, n_individuals)

    # Break when variance reaches target two-times 
    n_times_var_correct = 0
    if rho != 0
        n_particles_max = 5000
    else
        n_particles_max = 5000
    end

    times_run_filter = tune_part_data.n_times_run_filter
    # Filter for variance 
    n_part_use = tune_part_data.n_particles_try
    println(n_part_use)
    n_part_use = 20

    n_particles::Array{Int64, 1} = Array{Int64, 1}(undef, n_individuals)

    # Get number of particles per individual
    while n_part_use < n_particles_max

        log_lik_vals = Array{FLOAT, 1}(undef, times_run_filter)

        # Run filter provided number of times 
        for j in 1:times_run_filter

            # In parallel over individuals 
            filter_var_arr = [change_filter_opt(filter_arr[k], n_part_use, 0.0) for k in 1:n_individuals]
            rand_num_arr = [create_rand_num(ind_data_arr[k], model_arr[k], filter_var_arr[k]) for k in 1:n_individuals]
            log_lik_ind = Array{FLOAT, 1}(undef, n_individuals)

            Threads.@threads for k in 1:n_individuals    
            
                log_lik_ind[k] = run_filter(filter_var_arr[k], mod_param_arr[k], rand_num_arr[k], 
                                            model_arr[k], ind_data_arr[k])

                update_random_numbers!(rand_num_arr[k], filter_var_arr[k])
            end

            log_lik_vals[j] = sum(log_lik_ind)
        end

        est_var = var(log_lik_vals)
        if isnan(est_var)
            est_var = Inf
        end

        # Obtain target variance depending on correlation or not 
        if rho == 0
            target_var = 2.0
            if n_part_use > 100
                increment_particles = 50    
            elseif n_part_use > 1000
                increment_particles = 250
            else 
                increment_particles = 10
            end
        else
            
            log_lik_vec1 = Array{FLOAT, 1}(undef, times_run_filter)
            log_lik_vec2 = Array{FLOAT, 1}(undef, times_run_filter)
            for j in 1:times_run_filter

                filter_use_arr = [change_filter_opt(filter_arr[k], n_part_use, rho) for k in 1:n_individuals]    

                log_lik_ind1::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)
                log_lik_ind2::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)
                Threads.@threads for k in 1:n_individuals
                    rand_num1 = create_rand_num(ind_data_arr[k], model_arr[k], filter_use_arr[k])
                    rand_num2 = deepcopy(rand_num1)
                    update_random_numbers!(rand_num2, rand_num1, filter_use_arr[k])

                    # Important to not use variance filter here 
                    log_lik_ind1[k] = run_filter(filter_use_arr[k], mod_param_arr[k], rand_num1, 
                                          model_arr[k], ind_data_arr[k])
                    log_lik_ind2[k] = run_filter(filter_use_arr[k], mod_param_arr[k], rand_num2, 
                                            model_arr[k], ind_data_arr[k])
                end
                log_lik_vec1[j] = sum(log_lik_ind1)
                log_lik_vec2[j] = sum(log_lik_ind2)
            end

            est_cor = cor(log_lik_vec1, log_lik_vec2)
            target_var = 2.16^2 / (1 - est_cor^2)
            
            if n_part_use > 100
                increment_particles = 50    
            elseif n_part_use > 1000
                increment_particles = 250
            else 
                increment_particles = 10
            end

        end

        @printf("Est_var = %.3f, target_var = %.3f\n", est_var, target_var)

        # Terminate if target variance has been reached two-times 
        if est_var > target_var || (abs(est_var) == Inf || abs(target_var) == Inf)
            n_part_use += increment_particles
            n_times_var_correct = 0
        else 
            n_times_var_correct += 1
            if n_times_var_correct > 1
                break 
            end
        end
    end


    n_particles .= n_part_use

    #n_particles .= convert(Int64, round(median(n_particles)))
    println(n_particles)
    n_particles .= maximum(n_particles)
    #n_particles .= 1000

    return n_particles
end
"""
Method for alternative sampler. Here the target is to keep the likelihood
under the limit for each individual -> likelly fewer particles 
are required. 
"""
function tune_particles_mixed(n_individuals::T,    
                              exp_id::T,
                              pop_param_info::ParamInfoPop, 
                              ind_param_info::ParamInfoIndPre, 
                              model,
                              pop_sampler_opt, 
                              kappa_sigma_sampler_opt, 
                              filter_opt,
                              file_loc::FileLocations, 
                              tune_part_data::TuneParticlesMixed) where T<:Signed

    # Dimension required for allocating arrays, and correlation to try 
    dim_mean = length(pop_param_info.prior_pop_param_mean)
    rho = filter_opt.rho

    n_ind_param = ind_param_info.n_param

    # Population parameter sampler 
    pop_sampler = init_pop_sampler(pop_sampler_opt, n_individuals, dim_mean)
    pop_par_tmp = init_pop_param_curr(pop_param_info)
    

    # Kappa-sigma parameters sampler 
    n_kappa_sigma = length(pop_param_info.prior_pop_param_kappa) + length(pop_param_info.prior_pop_param_sigma)
    kappa_sigma_sampler = init_kappa_sigma_sampler(kappa_sigma_sampler_opt, n_kappa_sigma)

    # Take initial parameter values from pilot runs 
    dist_ind = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)

    ind_data_arr = init_ind_data_arr(file_loc, filter_opt)
    pop_param_info_pilot, ind_param_info_arr = init_param_pilot(pop_param_info, 
        ind_param_info, file_loc, dist_ind, exp_id, sampler="alt")
    mod_param_arr = init_model_parameters_arr(ind_param_info_arr, pop_param_info_pilot, 
        model, n_individuals, ind_data_arr)

    # Current model parameters not transformed
    ind_param_current = Array{FLOAT, 2}(undef, (n_ind_param, n_individuals))
    for i in 1:n_individuals
        ind_param_current[:, i] .= ind_param_info_arr[i].init_ind_param
    end
    # Current kappa-sigma not transformed
    kappa_sigma_current = vcat(pop_param_info_pilot.init_pop_param_kappa, pop_param_info_pilot.init_pop_param_sigma)
    log_kappa::Array{Bool, 1} = pop_param_info.log_pop_param_kappa
    log_sigma::Array{Bool, 1} = pop_param_info.log_pop_param_sigma
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:length(kappa_sigma_current)]
    kappa_index = .!sigma_index

    # Ensures correct transformation
    @simd for j in 1:n_individuals
        @inbounds map_sigma_to_mod_param_mixed!(mod_param_arr[j], log_sigma, (@view kappa_sigma_current[sigma_index]))
    end
    @simd for j in 1:n_individuals
        @inbounds map_kappa_to_mod_param_mixed!(mod_param_arr[j], log_kappa, (@view kappa_sigma_current[kappa_index]))
    end
    @simd for j in 1:n_individuals
        @inbounds update_ind_param_ind!(mod_param_arr[j].individual_parameters, ind_param_current[:, j], 
             ind_param_info_arr[j].log_ind_param)
    end

    # Array with particle filters (for parallel computations)
    filter_arr = Array{typeof(filter_opt), 1}(undef, n_individuals)
    for i in 1:n_individuals
        filter_arr[i] = deepcopy(filter_opt)
    end

    # Array with model (for parallel computations)
    model_arr = Array{typeof(model), 1}(undef, n_individuals)
    for i in 1:n_individuals
        model_arr[i] = deepcopy(model)
    end

    # Pre-allocate arrays for tuning particle data 
    log_lik_var = Array{FLOAT, 1}(undef, n_individuals)
    log_lik_target = Array{FLOAT, 1}(undef, n_individuals)
    n_particles = Array{Int64, 1}(undef, n_individuals)

    # Get number of particles per individual
    Threads.@threads for i in 1:n_individuals
        
        # Filter for variance 
        n_part_use = tune_part_data.n_particles_try
        n_part_use = 10
        times_run_filter = tune_part_data.n_times_run_filter
        log_lik_vals = Array{FLOAT, 1}(undef, times_run_filter)

        # Break when variance reaches target two-times 
        n_times_var_correct = 0
        if rho != 0
            n_particles_max = 2000
        else
            n_particles_max = 5000
        end

        while n_part_use < n_particles_max

            # Obtain variance of log-likelihood
            filter_var = change_filter_opt(filter_arr[i], n_part_use, 0.0)
            rand_num_var = create_rand_num(ind_data_arr[i], model_arr[i], filter_var)
            for j in 1:times_run_filter
                log_lik = run_filter(filter_var, mod_param_arr[i], rand_num_var, 
                                        model_arr[i], ind_data_arr[i])
                log_lik_vals[j] = log_lik
                        
                update_random_numbers!(rand_num_var, filter_var)
            end
            est_var = var(log_lik_vals)

            # Obtain target variance depending on correlation or not 
            if rho == 0
                target_var = 2.0

                if n_part_use > 100
                    increment_particles = 50    
                elseif n_part_use > 1000
                    increment_particles = 250
                else 
                    increment_particles = 10
                end

            else
                filter_use = change_filter_opt(filter_arr[i], n_part_use, rho)
                log_lik_vec1 = Array{FLOAT, 1}(undef, times_run_filter)
                log_lik_vec2 = Array{FLOAT, 1}(undef, times_run_filter)
                # Run-filter to get correlation 
                for j in 1:times_run_filter
                    rand_num1 = create_rand_num(ind_data_arr[i], model_arr[i], filter_use)
                    rand_num2 = deepcopy(rand_num1)
                    update_random_numbers!(rand_num2, rand_num1, filter_use)

                    # Important to not use variance filter here 
                    log_lik1 = run_filter(filter_use, mod_param_arr[i], rand_num1, 
                                            model_arr[i], ind_data_arr[i])
                    log_lik2 = run_filter(filter_use, mod_param_arr[i], rand_num2, 
                                            model_arr[i], ind_data_arr[i])
                    log_lik_vec1[j] = log_lik1
                    log_lik_vec2[j] = log_lik2
                end

                est_cor = cor(log_lik_vec1, log_lik_vec2)
                target_var = 2.16^2 / (1 - est_cor^2)

                if n_part_use > 100
                    increment_particles = 50    
                elseif n_part_use > 1000
                    increment_particles = 250
                else 
                    increment_particles = 10
                end
            end

            # Terminate if target variance has been reached two-times 
            if est_var > target_var || (abs(est_var) == Inf || abs(target_var) == Inf)
                n_part_use += increment_particles
                n_times_var_correct = 0
            else 
                n_times_var_correct += 1
                if n_times_var_correct > 1
                    n_particles[i] = n_part_use
                    break 
                end
            end
        end

        # Report particles if reaching maximum 
        if n_part_use >= n_particles_max
            n_particles[i] = n_part_use
        end

    end

    println(n_particles)
    
    return n_particles
end


"""
    save_tune_particle_data(n_particles, file_loc, rho, n_individuals, exp_id)

Save number of particles obtained form tuning to disk. 
"""
function save_tune_particle_data(n_particles, file_loc, rho, n_individuals, exp_id; sampler::String="standard")

    if sampler == "standard"
        dir_save_tuning = file_loc.dir_save * "/Pilot_run_data/Exp_tag" * string(exp_id) * "/Tune_particles/"
    elseif sampler == "alt"
        dir_save_tuning = file_loc.dir_save * "/Pilot_run_data_alt/Exp_tag" * string(exp_id) * "/Tune_particles/"
    end

    if !isdir(dir_save_tuning)
        mkpath(dir_save_tuning)
    end
    file_save = dir_save_tuning * "/Rho" * replace(string(rho), "." => "d") * ".csv"
    data_save = convert(DataFrame, n_particles')
    rename!(data_save, "Ind" .* string.(1:n_individuals))
    CSV.write(file_save, data_save)
end


"""
    tune_particles_opt1(tune_part_data::TuneParticlesMixed, 
                        pop_param_info::ParamInfoPop, 
                        ind_param_info::ParamInfoIndPre, 
                        file_loc::FileLocations,
                        model::SdeModel,
                        filter_opt, 
                        mcmc_sampler_ind, 
                        mcmc_sampler_pop, 
                        pop_sampler_opt)

Given samplers, filter, parameter-information (e.g transformations) tunes the number 
of particles for each starting value in the pilot_run_info array by 1) Running a 
pilot-run to get central-posterior location and (2) trying out different number 
of particles at central-posterior location to get an optimal variance of the 
log-likelihood. For each starting values in pilot_run_info, different correlation 
levels can be investigated. Pilot-run is not run if pilot-run exits for certain 
number of particles, starting values and number of samples. 

# Args
- `tune_part_data`: pilot-run (samples and particles) and tuning (times run filter) data for tuning (see [`init_pilot_run`](@ref))
- `pop_param_info`: priors, transformations and initial values for population parameters (η, σ, κ) (see [`init_pop_param_info`](@ref))
- `ind_param_info`: initial values and transformation for individual parameters (Φ_i) (see [`init_ind_param_info`](@ref))
- `file_loc`: directory to data and directory for storing result (see [`init_file_loc`](@ref))
- `model`: user provided model (currently SdeModel only supported)
- `filter_opt`: particle filter-options (correlation leven, number of particles and step-length (see [`init_filter`](@ref))
- `mcmc_sampler_ind`: mcmc-sampler (AM, GenAM or RAM) to use for Φ_i (see [`init_mcmc`](@ref))
- `mcmc_sampler_pop`: mcmc-sampler (AM, GenAM or RAM) to use for (σ, κ) (see [`init_mcmc`](@ref))
- `pop_sampler_opt`: NUTS-options for η-sampler (see [`init_pop_sampler_opt`](@ref))

See also: [`TuneParticlesMixed`](@ref)
"""
function tune_particles_opt1(tune_part_data::TuneParticlesMixed, 
                             pop_param_info::ParamInfoPop, 
                             ind_param_info::ParamInfoIndPre, 
                             file_loc::FileLocations,
                             model,
                             filter_opt, 
                             mcmc_sampler_ind, 
                             mcmc_sampler_pop, 
                             pop_sampler_opt)

    @printf("Starting tuning of particles\n")

    # Calculate number of individuals 
    ind_data_arr = init_ind_data_arr(file_loc, filter_opt)
    n_individuals = length(ind_data_arr)

    # Ensure correct file_locaiton 
    file_loc = deepcopy(file_loc)
    file_loc.dir_save *= "/" * mcmc_sampler_ind.name_sampler 

    # Run pilot run if required 
    filter_pilot = change_filter_opt(filter_opt, tune_part_data.n_particles_pilot, filter_opt.rho)
    pop_param_info_pilot = change_pop_par_info(tune_part_data, pop_param_info)

    # Run pilot-run
    pilot_exist, exp_id = check_if_pilot_exist(tune_part_data, file_loc)
    if !pilot_exist
        @printf("Running pilot run\n")
        exp_id = run_pilot_run(tune_part_data.n_samples_pilot, deepcopy(pop_param_info_pilot), ind_param_info, file_loc,
                    model, filter_pilot, mcmc_sampler_ind, mcmc_sampler_pop, pop_sampler_opt, tune_part_data)
    else
        @printf("Pilot run already exists")
    end

    # Tune the particles 
    for rho in tune_part_data.rho_list
        @printf("Tuning particles for rho = %.3f\n", rho)
        filter_tune = change_filter_opt(filter_opt, 10, rho)
        n_particles = tune_particles_mixed(n_individuals, exp_id, pop_param_info, ind_param_info, 
                                           model, pop_sampler_opt, filter_tune, file_loc, tune_part_data)
        save_tune_particle_data(n_particles, file_loc, rho, n_individuals, exp_id)
    end

    @printf("Done\n")
end


"""
    tune_particles_opt2(tune_part_data::TuneParticlesMixed, 
                        pop_param_info::ParamInfoPop, 
                        ind_param_info::ParamInfoIndPre, 
                        file_loc::FileLocations,
                        model,
                        filter_opt, 
                        mcmc_sampler_ind, 
                        mcmc_sampler_pop, 
                        pop_sampler_opt, 
                        kappa_sigma_sampler_opt)

Given samplers, filter, parameter-information (e.g transformations) tunes the number 
of particles for each starting value in the pilot_run_info array by 1) Running a 
pilot-run to get central-posterior location and (2) trying out different number 
of particles at central-posterior location to get an optimal variance of the 
log-likelihood. For each starting values in pilot_run_info, different correlation 
levels can be investigated. Pilot-run is not run if pilot-run exits for certain 
number of particles, starting values and number of samples. Here, in contrast 
to tune_particles an alternative sampler is used, where the gibbs stage 2 step 
is side-stepped by allowing the kappa-sigma parameters to have a small variance. 

# Args
- `tune_part_data`: pilot-run (samples and particles) and tuning (times run filter) data for tuning (see [`init_pilot_run`](@ref))
- `pop_param_info`: priors, transformations and initial values for population parameters (η, σ, κ) (see [`init_pop_param_info`](@ref))
- `ind_param_info`: initial values and transformation for individual parameters (Φ_i) (see [`init_ind_param_info`](@ref))
- `file_loc`: directory to data and directory for storing result (see [`init_file_loc`](@ref))
- `model`: user provided model (currently SdeModel only supported)
- `filter_opt`: particle filter-options (correlation leven, number of particles and step-length (see [`init_filter`](@ref))
- `mcmc_sampler_ind`: mcmc-sampler (AM, GenAM or RAM) to use for Φ_i (see [`init_mcmc`](@ref))
- `mcmc_sampler_pop`: mcmc-sampler (AM, GenAM or RAM) to use for (σ, κ) (see [`init_mcmc`](@ref))
- `pop_sampler_opt`: NUTS-options for η-sampler (see [`init_pop_sampler_opt`](@ref))
- `pop_sampler_opt`: NUTS-options for kappa-sigma-sampler (see [`init_kappa_sigma_sampler_opt`](@ref))

See also: [`TuneParticlesMixed`](@ref)
"""
function tune_particles_opt2(tune_part_data::TuneParticlesMixed, 
                            pop_param_info::ParamInfoPop, 
                            ind_param_info::ParamInfoIndPre, 
                            file_loc::FileLocations,
                            model,
                            filter_opt, 
                            mcmc_sampler_ind, 
                            mcmc_sampler_pop, 
                            pop_sampler_opt, 
                            kappa_sigma_sampler_opt)

    @printf("Starting tuning of particles using alternative sampler\n")

    # Calculate number of individuals 
    ind_data_arr = init_ind_data_arr(file_loc, filter_opt)
    n_individuals = length(ind_data_arr)

    # Ensure correct file_locaiton 
    file_loc = deepcopy(file_loc)
    file_loc.dir_save *= "/" * mcmc_sampler_ind.name_sampler 

    # Run pilot run if required 
    filter_pilot = change_filter_opt(filter_opt, tune_part_data.n_particles_pilot, filter_opt.rho)
    pop_param_info_pilot = change_pop_par_info(tune_part_data, pop_param_info)

    # Run pilot-run
    pilot_exist, exp_id = check_if_pilot_exist(tune_part_data, file_loc, sampler="alt")

    if !pilot_exist
        @printf("Running pilot run\n")
        exp_id = run_pilot_run(tune_part_data.n_samples_pilot, 
                               deepcopy(pop_param_info_pilot), 
                               ind_param_info, file_loc,
                               model, 
                               filter_pilot, 
                               mcmc_sampler_ind, 
                               mcmc_sampler_pop, 
                               pop_sampler_opt, 
                               kappa_sigma_sampler_opt,
                               tune_part_data)
    else
        @printf("Pilot run already exists")
    end

    # Tune the particles 
    for rho in tune_part_data.rho_list
        @printf("Tuning particles for rho = %.3f\n", rho)
        filter_tune = change_filter_opt(filter_opt, 10, rho)
        n_particles = tune_particles_mixed(n_individuals, 
                                           exp_id, 
                                           pop_param_info, 
                                           ind_param_info, 
                                           model, 
                                           pop_sampler_opt, 
                                           kappa_sigma_sampler_opt, 
                                           filter_tune, 
                                           file_loc, 
                                           tune_part_data)

        save_tune_particle_data(n_particles, file_loc, rho, n_individuals, exp_id, sampler="alt")
    end

    @printf("Done\n")
end
