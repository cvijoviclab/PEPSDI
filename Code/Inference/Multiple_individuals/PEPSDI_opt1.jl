"""
    run_PEPSDI_opt1(n_samples::T1, 
                    pop_param_info::ParamInfoPop, 
                    ind_param_info::ParamInfoIndPre, 
                    file_loc::FileLocations,
                    model,
                    filter_opt, 
                    mcmc_sampler_ind, 
                    mcmc_sampler_pop, 
                    pop_sampler_opt;
                    pilot::Bool=false, 
                    pilot_id=0) where T1<:Signed

Take n-samples of η, σ, κ and Φ_i using Gibbs-sampler for user provided state-space mixed effects model. 

Currently AM, GenAm and RAM-sampler are available for proposing σ, κ and Φ_i. For η LKJ-normal and 
ornstein-mixed parameterisation are available. However, users can write own modules for η-parameterisation.  

# Args
- `n_samples`: number of samples to take from posterior. 
- `pop_param_info`: priors, transformations and initial values for population parameters (η, σ, κ) (see [`init_pop_param_info`](@ref))
- `ind_param_info`: initial values and transformation for individual parameters (Φ_i) (see [`init_ind_param_info`](@ref))
- `file_loc`: directory to data and directory for storing result (see [`init_file_loc`](@ref))
- `model`: user provided model (currently SdeModel only supported)
- `filter_opt`: particle filter-options (correlation leven, number of particles and step-length (see [`init_filter`](@ref))
- `mcmc_sampler_ind`: mcmc-sampler (AM, GenAM or RAM) to use for Φ_i (see [`init_mcmc`](@ref))
- `mcmc_sampler_pop`: mcmc-sampler (AM, GenAM or RAM) to use for (σ, κ) (see [`init_mcmc`](@ref))
- `mcmc_sampler_pop`: mcmc-sampler (AM, GenAM or RAM) to use for (σ, κ) (see [`init_mcmc`](@ref))
- `pop_sampler_opt`: NUTS-options for η-sampler (see [`init_pop_sampler_opt`](@ref))
- `pilot`: if true result is not written to disk as pilot-run is assumed. 
- `pilot_id`: if non-zero uses pilot-run data with tag pilot_id (integer) to set initial-values for mcmc-samplers and (η, σ, κ and Φ_i), 
    and using number of particles obtained from particle-tuning (see [`tune_particles_mixed`](@ref))
"""
function run_PEPSDI_opt1(n_samples::T1, 
                         pop_param_info::ParamInfoPop, 
                         ind_param_info::ParamInfoIndPre, 
                         file_loc::FileLocations,
                         model,
                         filter_opt, 
                         mcmc_sampler_ind, 
                         mcmc_sampler_pop, 
                         pop_sampler_opt;
                         pilot::Bool=false, 
                         pilot_id=0, 
                         benchmark=false) where T1<:Signed

    # Update the individual parameters and random numbers Note, the mcmc-samplers are also 
    # updated in the gibbs-step. Nested functions do not hurt-performance. 
    function gibbs_individual_parameters!(n_individuals::T1, ix::T1) where T1<:Signed

        # Propose parameters, run in parallel!
        @inbounds Threads.@threads for j in 1:n_individuals
            
            local dist_ind_param = calc_dist_ind_param(pop_par_current, pop_sampler, j)

            # Update random-numbers (see if accepted or not)
            update_random_numbers!(rand_num_new_arr[j], rand_num_current_arr[j], filter_opt_arr[j])

            local n_param = ind_param_info_arr[j].n_param
            x_old = @view ind_param_current[1:n_param, j]
            x_prop::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_param)
                
            propose_parameters(x_prop, x_old, mcmc_sampler_ind_arr[j],
                n_param, ind_param_info_arr[j].pos_ind_param)

            update_ind_param_ind!(mod_param_arr[j].individual_parameters, x_prop, 
                ind_param_info_arr[j].log_ind_param)

            log_jac_new_j::FLOAT = calc_log_jac(x_prop, ind_param_info_arr[j].pos_ind_param, n_param)
            log_prior_new_j::FLOAT = logpdf(dist_ind_param, x_prop)
            #start_time2 = now()
            log_lik_new_j::FLOAT = run_filter(filter_opt_arr[j], mod_param_arr[j], rand_num_new_arr[j], 
                                        model_arr[j], ind_data_arr[j])

            log_u::FLOAT = log(rand())
            log_alpha_j::FLOAT = (log_lik_new_j - log_lik_old[j]) + (log_prior_new_j - log_prior_old_ind_arr[j]) + 
                        (log_jac_old_ind_arr[j] - log_jac_new_j)
            # In case of very bad-parameters (NaN) do not accept 
            if isnan(log_alpha_j)
                log_alpha_j = -Inf              
            end

            # Accept 
            if log_u < log_alpha_j
                log_lik_old[j] = log_lik_new_j
                log_prior_old_ind_arr[j] = log_prior_new_j
                log_jac_old_ind_arr[j] = log_jac_new_j

                mcmc_chains.ind_param[j][:, ix] .= x_prop
                mcmc_chains.log_lik_ind[ix, j] = log_lik_old[j]
                ind_param_current[:, j] .= x_prop
                rand_num_current_arr[j] = deepcopy(rand_num_new_arr[j])
                
            # Do not accept 
            else
                mcmc_chains.ind_param[j][:, ix] .= ind_param_current[:, j]
                mcmc_chains.log_lik_ind[ix, j] = log_lik_old[j]

                # Change mod-parameter array to old-values 
                update_ind_param_ind!(mod_param_arr[j].individual_parameters, x_old, 
                    ind_param_info_arr[j].log_ind_param)
            end

            # Update mcmc-proposal based on alpha 
            update_sampler!(mcmc_sampler_ind_arr[j], mcmc_chains.ind_param[j], ix, log_alpha_j)

            # If using SSA-filter update random-numbers 
            if typeof(filter_opt_arr[j]) <: BootstrapFilterSsa
                update_random_numbers!(rand_num_current_arr[j], filter_opt_arr[j])
            end
        end
    end


    # Update kappa sigma and mcmc-sampler. 
    function gibbs_kappa_sigma_parameters!(n_individuals::T1, ix::T1) where T1<:Signed

        propose_parameters(kappa_sigma_new, kappa_sigma_current, mcmc_sampler_pop,
                           n_kappa_sigma, pos_kappa_sigma)

        # Update parameters with new kappa and sigma (and relevant individual parameters)
        @simd for j in 1:n_individuals
            @inbounds map_sigma_to_mod_param_mixed!(mod_param_arr[j], log_sigma, (@view kappa_sigma_new[sigma_index]))
        end
        @simd for j in 1:n_individuals
            @inbounds map_kappa_to_mod_param_mixed!(mod_param_arr[j], log_kappa, (@view kappa_sigma_new[kappa_index]))
        end

        # Calculate likelihood (in parallel!)
        log_lik_arr::Array{FLOAT, 1} = zeros(n_individuals)
        Threads.@threads for j in 1:n_individuals
            log_lik_arr[j] = run_filter(filter_opt_arr[j], mod_param_arr[j], 
                    rand_num_current_arr[j], model_arr[j], ind_data_arr[j])
        end

        log_lik_new::FLOAT = sum(log_lik_arr)
        log_jac_new::FLOAT = calc_log_jac(kappa_sigma_new, pos_kappa_sigma, n_kappa_sigma)
        log_prior_new::FLOAT = calc_log_prior(kappa_sigma_new, prior_kappa_sigma, n_kappa_sigma)

        log_u::FLOAT = log(rand())
        log_alpha::FLOAT = (log_lik_new - sum(log_lik_old)) + (log_prior_new - log_prior_old_kappa_sigma[1]) + 
                    (log_jac_old_kappa_sigma[1] - log_jac_new)
        # In case of very bad-parameters (NaN) do not accept 
        if isnan(log_alpha)
            log_alpha = -Inf 
        end

        if log_u < log_alpha
            log_lik_old .= log_lik_arr
            log_prior_old_kappa_sigma[1] = log_prior_new
            log_jac_old_kappa_sigma[1] = log_jac_new
            mcmc_chains.kappa_sigma[:, ix] .= kappa_sigma_new
            kappa_sigma_current .= kappa_sigma_new
            mcmc_chains.log_lik_kappa_sigma[ix] = log_lik_new
            
        # Do not accept 
        else
            mcmc_chains.kappa_sigma[:, ix] .= kappa_sigma_current
            mcmc_chains.log_lik_kappa_sigma[ix] = sum(log_lik_old)
        end

        # Ensure that mod-param array has correct sigma-values 
        @simd for j in 1:n_individuals
            @inbounds map_sigma_to_mod_param_mixed!(mod_param_arr[j], log_sigma, (@view kappa_sigma_current[sigma_index]))
        end
        @simd for j in 1:n_individuals
            @inbounds map_kappa_to_mod_param_mixed!(mod_param_arr[j], log_kappa, (@view kappa_sigma_current[kappa_index]))
        end

        # Update mcmc-proposal based on alpha 
        update_sampler!(mcmc_sampler_pop, mcmc_chains.kappa_sigma, ix, log_alpha)
    end

    # Ensure correct directory for saving result 
    file_loc = deepcopy(file_loc)
    file_loc.dir_save *= "/" * mcmc_sampler_ind.name_sampler * "/"

    # Observed data for each individual
    ind_data_arr = init_ind_data_arr(file_loc, filter_opt)

    # Relevant dimensions 
    dim_mean = length(pop_param_info.init_pop_param_mean)
    n_individuals = length(ind_data_arr)
    n_kappa_sigma = length(pop_param_info.prior_pop_param_kappa) + length(pop_param_info.prior_pop_param_sigma)

    # Each individual has unique model object (for propegating in parallel)
    model_arr = [deepcopy(model) for i in 1:n_individuals]
    
    # Sampler for the population parameters 
    pop_sampler = init_pop_sampler(pop_sampler_opt, n_individuals, dim_mean)

    # Individual parameter distribution 
    pop_par_tmp::PopParam = init_pop_param_curr(pop_param_info)

    # Starting values are either choosen from pilot-data or user input 
    if pilot_id == 0
    
        # Each individual has a unqie sampler and parameter-information object 
        local pop_par_tmp = init_pop_param_curr(pop_param_info)
        local dist_ind_param = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)
        ind_param_info_arr = init_param_info_arr(ind_param_info, dist_ind_param, n_individuals)
        mcmc_sampler_ind_arr = init_mcmc_arr(mcmc_sampler_ind, ind_param_info_arr, n_individuals)

        # Each individual uses a unique filter 
        filter_opt_arr = init_filter_opt_arr(filter_opt, n_individuals)

    else
        local pop_par_tmp = init_pop_param_curr(pop_param_info)
        local dist_ind_param = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)
        
        # Initialise starting values from pilot 
        pop_param_info, ind_param_info_arr = init_param_pilot(pop_param_info, 
            ind_param_info, file_loc, dist_ind_param, pilot_id)

        mcmc_sampler_ind_arr, mcmc_sampler_pop = init_mcmc_sampler_pilot(mcmc_sampler_ind, 
            mcmc_sampler_pop, file_loc, n_individuals, pilot_id)

        filter_opt_arr = init_filter_arr_pilot(filter_opt, file_loc, n_individuals, pilot_id)
        init_filter_arr_pilot(filter_opt, file_loc, n_individuals, pilot_id)
    end

    # Initialise mcmc-chains
    mcmc_chains::ChainsMixed = init_chains(pop_param_info, ind_param_info_arr, n_individuals, n_samples)
    
    # Arrays storing current chain value 
    kappa_sigma_current::Array{FLOAT, 1} = deepcopy(mcmc_chains.kappa_sigma[:, 1])
    pop_par_current::PopParam = init_pop_param_curr(mcmc_chains.mean[:, 1], mcmc_chains.scale[:, 1], mcmc_chains.corr[:, :, 1])
    ind_param_current::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (dim_mean, n_individuals))
    for i in 1:n_individuals
        ind_param_current[:, i] = deepcopy(mcmc_chains.ind_param[i][:, 1])
    end
    
    # Parameters useful when proposing new-parameters kappa sigma 
    pos_kappa::Array{Bool, 1} = pop_param_info.pos_pop_param_kappa
    pos_sigma::Array{Bool, 1} = pop_param_info.pos_pop_param_sigma
    pos_kappa_sigma::Array{Bool, 1} = vcat(pos_kappa, pos_sigma)
    prior_kappa_sigma = vcat(pop_param_info.prior_pop_param_kappa, pop_param_info.prior_pop_param_sigma)
    kappa_sigma_new::Array{FLOAT, 1} = deepcopy(kappa_sigma_current)
    log_kappa::Array{Bool, 1} = pop_param_info.log_pop_param_kappa
    log_sigma::Array{Bool, 1} = pop_param_info.log_pop_param_sigma
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:length(prior_kappa_sigma)]
    kappa_index = .!sigma_index

    # Individual parameters 
    ind_param_new::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (dim_mean, n_individuals))

    # Filter parameters 
    rand_num_current_arr = init_rand_num_arr(ind_data_arr, model_arr, filter_opt_arr, n_individuals)
    rand_num_new_arr = init_rand_num_arr(ind_data_arr, model_arr, filter_opt_arr, n_individuals)
    mod_param_arr = init_model_parameters_arr(ind_param_info_arr, pop_param_info, model, n_individuals, ind_data_arr)

    # Correctly transform parameters 
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

    # Acceptance probability arrays for single-individual parameters 
    log_lik_old::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)
    log_jac_old_ind_arr::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)
    log_prior_old_ind_arr::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)

    # Old values acceptance probability kappa and sigma 
    log_jac_old_kappa_sigma::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, 1)
    log_prior_old_kappa_sigma::Array{FLOAT, 1} = Array{FLOAT}(undef, 1)

    # Calculate initial values for kappa-sigma and individual parameters 
    # Individual parameters 
    println(n_individuals)
    for j in 1:n_individuals

        # Distribution of parameters and old-parameter values
        dist_ind_param = calc_dist_ind_param(pop_par_current, pop_sampler, j)
        n_param = ind_param_info_arr[j].n_param
        param_old = ind_param_current[1:n_param, j]

        log_jac_old_ind_arr[j] = calc_log_jac(param_old, ind_param_info_arr[j].pos_ind_param, n_param)
        log_prior_old_ind_arr[j] = logpdf(dist_ind_param, param_old)
        
        log_lik_old[j] = run_filter(filter_opt_arr[j], mod_param_arr[j], rand_num_current_arr[j], 
                                            model_arr[j], ind_data_arr[j])

        # Ensure that Nans are avoided 
        if isnan(log_lik_old[j])
            log_lik_old[j] = -Inf 
        end
    end    

    @printf("Log-likelihood starting value = %.3f\n", sum(log_lik_old))

    # Kappa-sigma 
    log_jac_old_kappa_sigma[1] = calc_log_jac(kappa_sigma_current, pos_kappa_sigma, n_kappa_sigma)
    log_prior_old_kappa_sigma[1] = calc_log_prior(kappa_sigma_current, prior_kappa_sigma, n_kappa_sigma)

    # Warm up population nuts-sampler 
    chain_current = gibbs_pop_param_warm_up(pop_par_current, pop_sampler, 
        pop_param_info, ind_param_current)

    # Perform the mcmc-gibbs-sampling     
    start_time_sampler = now()
    @showprogress "Running sampler..." for i in 2:n_samples

        # Stage 1 gibbs 
        gibbs_individual_parameters!(n_individuals, i)

        # Stage 2 gibbs
        gibbs_kappa_sigma_parameters!(n_individuals, i)
            
        # To guard against bad start-guess for ind_param_current re-tune NUTS-sampler
        if (i % 1000 == 0 || i == 100) && pilot == true
            chain_current = gibbs_pop_param_warm_up(pop_par_current, pop_sampler, pop_param_info, ind_param_current)
        end
        if (i == 1000) && pilot == false
            chain_current = gibbs_pop_param_warm_up(pop_par_current, pop_sampler, pop_param_info, ind_param_current)
        end
                
        # Stage 3 gibbs (population parameters via nuts), redo-warm up every 100:th iteration to properly tune-NUTS 
        chain_current = gibbs_pop_param!(pop_par_current, pop_sampler, chain_current, ind_param_current)
        mcmc_chains.mean[:, i] .= pop_par_current.mean_vec
        mcmc_chains.scale[:, i] .= pop_par_current.scale_vec
        mcmc_chains.corr[:, :, i] .= pop_par_current.corr_mat
    end

    end_time_sampler = now()
    run_time_sampler = end_time_sampler - start_time_sampler

    if pilot == false
        write_result_file(mcmc_chains, n_samples, file_loc, filter_opt, pop_param_info, pilot_id, run_time_sampler)
    end
    if pilot == false && benchmark == false
        @printf("Running posterior visual check...")
        run_pvc_mixed(model, mcmc_chains, ind_data_arr, pop_param_info, ind_param_info_arr, file_loc, filter_opt, pop_sampler)
        @printf("done\n")
    end

    return tuple(mcmc_chains, mcmc_sampler_ind_arr, mcmc_sampler_pop)
end 