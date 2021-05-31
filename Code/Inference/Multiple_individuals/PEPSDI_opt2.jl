"""
    init_kappa_sigma_sampler_opt(sampler::KappaSigmaNormal; variances=0.05, n_warm_up=20, acc_prop=0.65)

Initialise (ĸ_pop, ξ_pop)-parameterisation options for NUTS-sampler. 

NUTS options are acceptance probability and number of warm-up samples. 

Parameterisation is (ĸ, ξ) ~ N((ĸ_pop, ξ_pop), εI), ε << 1. As shown in the paper 
this sidesteps the need of step2 in PEPSDI running option 1.  
"""
function init_kappa_sigma_sampler_opt(sampler::KappaSigmaNormal; variances=0.05, n_warm_up=20, acc_prop=0.65)
    return KappaSigmaNormalSamplerOpt(acc_prop, n_warm_up, variances)
end


"""
    init_kappa_sigma_sampler(sampler_opt::KappaSigmaNormalSamplerOpt, n_kappa_sigma)

Initalise struct used within Gibbs-sampler to sample (ĸ_pop, ξ_pop). 

For efficiency number of kappa, ξ parameters are stored. 
"""
function init_kappa_sigma_sampler(sampler_opt::KappaSigmaNormalSamplerOpt, n_kappa_sigma)
    
    
    variances = sampler_opt.variances 
    variances_use::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_kappa_sigma)
    if typeof(variances) <: AbstractFloat
        variances_use .= variances
    elseif typeof(variances) <: Array{<:AbstractFloat, 1}
        variances_use .= variances
    else
        @printf("Error: Variances should be array or single value, and of type float")
    end

    return KappaSigmaNormalSampler(sampler_opt.acc_prop, 
                                   sampler_opt.n_warm_up, 
                                   variances_use, 
                                   n_kappa_sigma)
end


"""
    calc_dist_kappa_sigma(pop_param_info, kappa_sigma_sampler::KappaSigmaNormalSampler)

Based on current (ĸ_pop, ξ_pop) calculate distribuion-struct for (ĸ_pop, ξ_pop). 
"""
function calc_dist_kappa_sigma(pop_param_info, kappa_sigma_sampler::KappaSigmaNormalSampler)

    cov_mat = Diagonal(kappa_sigma_sampler.variances)
    mean_val = vcat(pop_param_info.init_pop_param_kappa, pop_param_info.init_pop_param_sigma)
    dist = MvNormal(mean_val, cov_mat)

    return dist 
end
function calc_dist_kappa_sigma(kappa_sigma_mean::Array{FLOAT, 1}, kappa_sigma_sampler::KappaSigmaNormalSampler)

    cov_mat = Diagonal(kappa_sigma_sampler.variances)
    dist = MvNormal(kappa_sigma_mean, cov_mat)

    return dist 
end


"""
    turing_kappa_sigma_normal(prior_mean, variances, n_param, kappa_sigma_current, n_individuals)

Turing-library model-function for sampling (ĸ_pop, ξ_pop). 

See sampler-parameterisation in documentation for [`init_kappa_sigma_sampler_opt`](@ref)
"""
@model function turing_kappa_sigma_normal(prior_mean, variances, n_param, kappa_sigma_current, n_individuals)
        
    mean_val ~ arraydist(prior_mean)
    cov_mat = Diagonal(variances)
    kappa_sigma_current ~ filldist(MvNormal(mean_val, cov_mat), n_individuals)
    
    return mean_val
end


"""
    gibbs_kappa_sigma_warm_up(kappa_sigma_mean::Array{FLOAT, 1}, 
                              kappa_sigma_sampler::KappaSigmaNormalSampler, 
                              pop_param_info::ParamInfoPop, 
                              kappa_sigma_current::Array{FLOAT, 2})::Chains
    
Warm up NUTS-sampler using number of samples specified by kappa_sigma_sampler and return Turing-chain object.

For sampler-parameterisation see documentation for [`init_kappa_sigma_sampler_opt`](@ref)
"""
function gibbs_kappa_sigma_warm_up(kappa_sigma_mean::Array{FLOAT, 1}, 
                                   kappa_sigma_sampler::KappaSigmaNormalSampler, 
                                   pop_param_info::ParamInfoPop, 
                                   kappa_sigma_current::Array{FLOAT, 2})::Chains
    
    prior_mean = vcat(pop_param_info.prior_pop_param_kappa, pop_param_info.prior_pop_param_sigma)

    n_param = kappa_sigma_sampler.n_kappa_sigma
    n_individuals = size(kappa_sigma_current)[2]

    mod_pop_param = turing_kappa_sigma_normal(prior_mean, kappa_sigma_sampler.variances, n_param, kappa_sigma_current, n_individuals)

    # Sampler options 
    acc_prop::Float64 = convert(Float64, kappa_sigma_sampler.acc_prop)
    n_warm_up::Int64 = convert(Int64, kappa_sigma_sampler.n_warm_up)

    Turing.setadbackend(:forwarddiff)

    # If domain error is thrown by NUTS, don't update parameters
    local iter = 0
    while iter < 5
        try
            local sampled_chain
            @suppress begin
            sampled_chain = sample(mod_pop_param, NUTS(n_warm_up, acc_prop), 1, save_state=true)
            end
            return sampled_chain
        catch
            @printf("Error in warm-up for population sampler\n")
            iter += 1
        end
        @printf("i = %d\n", iter)
    end

    @printf("Failed with population sampler warm-up\n")
    exit(1)

end


"""
    gibbs_pop_param!(pop_par_current::PopParam, 
                     pop_sampler::PopSamplerOrnstein, 
                     current_chain::Chains,
                     ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

Update (ĸ_pop, ξ_pop)-parameters based on kappa_sigma_current and store in kappa_sigma_mean. Return Turing-chain from update. 

Update is performed via Turing-resume function to avoid allocating new Chains-struct. 
Upon updating the chain generated by Turing-resume is returned.

For sampler-parameterisation see documentation for [`init_kappa_sigma_sampler_opt`](@ref)
"""
function gibbs_kappa_sigma!(kappa_sigma_mean::Array{FLOAT, 1}, 
                            kappa_sigma_sampler::KappaSigmaNormalSampler, 
                            current_chain::Chains,
                            pop_param_info::ParamInfoPop,
                            kappa_sigma_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Change the individual parameters current value 
    n_param = kappa_sigma_sampler.n_kappa_sigma
    current_chain.info.model.args.kappa_sigma_current .= kappa_sigma_current

    @suppress begin
    current_chain = resume(current_chain, 1, save_state=true)
    end

    name_mean_val = ["mean_val["*string(i)*"]" for i in 1:n_param]

    kappa_sigma_mean .= current_chain[name_mean_val].value[1:end]

    return current_chain
end


"""
    run_PEPSDI_opt2(n_samples::T1, 
                    pop_param_info::ParamInfoPop, 
                    ind_param_info::ParamInfoIndPre, 
                    file_loc::FileLocations,
                    model,
                    filter_opt, 
                    mcmc_sampler_ind, 
                    mcmc_sampler_pop, 
                    pop_sampler_opt,
                    kappa_sigma_sampler_opt;
                    pilot::Bool=false, 
                    pilot_id=0,                        
                    benchmark=false) where T1<:Signed

Take n_samples of η, ξ_pop, κ_pop and c_i using PEPSDI-option 2 (default option)

PEPSDI running option 2 is to slighly perturb (ξ, κ) and let these weakly vary between cells. 
The inference focuses on inferring η, alongside ξ_pop, κ_pop; (ξ_i, κ_i) ~ π(ξ_pop, κ_pop). As seen in 
the paper this side-steps the need of step 2 in the original Gibbs-sampler (PEPSDI option 1) and 
considerble speeds up inference. 

Currently AM, GenAm and RAM-sampler are available for proposing ξ_i, κ_i and c_i. For η LKJ-normal and 
ornstein-mixed parameterisation are available. However, users can write own modules for η-parameterisation.  

The model can be a wide-range of state-space models; a SDE-model (stochastic differential), SSA-model (Gillespie model), 
Extrande-model (like SSA but allow time-varying c_i) and Poisson-model (tau-leaping model). Particle filters supported 
is bootstrap filter for each model-type, where particles can be correlated for SDE and Poisson-models. For SDE-models 
with a linear observation and error-model, y = Px + ε, ε ~ N(0, σI), the modified diffusion bridge filter can be employed 
(often more statistcally efficient). How to setup a model to be inferred using PEPSDI see provided notebooks. 

# Args
- `n_samples`: number of samples to take from posterior. 
- `pop_param_info`: priors, transformations and initial values for population parameters (η, ξ, κ) (see [`init_pop_param_info`](@ref))
- `ind_param_info`: initial values and transformation for individual parameters (c_i) (see [`init_ind_param_info`](@ref))
- `file_loc`: directory to data and directory for storing result (see [`init_file_loc`](@ref))
- `model`: user provided model 
- `filter_opt`: particle filter-options (correlation level, number of particles and step-length) (see [`init_filter`](@ref))
- `mcmc_sampler_ind`: mcmc-sampler (AM, GenAM or RAM) to use for c_i (see [`init_mcmc`](@ref))
- `mcmc_sampler_pop`: mcmc-sampler (AM, GenAM or RAM) to use for (ξ_i, κ_i) (see [`init_mcmc`](@ref))
- `pop_sampler_opt`: NUTS-options for η-sampler (see [`init_pop_sampler_opt`](@ref))
- `kappa_sigma_sampler_opt`: NUTS-options for sampling (ĸ_pop, ξ_pop) (see [`init_kappa_sigma_sampler_opt`](@ref))
- `pilot`: if true result is not written to disk as pilot-run is assumed. 
- `pilot_id`: if non-zero uses pilot-run data with tag pilot_id (integer) to set initial-values for mcmc-samplers and (η, σ, κ and Φ_i), 
    and using number of particles obtained from particle-tuning (see [`tune_particles_mixed`](@ref))
- `benchmark`: bool dictating if benchmark is run, deafult false. 
"""
function run_PEPSDI_opt2(n_samples::T1, 
                         pop_param_info::ParamInfoPop, 
                         ind_param_info::ParamInfoIndPre, 
                         file_loc::FileLocations,
                         model,
                         filter_opt, 
                         mcmc_sampler_ind, 
                         mcmc_sampler_pop, 
                         pop_sampler_opt,
                         kappa_sigma_sampler_opt;
                         pilot::Bool=false, 
                         pilot_id=0, 
                         benchmark=false) where T1<:Signed

    # Update the individual parameters and random numbers Note, the mcmc-samplers are also 
    # updated in the gibbs-step. Nested functions do not hurt-performance. 
    function gibbs_individual_parameters!(n_individuals::T1, ix::T1) where T1<:Signed

        # Propose parameters, run in parallel!
        dist_kappa_sigma = calc_dist_kappa_sigma(kappa_sigma_mean, kappa_sigma_sampler)
        # Thread safe 
        dist_kappa_sigma_arr = [deepcopy(dist_kappa_sigma) for j in 1:n_individuals]

        @inbounds Threads.@threads for j in 1:n_individuals
            
            local dist_ind_param = calc_dist_ind_param(pop_par_current, pop_sampler, j)

            # Update random-numbers (see if accepted or not)
            update_random_numbers!(rand_num_new_arr[j], rand_num_current_arr[j], filter_opt_arr[j])

            # Propose individual parameters 
            local n_param = ind_param_info_arr[j].n_param
            x_old = @view ind_param_current[1:n_param, j]
            x_prop::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_param)
            propose_parameters(x_prop, x_old, mcmc_sampler_ind_arr[j],
                n_param, ind_param_info_arr[j].pos_ind_param)

            # Propose new kappa sigm a
            kappa_sigma_old = @view kappa_sigma_current[1:n_kappa_sigma, j]
            kappa_sigma_prop::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_kappa_sigma)
            propose_parameters(kappa_sigma_prop, kappa_sigma_old, mcmc_sampler_pop_arr[j],
                n_kappa_sigma, pos_kappa_sigma)
            
            # Update individual parameters and kappa-sigma 
            update_ind_param_ind!(mod_param_arr[j].individual_parameters, x_prop, 
                ind_param_info_arr[j].log_ind_param)
            map_sigma_to_mod_param_mixed!(mod_param_arr[j], log_sigma, (@view kappa_sigma_prop[sigma_index]))
            map_kappa_to_mod_param_mixed!(mod_param_arr[j], log_kappa, (@view kappa_sigma_prop[kappa_index]))

            # Prior and jacobian must consider both kappa-sigma and indivdual parameters 
            log_jac_new_j::FLOAT = 0.0
            log_prior_new_j::FLOAT = 0.0
            # Individual parameters 
            log_jac_new_j += calc_log_jac(x_prop, ind_param_info_arr[j].pos_ind_param, n_ind_param)
            log_prior_new_j += logpdf(dist_ind_param, x_prop)
            # Kappa sigma 
            log_jac_new_j += calc_log_jac(kappa_sigma_prop, pos_kappa_sigma, n_kappa_sigma)
            log_prior_new_j += logpdf(dist_kappa_sigma_arr[j], kappa_sigma_prop)

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
                kappa_sigma_val[j][:, ix] .= kappa_sigma_prop
                mcmc_chains.log_lik_ind[ix, j] = log_lik_old[j]

                kappa_sigma_current[:, j] .= kappa_sigma_prop
                ind_param_current[:, j] .= x_prop
                rand_num_current_arr[j] = deepcopy(rand_num_new_arr[j])
                
            # Do not accept 
            else
                mcmc_chains.ind_param[j][:, ix] .= ind_param_current[:, j]
                kappa_sigma_val[j][:, ix] .= kappa_sigma_current[:, j]
                mcmc_chains.log_lik_ind[ix, j] = log_lik_old[j]
                
            end

            # Update mcmc-proposal based on alpha 
            update_sampler!(mcmc_sampler_ind_arr[j], mcmc_chains.ind_param[j], ix, log_alpha_j)
            update_sampler!(mcmc_sampler_pop_arr[j], kappa_sigma_val[j], ix, log_alpha_j)

        end
    end
        
    # Ensure correct directory for saving result 
    file_loc = deepcopy(file_loc)
    file_loc.dir_save *= "/" * mcmc_sampler_ind.name_sampler * "/"

    n_ind_param = ind_param_info.n_param

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
    kappa_sigma_sampler = init_kappa_sigma_sampler(kappa_sigma_sampler_opt, n_kappa_sigma)

    # Individual parameter distribution 
    pop_par_tmp::PopParam = init_pop_param_curr(pop_param_info)
    dist_ind_param = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)
    dist_kappa_sigma = calc_dist_kappa_sigma(pop_param_info, kappa_sigma_sampler)

    # Starting values are either choosen from pilot-data or user input 
    if pilot_id == 0
    
        # Each individual has a unqie sampler and parameter-information object 
        dist_ind_param_tmp = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)
        dist_ind_param_arr = Array{typeof(dist_ind_param_tmp), 1}(undef, n_individuals)
        for i in 1:n_individuals
            dist_ind_param_arr[i] = calc_dist_ind_param(pop_par_tmp, pop_sampler, i)
        end
        ind_param_info_arr = init_param_info_arr(ind_param_info, dist_ind_param_arr, n_individuals)
        mcmc_sampler_ind_arr = init_mcmc_arr(mcmc_sampler_ind, ind_param_info_arr, n_individuals)
        mcmc_sampler_pop_arr = init_mcmc_arr(mcmc_sampler_pop, pop_param_info, n_individuals)

        # Each individual uses a unique filter 
        filter_opt_arr = init_filter_opt_arr(filter_opt, n_individuals)

    else
        local pop_par_tmp = init_pop_param_curr(pop_param_info)
        local dist_ind = calc_dist_ind_param(pop_par_tmp, pop_sampler, 1)
        
        # Initialise starting values from pilot 
        pop_param_info, ind_param_info_arr = init_param_pilot(pop_param_info, 
            ind_param_info, file_loc, dist_ind_param, pilot_id, sampler="alt")

        mcmc_sampler_ind_arr, mcmc_sampler_pop_arr = init_mcmc_sampler_pilot(mcmc_sampler_ind, 
            mcmc_sampler_pop, file_loc, n_individuals, pilot_id, "alt")

        filter_opt_arr = init_filter_arr_pilot(filter_opt, file_loc, n_individuals, pilot_id, sampler="alt")
    end


    # Initialise mcmc-chains
    mcmc_chains::ChainsMixed = init_chains(pop_param_info, ind_param_info_arr, n_individuals, n_samples)
    
    # Arrays storing current chain value 
    pop_par_current::PopParam = init_pop_param_curr(mcmc_chains.mean[:, 1], mcmc_chains.scale[:, 1], mcmc_chains.corr[:, :, 1])
    ind_param_current::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (ind_param_info_arr[1].n_param, n_individuals))
    for i in 1:n_individuals
        ind_param_current[:, i] .= deepcopy(mcmc_chains.ind_param[i][:, 1])
    end

    kappa_sigma_mean::Array{FLOAT, 1} = deepcopy(mcmc_chains.kappa_sigma[:, 1])
    kappa_sigma_current::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (n_kappa_sigma, n_individuals))
    kappa_sigma_val::Array{Array{FLOAT, 2}} = Array{Array{FLOAT, 2}}(undef, n_individuals)
    for i in 1:n_individuals
        kappa_sigma_current[:, i] .= kappa_sigma_mean
        kappa_sigma_val[i] = Array{FLOAT, 2}(undef, (n_kappa_sigma, n_samples))
        kappa_sigma_val[i][:, 1] .= kappa_sigma_mean
    end
    
    # Parameters useful when proposing new-parameters kappa sigma 
    pos_kappa::Array{Bool, 1} = pop_param_info.pos_pop_param_kappa
    pos_sigma::Array{Bool, 1} = pop_param_info.pos_pop_param_sigma
    pos_kappa_sigma::Array{Bool, 1} = vcat(pos_kappa, pos_sigma)
    prior_kappa_sigma = vcat(pop_param_info.prior_pop_param_kappa, pop_param_info.prior_pop_param_sigma)
    kappa_sigma_new::Array{FLOAT, 2} = deepcopy(kappa_sigma_current)
    log_kappa::Array{Bool, 1} = pop_param_info.log_pop_param_kappa
    log_sigma::Array{Bool, 1} = pop_param_info.log_pop_param_sigma
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:length(prior_kappa_sigma)]
    kappa_index = .!sigma_index

    # Individual parameters 
    ind_param_new::Array{FLOAT, 2} = Array{FLOAT, 2}(undef, (ind_param_info_arr[1].n_param, n_individuals))

    # Filter parameters 
    rand_num_current_arr = init_rand_num_arr(ind_data_arr, model_arr, filter_opt_arr, n_individuals)
    rand_num_new_arr = init_rand_num_arr(ind_data_arr, model_arr, filter_opt_arr, n_individuals)
    mod_param_arr = init_model_parameters_arr(ind_param_info_arr, pop_param_info, model, n_individuals, ind_data_arr)

    # Correctly transform parameters 
    @simd for j in 1:n_individuals
        @inbounds map_sigma_to_mod_param_mixed!(mod_param_arr[j], log_sigma, (@view kappa_sigma_current[sigma_index, j]))
    end
    @simd for j in 1:n_individuals
        @inbounds map_kappa_to_mod_param_mixed!(mod_param_arr[j], log_kappa, (@view kappa_sigma_current[kappa_index, j]))
    end
    @simd for j in 1:n_individuals
        @inbounds update_ind_param_ind!(mod_param_arr[j].individual_parameters, ind_param_current[:, j], 
             ind_param_info_arr[j].log_ind_param)
    end
    @simd for j in 1:n_individuals
        model.calc_x0!(mod_param_arr[j].x0, mod_param_arr[j].individual_parameters)
    end

    println("Starting to calculate likelihood :)")

    # Acceptance probability arrays for single-individual parameters 
    log_lik_old::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)
    log_jac_old_ind_arr::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)
    log_prior_old_ind_arr::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, n_individuals)

    # Old values acceptance probability kappa and sigma 
    log_jac_old_kappa_sigma::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, 1)
    log_prior_old_kappa_sigma::Array{FLOAT, 1} = Array{FLOAT}(undef, 1)

    # Calculate initial values for kappa-sigma and individual parameters 
    # Individual parameters 
    for j in 1:n_individuals

        # Distribution of parameters and old-parameter values
        dist_ind_param = calc_dist_ind_param(pop_par_current, pop_sampler, j)
        dist_kappa_sigma = calc_dist_kappa_sigma(kappa_sigma_mean, kappa_sigma_sampler)

        log_jac_new_j::FLOAT = 0.0
        log_prior_new_j::FLOAT = 0.0
        # Individual parameters 
        n_param = ind_param_info_arr[j].n_param
        log_jac_new_j += calc_log_jac(ind_param_current[:, j], ind_param_info_arr[j].pos_ind_param, n_ind_param)
        log_prior_new_j += logpdf(dist_ind_param, ind_param_current[:, j])
        # Kappa sigma 
        log_jac_new_j += calc_log_jac(kappa_sigma_current[:, j], pos_kappa_sigma, n_kappa_sigma)
        log_prior_new_j += logpdf(dist_kappa_sigma, kappa_sigma_current[:, j])

        log_jac_old_ind_arr[j] = log_jac_new_j
        log_prior_old_ind_arr[j] = log_prior_new_j        
        log_lik_old[j] = run_filter(filter_opt_arr[j], mod_param_arr[j], rand_num_current_arr[j], 
                                            model_arr[j], ind_data_arr[j])

        # Ensure that Nans are avoided 
        if isnan(log_lik_old[j])
            log_lik_old[j] = -Inf 
        end
    end    
        
    @printf("Log-likelihood starting value = %.3f\n", sum(log_lik_old))
    println(log_lik_old)
    
    # Warm up population nuts-sampler 
    chain_current = gibbs_pop_param_warm_up(pop_par_current, pop_sampler, 
        pop_param_info, ind_param_current)
    chain_kappa_sigma = gibbs_kappa_sigma_warm_up(kappa_sigma_mean, kappa_sigma_sampler, 
        pop_param_info, kappa_sigma_current)

    # Perform the mcmc-gibbs-sampling 
    time1 = now() - now()
    time2 = now() - now()
    time3 = now() - now()   
    
    start_time_sampler = now()
    @showprogress "Running sampler..." for i in 2:n_samples

        # Stage 1 gibbs 
        start_time = now()
        gibbs_individual_parameters!(n_individuals, i)
        end_time = now()
        time1 += end_time - start_time 

        # To guard against bad start-guess for ind_param_current re-tune NUTS-sampler
        if (i % 1000 == 0 || i == 100) && pilot == true
            chain_current = gibbs_pop_param_warm_up(pop_par_current, pop_sampler, pop_param_info, ind_param_current)
            chain_kappa_sigma = gibbs_kappa_sigma_warm_up(kappa_sigma_mean, kappa_sigma_sampler, 
                pop_param_info, kappa_sigma_current)
        end
        if (i == 1000) && pilot == false
            chain_current = gibbs_pop_param_warm_up(pop_par_current, pop_sampler, pop_param_info, ind_param_current)
            chain_kappa_sigma = gibbs_kappa_sigma_warm_up(kappa_sigma_mean, kappa_sigma_sampler, 
                pop_param_info, kappa_sigma_current)
        end
                
        # Stage 3 gibbs (population parameters via nuts), redo-warm up every 100:th iteration to properly tune-NUTS 
        start_time = now()

        # Population parameters 
        chain_current = gibbs_pop_param!(pop_par_current, pop_sampler, chain_current, ind_param_current)
        mcmc_chains.mean[:, i] .= pop_par_current.mean_vec
        mcmc_chains.scale[:, i] .= pop_par_current.scale_vec
        mcmc_chains.corr[:, :, i] .= pop_par_current.corr_mat

        # Kappa-sigma parameters 
        chain_kappa_sigma = gibbs_kappa_sigma!(kappa_sigma_mean, kappa_sigma_sampler, chain_kappa_sigma, pop_param_info, kappa_sigma_current)
        mcmc_chains.kappa_sigma[:, i] .= kappa_sigma_mean

        end_time = now()
        time3 += end_time - start_time
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

    println("Time 1: ", time1)
    println("Time 3: ", time3)

    return tuple(mcmc_chains, mcmc_sampler_ind_arr, mcmc_sampler_pop_arr)
end 