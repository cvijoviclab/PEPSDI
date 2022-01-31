#=
    Functions for performing posterior visual checks for the PEPSDI running 
    options 1 and 2 using the inferred posterior distribution. 
=# 


"""
    calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterEm)

Calculate time-vector as the maximum time-span in individual data for posterior visual check. 

Calculate for bootstrap EM-filter. 
"""
function calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterEm)
    t_max = 0.0
    for i in 1:length(ind_data_arr)
        if t_max < maximum(ind_data_arr[i].t_vec)
            t_max = maximum(ind_data_arr[i].t_vec)
        end
    end
    dt = filter_opt.delta_t / 2.0
    t_vec = 0.0:dt:t_max

    return t_vec, dt
end
"""
    calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterSsa)

For bootstrap SSA (Gillespie) particle filter. 
"""
function calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterSsa)
    t_max = 0.0
    for i in 1:length(ind_data_arr)
        if t_max < maximum(ind_data_arr[i].t_vec)
            t_max = maximum(ind_data_arr[i].t_vec)
        end
    end
    dt = t_max / 100
    t_vec = 0.0:dt:t_max

    return t_vec, dt
end
"""
    calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterExtrand)

For bootstrap Extrand particle filter 
"""
function calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterExtrand)
    t_max = 0.0
    for i in 1:length(ind_data_arr)
        if t_max < maximum(ind_data_arr[i].t_vec)
            t_max = maximum(ind_data_arr[i].t_vec)
        end
    end
    dt = t_max / 100
    t_vec = 0.0:dt:t_max

    return t_vec, dt
end
"""
    calc_t_vec_pcv(ind_data_arr, filter_opt::ModDiffusionFilter)

For modified diffusion bridge filter 
"""
function calc_t_vec_pcv(ind_data_arr, filter_opt::ModDiffusionFilter)
    t_max = 0.0
    for i in 1:length(ind_data_arr)
        if t_max < maximum(ind_data_arr[i].t_vec)
            t_max = maximum(ind_data_arr[i].t_vec)
        end
    end
    dt = filter_opt.delta_t / 2.0
    t_vec = 0.0:dt:t_max

    return t_vec, dt
end
"""
    calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterPois)

For Poisson (tau-leaping) bootstrap filter 
"""
function calc_t_vec_pcv(ind_data_arr, filter_opt::BootstrapFilterPois)
    t_max = 0.0
    for i in 1:length(ind_data_arr)
        if t_max < maximum(ind_data_arr[i].t_vec)
            t_max = maximum(ind_data_arr[i].t_vec)
        end
    end
    dt = filter_opt.delta_t / 2.0
    t_vec = 0.0:dt:t_max

    return t_vec, dt
end


"""
    solve_model(model::SdeModel, time_span, u0_vec::Array{FLOAT, 1}, individual_parameters, dt)

Simulate trajectory for pvc for a SDE-model for data-points in time_span and starting value u0_vec

Step-length is calculated as half the step-length in the inference. 
"""
function solve_model(model::SdeModel, time_span, u0_vec::Array{FLOAT, 1}, individual_parameters, dt)

    t_vec1, u_mat1 = solve_sde_em(model, time_span, u0_vec, individual_parameters, dt)    
    return t_vec1, u_mat1
end
"""
    solve_model(model::SsaModel, time_span, u0_vec::Array{UInt16, 1}, individual_parameters, dt)

Simulate trajectory for pvc for a SSA(Gillespie)-model for data-points in time_span and starting value u0_vec
"""
function solve_model(model::SsaModel, time_span, u0_vec::Array{UInt16, 1}, individual_parameters, dt)

    time_vec_use = time_span[1]:dt:time_span[2]
    t_vec1, u_mat1 = solve_ssa_model(model, time_vec_use, individual_parameters)    
    return t_vec1, u_mat1
end
"""
    solve_model(model::ExtrandModel, time_span, u0_vec::Array{UInt16, 1}, individual_parameters, dt)

Simulate trajectory for pvc for an Extrande-model for data-points in time_span and starting value u0_vec
"""
function solve_model(model::ExtrandModel, time_span, u0_vec::Array{UInt16, 1}, individual_parameters, dt)

    time_vec_use = time_span[1]:dt:time_span[2]
    t_vec1, u_mat1 = solve_extrande_model(model, time_vec_use, individual_parameters)    
    return t_vec1, u_mat1
end
"""
    solve_model(model::ExtrandModel, time_span, u0_vec::Array{UInt16, 1}, individual_parameters, dt)

Simulate trajectory for pvc for a Poisson(tau-leaping)-model for data-points in time_span and starting value u0_vec
"""
function solve_model(model::PoisonModel, time_span, u0_vec, individual_parameters, dt)

    time_vec_use = time_span[1]:dt:time_span[2]
    t_vec1, u_mat1 = solve_poisson_model(model, time_vec_use, individual_parameters, dt)

    return t_vec1, u_mat1
end


"""
    run_pvc_mixed(model, 
                  mcmc_chains::ChainsMixed, 
                  ind_data_arr::Array{IndData, 1}, 
                  pop_param_info::ParamInfoPop, 
                  ind_param_info_arr::Array{ParamInfoInd, 1},
                  file_loc::FileLocations, 
                  filter_opt, 
                  pop_sampler)

Perform two-types of pvc for each provided covariate values after inference. 

For a model (any type) based on the inferred mcmc-chains perform a pvc by 
first simulating the model 10,000 times, and by then re-simulating a data-set 
of size n-individuals 10,000 times. The first pvc (pvc_mixed) compuates the 0.05, 0.5 and 
0.95 quantiles from 10,000 runs. The second pvc (pvc_mixed_quant) compuates 95 credability intervall 
for the 0.05, 0.5 and 0.95 quantiles when the number of individuals (observed traces)
equals n-individuals. 
The pvc is performed for each covariate value provided in the file-location struct. 
The end result is saved in dir-save in the file-locations struct. 
"""
function run_pvc_mixed(model, 
                       mcmc_chains::ChainsMixed, 
                       ind_data_arr::Array{IndData, 1}, 
                       pop_param_info::ParamInfoPop, 
                       ind_param_info_arr::Array{ParamInfoInd, 1},
                       file_loc::FileLocations, 
                       filter_opt, 
                       pop_sampler)


    # Check if covariates are provided 
    if length(ind_data_arr[1].cov_val) == 0
        cov_val = Array{FLOAT, 1}(undef, 1)
        n_loops = 1
        cov_val_name = ""

    else
        cov_val = file_loc.cov_val
        n_loops = length(file_loc.cov_val)
        cov_val_name = file_loc.cov_name[1]
        cov_tag = "_" * cov_val_name .* string.(cov_val)

        println("Running pvc using covariate $cov_val_name and values ", cov_val)
    end


    for i in 1:n_loops

        @printf("Running pvc none-quantile\n")

        pvc_mixed(model, mcmc_chains, ind_data_arr, pop_param_info, ind_param_info_arr, 
            file_loc, filter_opt, pop_sampler, cov_val = cov_val[i], name_cov = cov_val_name, 
            dist_id = file_loc.dist_id[i])        
            
        @printf("Running pvc quantile\n")
        
        pvc_mixed_quant(model, mcmc_chains, ind_data_arr, pop_param_info, ind_param_info_arr, 
            file_loc, filter_opt, pop_sampler, cov_val = cov_val[i], name_cov = cov_val_name, dist_id = file_loc.dist_id[i])


    end

end


"""
    pvc_mixed(model, 
              mcmc_chains::ChainsMixed, 
              ind_data_arr::Array{IndData, 1}, 
              pop_param_info::ParamInfoPop, 
              ind_param_info_arr::Array{ParamInfoInd, 1},
              file_loc::FileLocations, 
              filter_opt, 
              pop_sampler; 
              n_runs=10000, burn_in=0.2, pvc_quant=false,
              cov_val=Array{FLOAT, 1}(undef, 0), name_cov="", dist_id = 1)

Perform visual predictive check by sampling η, and κ for a mixed-effects model. 

Disgarding the burn_in simulates n_runs by sampling η, and κ from the mcmc-chains 
for the model and then simulating individual parameters using the distribution defined 
by pop_sampler. When done writes median, 0.05 and 0.95 quantiles to dir-save in 
file_loc. Model is simulated in the maximum time-span in ind_data_arr (individual 
data arary). Paramter-information is used to ensure correct parameter transformations. 
The run_pvc_mixed delegates the covariate values to consider. 
"""
function pvc_mixed(model, 
                   mcmc_chains::ChainsMixed, 
                   ind_data_arr::Array{IndData, 1}, 
                   pop_param_info::ParamInfoPop, 
                   ind_param_info_arr::Array{ParamInfoInd, 1},
                   file_loc::FileLocations, 
                   filter_opt, 
                   pop_sampler; 
                   n_runs=10000, burn_in=0.2, pvc_quant=false,
                   cov_val=Array{FLOAT, 1}(undef, 0), name_cov="", dist_id = 1)

    # Indices to sample 
    n_param = size(mcmc_chains.mean)[1]
    n_samples = size(mcmc_chains.mean)[2]
    min_sample_use = convert(Int, floor(n_samples*burn_in))
    indices_use = min_sample_use:n_samples
    n_samples_use = length(indices_use)
    index_sample = rand(DiscreteUniform(1, n_samples_use), n_runs)

    # Calculate time-span to simulate within 
    t_vec, dt = calc_t_vec_pcv(ind_data_arr, filter_opt)
    time_span = (t_vec[1], t_vec[end])

    # Matrix for storing simulated results 
    sol_mat = zeros(model.dim_obs*n_runs, length(t_vec))
    if typeof(model) <: SdeModel
        u0_vec = Array{FLOAT, 1}(undef, model.dim)
    else
        u0_vec = Array{UInt16, 1}(undef, model.dim)
    end

    # Array for holding model-parameters 
    log_kappa = pop_param_info.log_pop_param_kappa
    log_sigma = pop_param_info.log_pop_param_sigma
    n_kappa_sigma = length(vcat(log_kappa, log_sigma))
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:n_kappa_sigma]
    kappa_index = .!sigma_index

    # Correct transformation of individual parameters 
    log_ind_param = ind_param_info_arr[1].log_ind_param

    # Mcmc-chains 
    pop_mean = mcmc_chains.mean
    pop_scale = mcmc_chains.scale
    pop_corr = mcmc_chains.corr
    pop_kappa = mcmc_chains.kappa_sigma[kappa_index, :]
    pop_sigma = mcmc_chains.kappa_sigma[sigma_index, :]

    # Array for model-parameters 
    mod_param_arr = init_model_parameters_arr(ind_param_info_arr, pop_param_info, model, length(ind_data_arr), ind_data_arr)
    model_param = mod_param_arr[1]
    model_param.covariates .= cov_val

    n_ind_param = length(ind_param_info_arr[1].log_ind_param)
    ind_param_arr = Array{FLOAT, 2}(undef, (n_runs, n_ind_param))

    # Run appropriate number of times 
    i = 1
    run_i = 0
    while i < n_runs && run_i < n_runs * 10

        # Selecting random samples from posterior 
        i_sample = indices_use[index_sample[i]]
        mean_vec = pop_mean[:, i_sample]
        scale_vec = pop_scale[:, i_sample]
        corr_mat = pop_corr[:, :, i_sample]

        dist_ind_param = calc_dist_ind_param(init_pop_param_curr(mean_vec, scale_vec, corr_mat), pop_sampler, dist_id)

        # Draw new individual parameters and populate model-param-struct 
        new_param = rand(dist_ind_param, 1)[:, 1]
        map_sigma_to_mod_param_mixed!(model_param, log_sigma, pop_sigma[:, i_sample])
        map_kappa_to_mod_param_mixed!(model_param, log_kappa, pop_kappa[:, i_sample])
        update_ind_param_ind!(model_param.individual_parameters, new_param, log_ind_param)

        # Guard against crazy NUTS-sampler
        if sum(abs.(model_param.individual_parameters.c) .== Inf) > 0
            index_sample[i] = rand(DiscreteUniform(1, n_samples_use), 1)[1]
            run_i += 1
            continue
        end

        local t_vec1
        local u_mat1
        try 
            model.calc_x0!(u0_vec, model_param.individual_parameters)
            t_vec1, u_mat1 = solve_model(model, time_span, u0_vec, model_param.individual_parameters, dt)        
        catch
            run_i += 1
            continue
        end
        
        # Guard against NaN
        if sum(isnan.(u_mat1)) > 0
            run_i += 1
            continue
        end
        
        i_low = model.dim_obs * i - (model.dim_obs - 1)
        i_upp = model.dim_obs * i

        # Calculate observed output 
        y_vec = Array{Float64, 1}(undef, model.dim_obs)
        for j in 1:length(t_vec1)
            model.calc_obs(y_vec, u_mat1[1:model.dim, j], model_param.individual_parameters, 0.0)
            sol_mat[i_low:i_upp, j] .= y_vec
        end

        if sum(isnan.(sol_mat[i_low:i_upp, :])) > 0
            run_i += 1
            continue
        end

        # Save individual parameters used 
        ind_param_arr[i, :] .= new_param

        i += 1
        run_i += 1
    end

    if run_i > n_runs * 10
        @printf("Problem in pvc (many failed runs)\n")
    end

    # Aggregate the median and mean values, note the first
    # three columns are the first state, the second pair is the
    # second state etc.
    quant = zeros(model.dim_obs*5, length(t_vec))
    for i in 1:model.dim_obs
        i_rows = i:model.dim_obs:(n_runs*model.dim_obs)
        start_index = (i - 1) *3 + 1
        quant[start_index, :] = median(sol_mat[i_rows, :], dims=1)
        # Fixing the quantiles
        for j in 1:length(t_vec)
            i_low = start_index+1
            i_upp = start_index+2
            quant[i_low:i_upp, j] = quantile(sol_mat[i_rows, j], [0.05, 0.95])
        end
        # Fixing mean and std 
        quant[start_index + 3, :] = mean(sol_mat[i_rows, :], dims=1)
        quant[start_index + 4, :] = std(sol_mat[i_rows, :], dims=1)
    end

    # Save result Correctly
    if pvc_quant == false
        # Ensure directory for storing results exist 
        dir_save = file_loc.dir_save 
        if !isdir(dir_save)
            mkpath(dir_save)
        end
        path_save = dir_save * "Pvc_" * name_cov * string(cov_val) * ".csv"
        full_data = vcat(quant, collect(t_vec)')'
        data_save = DataFrame(full_data)
        # Ensure correct column-names 
        col_names = Array{String, 1}(undef, model.dim_obs*5 + 1)
        for i in 1:model.dim_obs
            med_tag = "y" * string(i) * "_med"
            q1_tag = "y" * string(i) * "_qu05"
            q2_tag = "y" * string(i) * "_qu95"
            mean_tag = "y" * string(i) * "_mean"
            std_tag = "y" * string(i) * "_std"

            col_names[(i-1)*3+1] = med_tag
            col_names[(i-1)*3+2] = q1_tag
            col_names[(i-1)*3+3] = q2_tag
            col_names[(i-1)*3+4] = mean_tag
            col_names[(i-1)*3+5] = std_tag
        end
        col_names[end] = "time"
    
        rename!(data_save, col_names)
        CSV.write(path_save, data_save)

        # Save individual parameters 
        col_names_ind = ["c" * string(i) for i in 1:n_ind_param]
        data_save_ind = convert(DataFrame, ind_param_arr)
        path_save_ind = dir_save * "Pvc_ind_param.csv"
        rename!(data_save_ind, col_names_ind)
        CSV.write(path_save_ind, data_save_ind)

        # Save solution matrix 
        sol_mat_save = convert(DataFrame, sol_mat)
        CSV.write(dir_save * "Sol_mat.csv", sol_mat_save)

    end

    return tuple(t_vec, quant)
end


"""
    pvc_mixed_quant(model, 
                    mcmc_chains::ChainsMixed, 
                    ind_data_arr::Array{IndData, 1}, 
                    pop_param_info::ParamInfoPop, 
                    ind_param_info_arr::Array{ParamInfoInd, 1},
                    file_loc::FileLocations, 
                    filter_opt, 
                    pop_sampler; 
                    n_runs = 5000,
                    cov_val=Array{FLOAT, 1}(undef, 0), name_cov="", dist_id=1)

Perform posterior visual check by simulating data-set n_runs and create intervall for quantiles. 

Similar to [`pvc_mixed`](@ref), but resimulates observed data set and creates credability 
intervalls for 0.05, 0.5 and 0.95 quantiles and stores in location provided by file_loc. 
"""
function pvc_mixed_quant(model, 
                         mcmc_chains::ChainsMixed, 
                         ind_data_arr::Array{IndData, 1}, 
                         pop_param_info::ParamInfoPop, 
                         ind_param_info_arr::Array{ParamInfoInd, 1},
                         file_loc::FileLocations, 
                         filter_opt, 
                         pop_sampler; 
                         n_runs = 5000,
                         cov_val=Array{FLOAT, 1}(undef, 0), name_cov="", dist_id=1)

    # Replicate observed experiment 
    n_individuals = length(ind_data_arr)
    n_samples = size(mcmc_chains.mean)[2]
    t_vec, dt = calc_t_vec_pcv(ind_data_arr, filter_opt)
    sol_mat = Array{FLOAT, 3}(undef, (model.dim_obs*5, length(t_vec), n_runs))
    for i in 1:n_runs
        t_vec_tmp, quant = pvc_mixed(model, mcmc_chains, ind_data_arr, pop_param_info, 
            ind_param_info_arr, file_loc, filter_opt, pop_sampler, n_runs=n_individuals, pvc_quant=true, 
            cov_val=cov_val, name_cov=name_cov, dist_id = dist_id)

        sol_mat[:, :, i] .= quant
    end

    # Aggregate the median and mean values, note the first
    # three columns are the first state, the second pair is the
    # second state etc.
    quant = zeros(model.dim_obs*10, length(t_vec))
    for i in 1:model.dim_obs
        # Calculate quantiles for the median, 0.05 and 0.95 quantiles 
        i_median = (i-1)*5+1
        i_quant05 = i_median + 1
        i_quant95 = i_median + 2
        i_mean = i_median + 3
        i_std = i_median + 4
        median_i = sol_mat[i_median, :, :]
        quant05_i = sol_mat[i_quant05, :, :]
        quant95_i = sol_mat[i_quant95, :, :]
        mean_i = sol_mat[i_mean, :, :]
        std_i = sol_mat[i_std, :, :]

        # Fixing the quantiles
        for j in 1:length(t_vec)
            # Indices median, quantile 0.05 and 0.95
            i_low_median = (i-1)*10+1
            i_upp_median = i_low_median + 1
            i_low_q05 = i_upp_median + 1
            i_upp_q05 = i_low_q05 + 1
            i_low_q95 = i_upp_q05 + 1
            i_upp_q95 = i_low_q95 + 1
            i_low_mean = i_upp_q95 + 1
            i_upp_mean = i_low_mean + 1
            i_low_std = i_upp_mean + 1
            i_upp_std = i_low_std + 1

            median_use = median_i[j, :]
            quant05_use = quant05_i[j, :]
            quant95_use = quant95_i[j, :]
            mean_use = mean_i[j, :]
            std_use = std_i[j, :]

            quant[i_low_median:i_upp_median, j] = quantile(median_use[.!isnan.(median_use)], [0.025, 0.975])
            quant[i_low_q05:i_upp_q05, j] = quantile(quant05_use[.!isnan.(quant05_use)], [0.025, 0.975])
            quant[i_low_q95:i_upp_q95, j] = quantile(quant95_use[.!isnan.(quant95_use)], [0.025, 0.975])
            quant[i_low_mean:i_upp_mean, j] = quantile(mean_use[.!isnan.(quant95_use)], [0.025, 0.975])
            quant[i_low_std:i_upp_std, j] = quantile(std_use[.!isnan.(std_use)], [0.025, 0.975])
        end
    end

    # Write result to disk 
    dir_save = file_loc.dir_save 

    if name_cov != ""
            cov_tag = name_cov * string(cov_val[1])
        else
            cov_tag = ""
    end

    path_save = dir_save * "Pvc_quant" * cov_tag * ".csv"

    full_data = vcat(quant, collect(t_vec)')'
    data_save = DataFrame(full_data)
    # Ensure correct column-names 
    col_names = Array{String, 1}(undef, model.dim_obs*10 + 1)
    for i in 1:model.dim_obs
        med_tag = "y" * string(i) * "_med" .* ["_low", "_upp"]
        q1_tag = "y" * string(i) * "_qu05" .* ["_low", "_upp"]
        q2_tag = "y" * string(i) * "_qu95" .* ["_low", "_upp"]
        mean_tag = "y" * string(i) * "_mean" .* ["_low", "_upp"]
        std_tag = "y" * string(i) * "_std" .* ["_low", "_upp"]

        i_start = (i - 1)*6 + 1
        col_names[i_start+0:i_start+1] .= med_tag
        col_names[i_start+2:i_start+3] .= q1_tag
        col_names[i_start+4:i_start+5] .= q2_tag
        col_names[i_start+6:i_start+7] .= mean_tag
        col_names[i_start+8:i_start+9] .= std_tag
    end
    col_names[end] = "time"
    rename!(data_save, col_names)
    CSV.write(path_save, data_save)

end


function pvc_mixed_mean_post(model, 
                             dist_use, 
                             pop_param_info::ParamInfoPop, 
                             ind_param_info,
                             file_loc::FileLocations, 
                             filter_opt, 
                             t_vec, 
                             kappa_vec;
                             n_runs=20000, 
                             pvc_quant=false,
                             tag_save = "")
                   

    # Calculate time-span to simulate within 
    dt = filter_opt.delta_t
    time_span = (t_vec[1], t_vec[end])

    # Matrix for storing simulated results 
    sol_mat = zeros(model.dim_obs*n_runs, length(t_vec))
    if typeof(model) <: SdeModel
        u0_vec = Array{FLOAT, 1}(undef, model.dim)
    else
        u0_vec = Array{UInt16, 1}(undef, model.dim)
    end

    # Array for holding model-parameters 
    log_kappa = pop_param_info.log_pop_param_kappa
    log_sigma = pop_param_info.log_pop_param_sigma
    n_kappa_sigma = length(vcat(log_kappa, log_sigma))
    sigma_index = [(i > length(log_kappa) ? true : false) for i in 1:n_kappa_sigma]
    kappa_index = .!sigma_index

    # Correct transformation of individual parameters 
    ind_param_info_arr = init_param_info_arr(ind_param_info, dist_use, 1)
    log_ind_param = ind_param_info_arr[1].log_ind_param

    # Mcmc-chains 
    pop_kappa = kappa_vec

    n_ind_param = length(dist_use)

    # Array for model-parameters 
    model_param = init_model_parameters(zeros(n_ind_param), [0.0], model, kappa=pop_kappa)
    map_kappa_to_mod_param_mixed!(model_param, log_kappa, pop_kappa)

    # Run appropriate number of times 
    i = 1
    run_i = 0
    while i < n_runs && run_i < n_runs * 10

        # Draw new individual parameters and populate model-param-struct 
        new_param = rand(dist_use, 1)[:, 1]
        update_ind_param_ind!(model_param.individual_parameters, new_param, log_ind_param)

        # Guard against crazy NUTS-sampler
        if sum(abs.(model_param.individual_parameters.c) .== Inf) > 0
            run_i += 1
            continue
        end

        local t_vec1
        local u_mat1
        try 
            model.calc_x0!(u0_vec, model_param.individual_parameters)
            t_vec1, u_mat1 = solve_model(model, time_span, u0_vec, model_param.individual_parameters, dt)        
        catch
            run_i += 1
            continue
        end
        
        # Guard against NaN
        if sum(isnan.(u_mat1)) > 0
            run_i += 1
            continue
        end
        
        i_low = model.dim_obs * i - (model.dim_obs - 1)
        i_upp = model.dim_obs * i

        # Calculate observed output 
        y_vec = Array{Float64, 1}(undef, model.dim_obs)
        for j in 1:length(t_vec1)
            model.calc_obs(y_vec, u_mat1[1:model.dim, j], model_param.individual_parameters, 0.0)
            sol_mat[i_low:i_upp, j] .= y_vec
        end

        i += 1
        run_i += 1
    end

    if run_i > n_runs * 10
        @printf("Problem in pvc (many failed runs)\n")
    end

    # Aggregate the median and mean values, note the first
    # three columns are the first state, the second pair is the
    # second state etc.
    quant = zeros(model.dim_obs*5, length(t_vec))
    for i in 1:model.dim_obs
        i_rows = i:model.dim_obs:(n_runs*model.dim_obs)
        start_index = (i - 1) *3 + 1
        quant[start_index, :] = median(sol_mat[i_rows, :], dims=1)
        # Fixing the quantiles
        for j in 1:length(t_vec)
            i_low = start_index+1
            i_upp = start_index+2
            quant[i_low:i_upp, j] = quantile(sol_mat[i_rows, j], [0.05, 0.95])
        end
        # Fixing mean and std 
        quant[start_index + 3, :] = median(sol_mat[i_rows, :], dims=1)
        quant[start_index + 4, :] = std(sol_mat[i_rows, :], dims=1)
    end

    # Save result Correctly
    if pvc_quant == false
        # Ensure directory for storing results exist 
        dir_save = file_loc.dir_save 
        println("dir_save = $dir_save")
        path_save = dir_save * "Pvc" * tag_save * ".csv"
        full_data = vcat(quant, collect(t_vec)')'
        data_save = DataFrame(full_data)
        # Ensure correct column-names 
        col_names = Array{String, 1}(undef, model.dim_obs*5 + 1)
        for i in 1:model.dim_obs
            med_tag = "y" * string(i) * "_med"
            q1_tag = "y" * string(i) * "_qu05"
            q2_tag = "y" * string(i) * "_qu95"
            mean_tag = "y" * string(i) * "_mean"
            std_tag = "y" * string(i) * "_std"

            col_names[(i-1)*3+1] = med_tag
            col_names[(i-1)*3+2] = q1_tag
            col_names[(i-1)*3+3] = q2_tag
            col_names[(i-1)*3+4] = mean_tag
            col_names[(i-1)*3+5] = std_tag
        end
        col_names[end] = "time"
    
        rename!(data_save, col_names)
        CSV.write(path_save, data_save)

    end

    return tuple(t_vec, quant)
end


function pvc_mixed_quant_mean(model, 
                              dist_use, 
                              pop_param_info::ParamInfoPop, 
                              ind_param_info,
                              file_loc::FileLocations, 
                              filter_opt, 
                              t_vec, 
                              n_individuals, 
                              kappa_vec;
                              n_runs=5000, 
                              tag_save="")

    # Replicate observed experiment 
    sol_mat = Array{FLOAT, 3}(undef, (model.dim_obs*5, length(t_vec), n_runs))
    for i in 1:n_runs
        t_vec_tmp, quant = pvc_mixed_mean_post(model, dist_use, pop_param_info, ind_param_info,
            file_loc, filter_opt, t_vec, kappa_vec, n_runs=n_individuals, pvc_quant=true)

        sol_mat[:, :, i] .= quant
    end

    # Aggregate the median and mean values, note the first
    # three columns are the first state, the second pair is the
    # second state etc.
    quant = zeros(model.dim_obs*10, length(t_vec))
    for i in 1:model.dim_obs
        # Calculate quantiles for the median, 0.05 and 0.95 quantiles 
        i_median = (i-1)*5+1
        i_quant05 = i_median + 1
        i_quant95 = i_median + 2
        i_mean = i_median + 3
        i_std = i_median + 4
        median_i = sol_mat[i_median, :, :]
        quant05_i = sol_mat[i_quant05, :, :]
        quant95_i = sol_mat[i_quant95, :, :]
        mean_i = sol_mat[i_mean, :, :]
        std_i = sol_mat[i_std, :, :]

        # Fixing the quantiles
        for j in 1:length(t_vec)
            # Indices median, quantile 0.05 and 0.95
            i_low_median = (i-1)*10+1
            i_upp_median = i_low_median + 1
            i_low_q05 = i_upp_median + 1
            i_upp_q05 = i_low_q05 + 1
            i_low_q95 = i_upp_q05 + 1
            i_upp_q95 = i_low_q95 + 1
            i_low_mean = i_upp_q95 + 1
            i_upp_mean = i_low_mean + 1
            i_low_std = i_upp_mean + 1
            i_upp_std = i_low_std + 1

            median_use = median_i[j, :]
            quant05_use = quant05_i[j, :]
            quant95_use = quant95_i[j, :]
            mean_use = mean_i[j, :]
            std_use = std_i[j, :]

            quant[i_low_median:i_upp_median, j] = quantile(median_use[.!isnan.(median_use)], [0.025, 0.975])
            quant[i_low_q05:i_upp_q05, j] = quantile(quant05_use[.!isnan.(quant05_use)], [0.025, 0.975])
            quant[i_low_q95:i_upp_q95, j] = quantile(quant95_use[.!isnan.(quant95_use)], [0.025, 0.975])
            quant[i_low_mean:i_upp_mean, j] = quantile(mean_use[.!isnan.(quant95_use)], [0.025, 0.975])
            quant[i_low_std:i_upp_std, j] = quantile(std_use[.!isnan.(std_use)], [0.025, 0.975])
        end
    end

    # Write result to disk 
    dir_save = file_loc.dir_save 

    path_save = dir_save * "Pvc_quant" * tag_save * ".csv"

    full_data = vcat(quant, collect(t_vec)')'
    data_save = DataFrame(full_data)
    # Ensure correct column-names 
    col_names = Array{String, 1}(undef, model.dim_obs*10 + 1)
    for i in 1:model.dim_obs
        med_tag = "y" * string(i) * "_med" .* ["_low", "_upp"]
        q1_tag = "y" * string(i) * "_qu05" .* ["_low", "_upp"]
        q2_tag = "y" * string(i) * "_qu95" .* ["_low", "_upp"]
        mean_tag = "y" * string(i) * "_mean" .* ["_low", "_upp"]
        std_tag = "y" * string(i) * "_std" .* ["_low", "_upp"]

        i_start = (i - 1)*10 + 1
        col_names[i_start+0:i_start+1] .= med_tag
        col_names[i_start+2:i_start+3] .= q1_tag
        col_names[i_start+4:i_start+5] .= q2_tag
        col_names[i_start+6:i_start+7] .= mean_tag
        col_names[i_start+8:i_start+9] .= std_tag
    end
    col_names[end] = "time"
    rename!(data_save, col_names)
    CSV.write(path_save, data_save)

end