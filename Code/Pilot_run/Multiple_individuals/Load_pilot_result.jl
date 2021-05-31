#= 
    Functions for reading pilot-run data from disk for multiple individuals. 
=#


"""
init_param_pilot(pop_param_info::ParamInfoPop, 
                 ind_param_info::ParamInfoIndPre, 
                 file_loc::FileLocations,
                 exp_id::T1) where T1<:Signed

Set starting values for population parameters and individual parameters from pilot-run with id exp_id. 

Changes the starting values in user provided pop_param_info (see [`init_pop_param_info`](@ref))
to those of pilot run exp_id (exits if pilot run does not exist). Further, instead of using 
starting values in ind_param_info using starting values from pilot-run when creating ind_param_info_arr 
(see [`init_param_info_arr`]). Dist-ind param is required to ensure that results follow a correct distribution.  
Returns pop_param_info-struct and ind_param_info_arr-array with starting values from pilot. 

See also: [`init_pop_param_info`, `init_param_info_arr`](@ref)
"""
function init_param_pilot(pop_param_info::ParamInfoPop, 
                          ind_param_info::ParamInfoIndPre, 
                          file_loc::FileLocations,
                          dist_ind_param, 
                          exp_id::T1;
                          sampler::String="standrad") where T1<:Signed

    if sampler == "standrad"
        dir_pilot = file_loc.dir_save * "/Pilot_run_data/"
    elseif sampler == "alt"
        dir_pilot = file_loc.dir_save * "/Pilot_run_data_alt/"
    end

    # Check that pilot-run results exist 
    
    if !isfile( dir_pilot * "/Pilot_info.csv")
        @printf("Error: Pilot-run with tag1 does not exist")
    end
    data_info = CSV.read(dir_pilot * "/Pilot_info.csv", DataFrame)
    if !(exp_id ∈ data_info[!, "id"])
        @printf("Error: Pilot data does not exist for experiment tag %d\n", exp_id)
    end

    # Directory for pilot-run results 
    dir_param = dir_pilot * "Exp_tag" * string(exp_id) * "/"
    if !isdir(dir_param)
        @printf("Error: Directory with pilot-results does not exist")
    end

    # Get ranges for kappa and sigma when mapping 
    n_kappa::Int64 = length(pop_param_info.prior_pop_param_kappa)
    n_sigma::Int64 = length(pop_param_info.prior_pop_param_sigma)

    # Parameters 
    corr_mat = convert(Array{FLOAT, 2}, CSV.read(dir_param * "Corr.csv", DataFrame))
    mean_vec = convert(Array{FLOAT, 2}, CSV.read(dir_param * "Mean.csv", DataFrame))
    scale_vec = convert(Array{FLOAT, 2}, CSV.read(dir_param * "Scale.csv", DataFrame))
    kappa_sigma = convert(Array{FLOAT, 2}, CSV.read(dir_param * "Kappa_sigma.csv", DataFrame))
    ind_param = convert(Array{FLOAT, 2}, CSV.read(dir_param * "Ind.csv", DataFrame))

    # Population parameters 
    pop_info_new::ParamInfoPop = deepcopy(pop_param_info)

    println(pop_info_new.init_pop_param_mean)
    println(mean_vec)


    pop_info_new.init_pop_param_corr .= corr_mat
    pop_info_new.init_pop_param_mean .= mean_vec[1, :]
    pop_info_new.init_pop_param_scale .= scale_vec[1, :]
    if n_kappa != 0
        pop_info_new.init_pop_param_kappa .= kappa_sigma[1:n_kappa]
    end
    if n_sigma != 0
        pop_info_new.init_pop_param_sigma .= kappa_sigma[(n_kappa+1):end]
    end

    # Fix the individual parameters 
    n_individuals = size(ind_param)[1]
    
    ind_param_info_arr = init_param_info_arr(ind_param_info, dist_ind_param, n_individuals)
    for i in 1:n_individuals
        ind_param_info_arr[i].init_ind_param .= ind_param[i, 1:end-1]
    end

    return pop_info_new, ind_param_info_arr
end


"""
    change_pop_par_info(tune_part_data::TuneParticlesMixed, pop_param_info::ParamInfoPop)::ParamInfoPop

Change initial values in ParamInfoPop-struct and return new struct. Required when tuning-particles. 
"""
function change_pop_par_info(tune_part_data::TuneParticlesMixed, pop_param_info::ParamInfoPop)::ParamInfoPop
    
    pop_par_info_new = deepcopy(pop_param_info)
    pop_par_info_new.init_pop_param_mean .= tune_part_data.init_mean
    pop_par_info_new.init_pop_param_scale .= tune_part_data.init_scale
    pop_par_info_new.init_pop_param_kappa .= tune_part_data.init_kappa
    pop_par_info_new.init_pop_param_sigma .= tune_part_data.init_sigma

    return pop_par_info_new
end


"""
    init_mcmc_sampler_pilot(mcmc_sampler_ind, mcmc_sampler_pop, file_loc, n_individuals::T, exp_tag::T) where T<:Signed

Initalise kappa-sigma and individual-parameters mcmc-samplers from pilot-run with exp_tag (integer). 

User provides the type of mcmc_sampler_ind (individual) and mcmc_sampler_pop (kappa-sigma). If pilot-run data does 
not exist for provided sampler types program exits. 
"""
function init_mcmc_sampler_pilot(mcmc_sampler_ind, mcmc_sampler_pop, file_loc, 
    n_individuals::T, exp_tag::T) where T<:Signed

    dir_mcmc_samplers = file_loc.dir_save * "/Pilot_run_data/Exp_tag" * string(exp_tag) * 
        "/Mcmc_param/"

    if !isdir(dir_mcmc_samplers)
        @printf("Error: Mcmc-sampler-param does not exist for exp_tag %d\n", exp_tag)
        exit(1)
    end
    
    # Array with individual samplers 
    sampler_use_ind = calc_sampler_type(mcmc_sampler_ind)
    mcmc_sampler_ind_arr::Array{typeof(mcmc_sampler_ind), 1} = Array{typeof(mcmc_sampler_ind), 1}(undef, n_individuals)
    for i in 1:n_individuals
        if typeof(mcmc_sampler_ind) <: McmcSamplerRandWalk
            mcmc_sampler_ind_arr[i] = init_mcmc_pilot(sampler_use_ind, file_loc, 0.0, 
                ind_id=string(i), dir_files=dir_mcmc_samplers)
        else
            mcmc_sampler_ind_arr[i] = init_mcmc_pilot(sampler_use_ind, file_loc, 0.0, 
                ind_id=string(i), dir_files=dir_mcmc_samplers, steps_before_update=mcmc_sampler_ind.steps_before_update, 
                update_it=mcmc_sampler_ind.update_it)
        end
    end
    # Kappa sigma sampler 
    sampler_use_kappa_sigma = calc_sampler_type(mcmc_sampler_pop)
    if typeof(mcmc_sampler_pop) <: McmcSamplerRandWalk
        mcmc_sampler_pop = init_mcmc_pilot(sampler_use_kappa_sigma, file_loc, 0.0, ind_id="Kappa_sigma", dir_files=dir_mcmc_samplers)
    else
        mcmc_sampler_pop = init_mcmc_pilot(sampler_use_kappa_sigma, file_loc, 0.0, ind_id="Kappa_sigma", dir_files=dir_mcmc_samplers, 
            update_it = mcmc_sampler_pop.update_it)
    end

    return tuple(mcmc_sampler_ind_arr, mcmc_sampler_pop)
end
"""
    When sampler pop is an array 
"""
function init_mcmc_sampler_pilot(mcmc_sampler_ind, mcmc_sampler_pop, file_loc, 
    n_individuals::T, exp_tag::T, sampler_type::String) where T<:Signed

    dir_mcmc_samplers = file_loc.dir_save * "/Pilot_run_data_alt/Exp_tag" * string(exp_tag) * 
        "/Mcmc_param/"

    if !isdir(dir_mcmc_samplers)
        @printf("Error: Mcmc-sampler-param does not exist for exp_tag %d\n", exp_tag)
        exit(1)
    end
    
    # Array with individual samplers 
    sampler_use_ind = calc_sampler_type(mcmc_sampler_ind)
    mcmc_sampler_ind_arr::Array{typeof(mcmc_sampler_ind), 1} = Array{typeof(mcmc_sampler_ind), 1}(undef, n_individuals)
    for i in 1:n_individuals
        if typeof(mcmc_sampler_ind) <: McmcSamplerRandWalk
            mcmc_sampler_ind_arr[i] = init_mcmc_pilot(sampler_use_ind, file_loc, 0.0, 
                ind_id=string(i), dir_files=dir_mcmc_samplers)
        else
            mcmc_sampler_ind_arr[i] = init_mcmc_pilot(sampler_use_ind, file_loc, 0.0, 
                ind_id=string(i), dir_files=dir_mcmc_samplers, steps_before_update=mcmc_sampler_ind.steps_before_update, 
                update_it=mcmc_sampler_ind.update_it)
        end
    end
    # Kappa sigma sampler 
    sampler_use_kappa_sigma = calc_sampler_type(mcmc_sampler_pop)
    mcmc_sampler_pop_arr::Array{typeof(mcmc_sampler_pop), 1} = Array{typeof(mcmc_sampler_pop), 1}(undef, n_individuals)
    for i in 1:n_individuals
        ind_id = "Kappa_sigma" * string(i)
        if typeof(mcmc_sampler_pop) <: McmcSamplerRandWalk
            mcmc_sampler_pop_arr[i] = init_mcmc_pilot(sampler_use_kappa_sigma, file_loc, 0.0, ind_id=ind_id, dir_files=dir_mcmc_samplers)
        else
            mcmc_sampler_pop_arr[i] = init_mcmc_pilot(sampler_use_kappa_sigma, file_loc, 0.0, ind_id=ind_id, dir_files=dir_mcmc_samplers, 
                update_it = mcmc_sampler_pop.update_it)
        end
    end

    return tuple(mcmc_sampler_ind_arr, mcmc_sampler_pop_arr)
end



"""
    init_filter_arr_pilot(filter_opt, 
                          file_loc::FileLocations, 
                          n_individuals::T, 
                          exp_id::T) where T<:Signed

Initalise array of filters of type filter_opt using particles from pilot-run with tag exp_id.  

If pilot-run data does not exist for provided filter exits. 
"""
function init_filter_arr_pilot(filter_opt, 
                               file_loc::FileLocations, 
                               n_individuals::T, 
                               exp_id::T;
                               sampler::String="standard") where T<:Signed

    # Check if particle tuning results exist 
    if sampler == "standard"
        dir_data = file_loc.dir_save * "Pilot_run_data/Exp_tag" * string(exp_id) * "/Tune_particles/"
    elseif sampler == "alt"
        dir_data = file_loc.dir_save * "Pilot_run_data_alt/Exp_tag" * string(exp_id) * "/Tune_particles/"
    end
    file_name = dir_data * "Rho" * replace(string(filter_opt.rho), "." => "d") * ".csv"
    if !isfile(file_name)
        @printf("Error: File with particle tuning does not exist for rho = %.3f\n", filter_opt.rho)
    end

    # Create filters 
    data_particles = convert(Array{Int64, 2}, CSV.read(file_name, DataFrame))[1, :]
    filter_arr::Array{typeof(filter_opt), 1} = Array{typeof(filter_opt), 1}(undef, n_individuals)
    for i in 1:n_individuals
        filter_i = change_filter_opt(filter_opt, data_particles[i], filter_opt.rho)
        filter_arr[i] = filter_i
    end

    return filter_arr
end
