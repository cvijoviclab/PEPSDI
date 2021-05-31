using Distributions
using Printf
using LinearAlgebra
using DelimitedFiles


# TODO: Add pilot-run init for random-walk proposal 


"""
    calc_dim_sampler(param_info)

Calculate dimension of sampler depending on param_info type input

Param-info can either be InitParameterInfo (for individual inference), 
ParamInfoInd (individual parameters in Gibbs-sampler) and ParamInfoPop
(population parameters in mixed effects inference)
"""
function calc_dim_sampler(param_info)
    dim::Int64 = 0
    if typeof(param_info) <: InitParameterInfo
        dim = length(param_info.prior_error_param) + length(param_info.prior_ind_param)
    elseif typeof(param_info) <: ParamInfoIndPre
        dim = param_info.n_param
    elseif typeof(param_info) <: ParamInfoInd
        dim = param_info.n_param
    elseif typeof(param_info) <: ParamInfoPop
        dim = param_info.dim_sigma + param_info.dim_kappa
    end

    return dim 
end


"""
    calc_sampler_type(mcmc_sampler)

Calculate mcmc-sampler to use when initialising mcmc-samplers in Gibbs-sampler. 

Based on sampler-provided by the user (either for individual parameters or kappa-sigma)
inferes provided sampler type. 
"""
function calc_sampler_type(mcmc_sampler)
    # Ensure correct mcmc-sampler is used 
    if typeof(mcmc_sampler) <: McmcSamplerGenAM
        sampler_use = GenAmSampler()
    elseif typeof(mcmc_sampler) <: McmcSamplerAM
        sampler_use = AmSampler()
    elseif typeof(mcmc_sampler) <: McmcSamplerRam
        sampler_use = RamSampler()
    elseif typeof(mcmc_sampler) <: McmcSamplerRandWalk
        sampler_use = RandomWalk()
    end

    return sampler_use
end


"""
    calc_mu_vec(param_info)

    Calculate μ of sampler depending on param_info type input (see below)

For individual parameters in mixed effects model mu_vec is Initalised as 
zeros as it is later filled in the run mcmc-function. 

See also: [`calc_dim_sampler`](@ref)
"""
function calc_mu_vec(param_info::InitParameterInfo)
    mu_vec::Array{FLOAT, 1} = vcat(param_info.init_ind_param, param_info.init_error_param)
    return mu_vec
end
function calc_mu_vec(param_info::ParamInfoIndPre)
    mu_vec::Array{FLOAT, 1} = zeros(param_info.n_param)
    return mu_vec
end
function calc_mu_vec(param_info::ParamInfoInd)
    mu_vec::Array{FLOAT, 1} = param_info.init_ind_param
    return mu_vec
end
function calc_mu_vec(param_info::ParamInfoPop)
    mu_vec::Array{FLOAT, 1} = vcat(param_info.init_pop_param_kappa, param_info.init_pop_param_sigma)
    return mu_vec
end
    

"""
    init_mcmc(sampler::RandomWalk, param_info; step_length=0.1, cov_mat=I)

Create (adapative) mcmc-sampler struct used for proposing new parameters. 

Param_info object contains information regarding number of parameters to sample. 
For random-walk sampler the identity matrix with step-length 0.1 is used by 
deafult when proposing new parameters. 

See also: [`McmcSamplerRandWalk`, `InitParameterInfo`](@ref)
"""
function init_mcmc(sampler::RandomWalk, param_info; step_length=0.1, cov_mat=I)

    # Dimension adapted for which parameter sampler is created 
    dim = calc_dim_sampler(param_info)

    # If the identity matrix is used for covariance 
    step_length::FLOAT = convert(FLOAT, step_length)
    cov_mat_use::Array{FLOAT, 2} = Array{FLOAT, 2}(I, (dim, dim))
    if cov_mat != I
        cov_mat_use::Array{FLOAT, 2} .= cov_mat
    end

    cov_mat_old::Array{FLOAT, 2} = deepcopy(cov_mat_use)
    mu_start::Array{FLOAT, 1} = calc_mu_vec(param_info)

    mcmc_sampler = McmcSamplerRandWalk(cov_mat_use, 
                                       step_length, 
                                       dim, 
                                       "Rand_walk_sampler", 
                                       cov_mat_old, 
                                       mu_start)

    return mcmc_sampler
end
"""
    init_mcmc(sampler::AmSampler, param_info; cov_mat=I, gamma0=1.0, 
        alpha_power=0.7, step_before_update=100)

For AM-sampler default parameters are adapted from Wiqvist et al and based 
on recomendation by Andrieu and Thoms. 

See also: [`McmcSamplerAM`](@ref)
"""
function init_mcmc(sampler::AmSampler, param_info; cov_mat=I, gamma0=1.0, 
    alpha_power=0.7, step_before_update=100, update_it=1, lambda=1.0)

    # Dimension adapted for which parameter sampler is created 
    dim::Int64 = calc_dim_sampler(param_info)

    cov_mat_use::Array{FLOAT, 2} = Array{FLOAT, 2}(I, (dim, dim))
    if cov_mat != I
        cov_mat_use::Array{FLOAT, 2} .= cov_mat
    end

    gamma0::FLOAT = convert(FLOAT, gamma0)
    alpha_power::FLOAT = convert(FLOAT, alpha_power)
    
    # Adapt mu depending sampler
    lambda::FLOAT = lambda
    mu_start::Array{FLOAT, 1} = calc_mu_vec(param_info)
    
    mcmc_sampler = McmcSamplerAM(cov_mat_use, mu_start, gamma0, alpha_power, lambda, 
        step_before_update, dim, "Am_sampler", update_it)

    return mcmc_sampler
end
"""
    init_mcmc(sampler::GenAmSampler, param_info; cov_mat=I, 
        steps_before_update=100, alpha_target=0.07, gamma0=1.0, alpha_power=0.7)

For general AM-sampler default parameters are adapted from Wiqvist et al. 

See also: [`McmcSamplerGenAM`](@ref)
"""
function init_mcmc(sampler::GenAmSampler, param_info; cov_mat=I, 
    step_before_update=100, alpha_target=0.07, gamma0=1.0, alpha_power=0.7, update_it=1, log_lambda=log(1.0))

    # Dimension adapted for which parameter sampler is created 
    dim::Int64 = calc_dim_sampler(param_info)

    cov_mat_use::Array{FLOAT, 2} = Array{FLOAT, 2}(I, (dim, dim))
    if cov_mat != I
        cov_mat_use::Array{FLOAT, 2} .= cov_mat
    end

    gamma0::FLOAT = convert(FLOAT, gamma0)
    alpha_power::FLOAT = convert(FLOAT, alpha_power)
    alpha_target::FLOAT = convert(FLOAT, alpha_target)

    # lambda = 2.38^2 / dim-of-sampling 
    log_lambda_start::FLOAT = log_lambda
    mu_start::Array{FLOAT, 1} = calc_mu_vec(param_info)
    
    mcmc_sampler_AM = McmcSamplerGenAM(cov_mat_use, mu_start, [log_lambda_start], alpha_target, 
        gamma0, alpha_power, step_before_update, dim, "Gen_am_sampler", update_it)

    return mcmc_sampler_AM
end
"""
    init_mcmc(sampler::RamSampler, param_info; cov_mat=I, 
        steps_before_update=100, alpha_target=0.07, gamma0=1.0, alpha_power=0.7)

For RAM-sampler default parameters are adapted from Vihola et al.  

See also: [`McmcSamplerRam`](@ref)
"""
function init_mcmc(sampler::RamSampler, param_info; cov_mat=I, 
    step_before_update=100, alpha_target=0.07, gamma0=1.0, alpha_power=0.7, update_it=1)

    dim::Int64 = calc_dim_sampler(param_info)

    cov_mat_use::Array{FLOAT, 2} = Array{FLOAT, 2}(I, (dim, dim))
    if cov_mat != I
        cov_mat_use::Array{FLOAT, 2} .= cov_mat
    end

    gamma0::FLOAT = convert(FLOAT, gamma0)
    alpha_power::FLOAT = convert(FLOAT, alpha_power)
    alpha_target::FLOAT = convert(FLOAT, alpha_target)

    # Vector of random numbers required for updating 
    q_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, dim)

    mcmc_sampler_ram = McmcSamplerRam(cov_mat_use, q_vec, alpha_target, gamma0, 
        alpha_power, step_before_update, dim, "Ram_sampler", update_it)
        
    return mcmc_sampler_ram
end


"""
    save_mcmc_opt_pilot(mcmc_sampler::McmcSamplerAM, file_loc, exp_tag, ind_id)

Save mcmc-sampler parameters (e.g Σ) from pilot-run to disk for individual i.  

Required for initalisation of mcmc-sampler via pilot-run. The sampler options are 
stored in the models intermediate-directory in a sub-directory tagged with the 
experimental id. Here, each file is tagged by the individual id. 
For AM-sampler μ and Σ are saved. 

See also: [`init_mcmc_pilot`](@ref)
"""
function save_mcmc_opt_pilot(mcmc_sampler::McmcSamplerAM, file_loc, exp_tag, ind_id; dir_save="")

    if dir_save == ""
        exp_tag_str = string(exp_tag)
        dir_save = file_loc.dir_save * "Pilot_runs/Sampler_opt/Exp_tag" * exp_tag_str * "/"
        if !isdir(dir_save)
            mkpath(dir_save)
        end
    end

    # Write log_lambda, cov_mat and mu to disk 
    file_cov = dir_save * "Cov_mat" * string(ind_id) * ".tsv"
    file_mu = dir_save * "Mu" * string(ind_id) * ".tsv"

    open(file_cov, "w") do io
        writedlm(io, mcmc_sampler.cov_mat)
    end
    open(file_mu, "w") do io
        writedlm(io, mcmc_sampler.mu)
    end
end
"""
For general AM-sampler, μ, Σ and log(λ) are saved. 
"""
function save_mcmc_opt_pilot(mcmc_sampler::McmcSamplerGenAM, file_loc, exp_tag, ind_id; dir_save="")

    if dir_save == ""
        exp_tag_str = string(exp_tag)
        dir_save = file_loc.dir_save * "Pilot_runs/Sampler_opt/Exp_tag" * exp_tag_str * "/"
        if !isdir(dir_save)
            mkpath(dir_save)
        end
    end

    # Write log_lambda, cov_mat and mu to disk 
    file_lambda = dir_save * "Log_lambda" * string(ind_id) * ".tsv"
    file_cov = dir_save * "Cov_mat" * string(ind_id) * ".tsv"
    file_mu = dir_save * "Mu" * string(ind_id) * ".tsv"

    open(file_lambda, "w") do io
        writedlm(io, mcmc_sampler.log_lambda)
    end
    open(file_cov, "w") do io
        writedlm(io, mcmc_sampler.cov_mat)
    end
    open(file_mu, "w") do io
        writedlm(io, mcmc_sampler.mu)
    end
end
"""
For RAM-sampler Σ is saved. 
"""
function save_mcmc_opt_pilot(mcmc_sampler::McmcSamplerRam, file_loc, exp_tag, ind_id; dir_save="")

    if dir_save == ""
        exp_tag_str = string(exp_tag)
        dir_save = file_loc.dir_save * "Pilot_runs/Sampler_opt/Exp_tag" * exp_tag_str * "/"
        if !isdir(dir_save)
            mkpath(dir_save)
        end
    end

    # Write log_lambda, cov_mat and mu to disk 
    file_cov = dir_save * "Cov_mat" * string(ind_id) * ".tsv"

    open(file_cov, "w") do io
        writedlm(io, mcmc_sampler.cov_mat)
    end
end
"""
For Random-walk-sampler Σ is saved. 
"""
function save_mcmc_opt_pilot(mcmc_sampler::McmcSamplerRandWalk, file_loc, exp_tag, ind_id; dir_save="")

    if dir_save == ""
        exp_tag_str = string(exp_tag)
        dir_save = file_loc.dir_save * "Pilot_runs/Sampler_opt/Exp_tag" * exp_tag_str * "/"
        if !isdir(dir_save)
            mkpath(dir_save)
        end
    end

    # Write log_lambda, cov_mat and mu to disk 
    file_cov = dir_save * "Cov_mat" * string(ind_id) * ".tsv"

    open(file_cov, "w") do io
        writedlm(io, mcmc_sampler.cov_mat_old)
    end
end


"""
    init_mcmc_pilot(sampler::AmSampler, file_loc, rho; steps_before_update=100, 
           alpha_target=0.07, gamma0=1.0, alpha_power=0.7, exp_tag=1, ind_id=1)

Create (adapative) mcmc-sampler struct using parameters (e.g Σ) from pilot run. 

Information for finding pilot-run data is in file-locations and correlation level 
rho. For AM-sampler μ and Σ are taken from the pilot-run. All other parameters 
are described in init_mcmc

See also: [`init_mcmc`, `FileLocations`](@ref)
"""
function init_mcmc_pilot(sampler::AmSampler, file_loc, rho; steps_before_update=100, 
    alpha_target=0.07, gamma0=1.0, alpha_power=0.7, exp_tag=1, ind_id=1, dir_files="", update_it=1)

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Am_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * string(ind_id) * ".tsv"
    file_mu = dir_files * "Mu" * string(ind_id) * ".tsv"

    if !(isfile(file_cov) && isfile(file_mu))
        @printf("Error: Sampler options for the AM-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat::Array{FLOAT, 2} = readdlm(file_cov, '\t', FLOAT, '\n')
    mu = readdlm(file_mu, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    dim = size(cov_mat)[1]
    mu_use::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, dim)
    mu_use .= mu[:, 1]
    lambda::FLOAT = 2.38^2 / dim
        
    # General sampler variables 
    gamma0::FLOAT = convert(FLOAT, gamma0)
    alpha_power::FLOAT = convert(FLOAT, alpha_power)
    alpha_target::FLOAT = convert(FLOAT, alpha_target)

    mcmc_sampler = McmcSamplerAM(cov_mat, mu_use, gamma0, alpha_power, lambda, 
        steps_before_update, dim, "Am_sampler", update_it)

    return mcmc_sampler
end
function init_mcmc_pilot(sampler::McmcSamplerAM, file_loc, rho; exp_tag=1, ind_id=1, dir_files="")

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Am_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * string(ind_id) * ".tsv"
    file_mu = dir_files * "Mu" * string(ind_id) * ".tsv"

    if !(isfile(file_cov) && isfile(file_mu))
        @printf("Error: Sampler options for the AM-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat::Array{FLOAT, 2} = readdlm(file_cov, '\t', FLOAT, '\n')
    mu = readdlm(file_mu, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    dim = size(cov_mat)[1]
    mu_use::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, dim)
    mu_use .= mu[:, 1]
    lambda::FLOAT = 2.38^2 / dim
        
    # General sampler variables 
    mcmc_sampler = deepcopy(sampler)
    mcmc_sampler.cov_mat .= cov_mat
    mcmc_sampler.mu .= mu_use

    return mcmc_sampler
end
"""
    init_mcmc_pilot(sampler::GenAmSampler, file_loc, rho; steps_before_update=100, 
        alpha_target=0.233, gamma0=1.0, alpha_power=0.7, exp_tag=1, ind_id=1)


For general AM-sampler μ, log(λ) and Σ are taken from the pilot-run. All other parameters 
are described in init_mcmc

See also: [`init_mcmc`, `FileLocations`](@ref)
"""
function init_mcmc_pilot(sampler::GenAmSampler, file_loc, rho; steps_before_update::T=100, 
    alpha_target=0.07, gamma0=1.0, alpha_power=0.7, exp_tag::T=1, ind_id::String="1", 
    dir_files="", update_it=1) where T<:Signed

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Gen_am_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * ind_id * ".tsv"
    file_lambda = dir_files * "Log_lambda" * ind_id * ".tsv"
    file_mu = dir_files * "Mu" * ind_id * ".tsv"

    if !(isfile(file_cov) && isfile(file_lambda) && isfile(file_lambda))
        @printf("Error: Sampler options for the General AM-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat::Array{FLOAT, 2} = readdlm(file_cov, '\t', FLOAT, '\n')
    mu = readdlm(file_mu, '\t', FLOAT, '\n')
    log_lambda = readdlm(file_lambda, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    log_lambda_use::Array{FLOAT, 1} = [convert(FLOAT, log_lambda[1])]
    dim::Int64 = size(cov_mat)[1]
    mu_use::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, dim)
    mu_use .= mu[:, 1]
        
    # General sampler variables 
    gamma0::FLOAT = convert(FLOAT, gamma0)
    alpha_power::FLOAT = convert(FLOAT, alpha_power)
    alpha_target::FLOAT = convert(FLOAT, alpha_target)

    mcmc_sampler_AM = McmcSamplerGenAM(cov_mat, mu_use, log_lambda_use, alpha_target, 
        gamma0, alpha_power, steps_before_update, dim, "Gen_am_sampler", update_it)

    return mcmc_sampler_AM
end
function init_mcmc_pilot(sampler::McmcSamplerGenAM, file_loc, rho; exp_tag::T=1, ind_id::String="1", dir_files="") where T<:Signed

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Gen_am_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * ind_id * ".tsv"
    file_lambda = dir_files * "Log_lambda" * ind_id * ".tsv"
    file_mu = dir_files * "Mu" * ind_id * ".tsv"

    if !(isfile(file_cov) && isfile(file_lambda) && isfile(file_lambda))
        @printf("Error: Sampler options for the General AM-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat::Array{FLOAT, 2} = readdlm(file_cov, '\t', FLOAT, '\n')
    mu = readdlm(file_mu, '\t', FLOAT, '\n')
    log_lambda = readdlm(file_lambda, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    log_lambda_use::Array{FLOAT, 1} = [convert(FLOAT, log_lambda[1])]
    dim::Int64 = size(cov_mat)[1]
    mu_use::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, dim)
    mu_use .= mu[:, 1]
        
    # General sampler variables 
    mcmc_sampler_AM = deepcopy(sampler)
    mcmc_sampler_AM.cov_mat .= cov_mat
    mcmc_sampler_AM.log_lambda .= log_lambda_use
    mcmc_sampler_AM.mu .= mu_use

    return mcmc_sampler_AM
end
"""
    init_mcmc_pilot(sampler::RamSampler, file_loc, rho; steps_before_update=100, 
        alpha_target=0.233, gamma0=1.0, alpha_power=0.7, exp_tag=1, ind_id=1)


For RAM-sampler is Σ taken from the pilot-run. All other parameters 
are described in init_mcmc

See also: [`init_mcmc`, `FileLocations`](@ref)
"""
function init_mcmc_pilot(sampler::RamSampler, file_loc, rho; steps_before_update=100, 
    alpha_target=0.07, gamma0=1.0, alpha_power=0.7, exp_tag=1, ind_id=1, dir_files="", update_it=1)

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Ram_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * string(ind_id) * ".tsv"

    if !isfile(file_cov)
        @printf("Error: Sampler options for the RAM-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat = readdlm(file_cov, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    dim = size(cov_mat)[1]

    # Vector of random-numbers 
    q_vec = Array{FLOAT, 1}(undef, dim)
            
    # General sampler variables 
    gamma0 = convert(FLOAT, gamma0)
    alpha_power = convert(FLOAT, alpha_power)
    alpha_target = convert(FLOAT, alpha_target)

    mcmc_sampler_ram = McmcSamplerRam(cov_mat, q_vec, alpha_target, gamma0, 
        alpha_power, steps_before_update, dim, "Ram_sampler", update_it)

    return mcmc_sampler_ram
end
function init_mcmc_pilot(sampler::McmcSamplerRam, file_loc, rho; exp_tag=1, ind_id=1, dir_files="")

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Ram_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * string(ind_id) * ".tsv"

    if !isfile(file_cov)
        @printf("Error: Sampler options for the RAM-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat = readdlm(file_cov, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    dim = size(cov_mat)[1]

    mcmc_sampler_ram = deepcopy(sampler)
    mcmc_sampler_ram.cov_mat .= cov_mat

    return mcmc_sampler_ram
end
"""
    init_mcmc_pilot(sampler::RandomWalk, file_loc, rho; step_length=0.1, exp_tag=1, ind_id=1, dir_files="")


For random-walk-sampler is Σ taken from the pilot-run. All other parameters 
are described in init_mcmc

See also: [`init_mcmc`, `FileLocations`](@ref)
"""
function init_mcmc_pilot(sampler::RandomWalk, file_loc, rho; step_length=0.1, exp_tag=1, ind_id=1, dir_files="")

    # Check if pilot-run data exists 
    if dir_files == ""
        if rho != 0
            tag_corr = "/Correlated/"
        elseif rho == 0
            tag_corr = "/Not_correlated/"
        end
        exp_tag = "Exp_tag" * string(exp_tag) * "/"
        dir_files = file_loc.dir_save * tag_corr * "Rand_walk_sampler/Pilot_runs/Sampler_opt/" * exp_tag
    end

    file_cov = dir_files * "Cov_mat" * string(ind_id) * ".tsv"

    if !isfile(file_cov)
        @printf("Error: Sampler options for the rand-walk-sampler does not exist\n")
        return 1
    end

    # Read sampler components as floats 
    cov_mat = readdlm(file_cov, '\t', FLOAT, '\n')

    # Ensure correct type of all variables 
    dim = size(cov_mat)[1]

    cov_mat_old::Array{FLOAT, 2} = Array{FLOAT, 2}(I, (dim, dim))
    mean_vec::Array{FLOAT, 1} = Array{FLOAT, 1}(undef, dim)

    mcmc_sampler = McmcSamplerRandWalk(cov_mat, 
                                       step_length, 
                                       dim, 
                                       "Rand_walk_sampler", 
                                       cov_mat_old, 
                                       mean_vec)
    return mcmc_sampler
end


"""
    propose_parameters(x_prop, x_old, mcmc_sampler::McmcSamplerRandWalk, n_param, param_pos::Array{Bool, 1})

Propose using (adapative) mcmc-proposal, where postive parameters are proposed via exponential transformation. 

Using x-old, a new x is proposed via a symmetric normal-proposal with covariance matrix and step-length 
found in the Mcmc-sampler struct. The positive parameters, which are keep account on via param_pos, are 
proposed using an exponential transformation. 

See also: [`McmcSamplerRandWalk`](@ref)
"""
function propose_parameters(x_prop, 
                            x_old, 
                            mcmc_sampler::McmcSamplerRandWalk, 
                            n_param, 
                            param_pos::Array{Bool, 1})
    
    normal_rand = randn(n_param)
    cov_mat = mcmc_sampler.cov_mat
    step_length = mcmc_sampler.step_length

    # For transforming variables that are transformed via a log-transform 
    x_old_cp = deepcopy(x_old)
    x_old_cp[param_pos] .= log.(x_old_cp[param_pos])    
        
    x_prop .= x_old_cp + cholesky(step_length*cov_mat).U * normal_rand
    x_prop[param_pos] .= exp.(x_prop[param_pos])
end
"""
See also: [`McmcSamplerAM`](@ref)
"""
function propose_parameters(x_prop, 
                            x_old, 
                            mcmc_sampler::McmcSamplerAM, 
                            n_param, 
                            param_pos::Array{Bool, 1})
    
    normal_rand = randn(n_param)
    cov_mat = mcmc_sampler.cov_mat
    lambda = mcmc_sampler.lambda

    # For transforming variables that are transformed via a log-transform 
    x_old_cp = deepcopy(x_old)
    x_old_cp[param_pos] .= log.(x_old_cp[param_pos])    
        
    x_prop .= x_old_cp + cholesky(lambda*cov_mat).U * normal_rand
    x_prop[param_pos] .= exp.(x_prop[param_pos])
end
"""
See also: [`McmcSamplerGenAM`](@ref)
"""
function propose_parameters(x_prop, 
                            x_old, 
                            mcmc_sampler::McmcSamplerGenAM, 
                            n_param, 
                            param_pos::Array{Bool, 1})
    
    normal_rand = randn(n_param)
    cov_mat::Array{FLOAT, 2} = deepcopy(mcmc_sampler.cov_mat)
    lambda::FLOAT = exp(mcmc_sampler.log_lambda[1])

    # For transforming variables that are transformed via a log-transform 
    x_old_cp = deepcopy(x_old)

    # Ensure correct transformation 
    @inbounds @simd for k in 1:n_param
        if param_pos[k]
            x_old_cp[k] = log(x_old_cp[k])    
        end
    end
        
    x_prop .= x_old_cp + cholesky(lambda*cov_mat).U * normal_rand

    @inbounds @simd for k in 1:n_param
        if param_pos[k]
            x_prop[k] = exp(x_prop[k])
        end
    end

end
"""
See also: [`McmcSamplerRam`](@ref)
"""
function propose_parameters(x_prop, 
                            x_old, 
                            mcmc_sampler::McmcSamplerRam, 
                            n_param, 
                            param_pos::Array{Bool, 1})
    
    normal_rand = randn(n_param)
    cov_mat = mcmc_sampler.cov_mat

    # For transforming variables that are transformed via a log-transform 
    x_old_cp = deepcopy(x_old)
    x_old_cp[param_pos] .= log.(x_old_cp[param_pos])    
        
    x_prop .= x_old_cp + cholesky(cov_mat).U * normal_rand
    x_prop[param_pos] .= exp.(x_prop[param_pos])

    # Save random-numbers used for proposing 
    mcmc_sampler.q_vec .= normal_rand
end


"""
    update_sampler!(mcmc_sampler::McmcSamplerRandWalk, samples, iteration, log_alpha)

Update parameters (e.g Σ) for adaptive mcmc-proposal. 

For algorithms based on adpating after acceptance rate, log-alpha is required. 
For algorithms with a decreasing adapation rate the taken samplers and current 
iteration are required. For random-walk nothing is updated. 
"""
function update_sampler!(mcmc_sampler::McmcSamplerRandWalk, samples, iteration, log_alpha)
    
    new_sample = samples[:, iteration]

    # Update mean 
    delta_mean = (new_sample - mcmc_sampler.mean) * (new_sample - mcmc_sampler.mean)'
    mcmc_sampler.cov_mat_old .= (iteration-2) / (iteration-1) * mcmc_sampler.cov_mat_old + 1/iteration * delta_mean

    # Update covariance 
    mcmc_sampler.mean .= 1/iteration *((iteration - 1) * mcmc_sampler.mean .+ new_sample)
    
end
"""
For AM-algorithm μ and  Σ are updated at each 30:th iteration after 
the sampler has run for a specific amount of iterations.
    
See also: [`McmcSamplerAM`](@ref)
"""
function update_sampler!(mcmc_sampler::McmcSamplerAM, samples, iteration, log_alpha)
    
    alpha = min(exp(log_alpha), 1)

    old_sample = samples[:, iteration-1]
    new_sample = samples[:, iteration]
    
    # Don't update if not having reached target
    if mcmc_sampler.steps_before_update > iteration
        return 
    end

    # Only update each user provided iteration
    if !(iteration % mcmc_sampler.update_it == 0)
        return 
    end

    gamma = mcmc_sampler.gamma0 / (iteration^mcmc_sampler.alpha_power)

    # Mcmc-parameters 
    mcmc_sampler.mu .+= gamma * (new_sample .- mcmc_sampler.mu)
    mu = mcmc_sampler.mu
    mcmc_sampler.cov_mat .+= gamma *((new_sample - mu)*(new_sample - mu)' .- mcmc_sampler.cov_mat)
    # For stability, ensure symmetry 
    mcmc_sampler.cov_mat .=  Symmetric(mcmc_sampler.cov_mat) 

end
"""
For general AM-algorithm μ, log(λ) and  Σ are updated at each iteration after 
the sampler has run for a specific amount of iterations. 

See also: [`McmcSamplerGenAM`](@ref)
"""
# Function for updating the covariance matrix for the General AM-sampler  
function update_sampler!(mcmc_sampler::McmcSamplerGenAM, samples, iteration, log_alpha)
    
    alpha = min(exp(log_alpha), 1)
    alpha_target = mcmc_sampler.alpha_target

    old_sample = samples[:, iteration-1]
    new_sample = samples[:, iteration]
    
    # Don't update if not having reached target
    if mcmc_sampler.steps_before_update > iteration
        return 
    end

    # Only update each user provided iteration
    if !(iteration % mcmc_sampler.update_it == 0)
        return 
    end

    gamma = min(1, mcmc_sampler.gamma0 / (iteration^mcmc_sampler.alpha_power))

    # Mcmc-parameters 
    mcmc_sampler.log_lambda[1] += gamma * (alpha - alpha_target)
    mcmc_sampler.mu .+= gamma * (new_sample .- mcmc_sampler.mu)
    mu = mcmc_sampler.mu

    mcmc_sampler.cov_mat .+= gamma *((new_sample - mu)*(new_sample - mu)' .- mcmc_sampler.cov_mat)
    # For stability, ensure symmetry 
    mcmc_sampler.cov_mat .= Symmetric(mcmc_sampler.cov_mat)

end
"""
For the RAM-algorithm Σ is updated at each iteration after 
the sampler has run for a specific amount of iterations. 

See also: [`McmcSamplerGenAM`](@ref)
"""
# Function for updating the covariance matrix for the General AM-sampler  
function update_sampler!(mcmc_sampler::McmcSamplerRam, samples, iteration, log_alpha)

    # Don't update if not having reached target
    if mcmc_sampler.steps_before_update > iteration
        return 
    end

    # Only update each user provided iteration
    if !(iteration % mcmc_sampler.update_it == 0)
        return 
    end

    alpha = min(exp(log_alpha), 1)
    alpha_target = mcmc_sampler.alpha_target

    gamma = min(1, mcmc_sampler.dim / (iteration^mcmc_sampler.alpha_power))

    # Ensure correct decomposition of covariance 
    cholesky_cov = cholesky(mcmc_sampler.cov_mat)

    # Random-number-term in update
    q_vec = mcmc_sampler.q_vec
    norm_sq = dot(q_vec, q_vec)
    rand_term = (alpha - alpha_target) * (I + (q_vec * transpose(q_vec) / norm_sq))

    # Mcmc-parameters 
    mcmc_sampler.cov_mat .+= cholesky_cov.L * (gamma * rand_term) * cholesky_cov.U
    # For stability, ensure symmetry 
    mcmc_sampler.cov_mat .= (mcmc_sampler.cov_mat + mcmc_sampler.cov_mat') ./ 2.0
end

