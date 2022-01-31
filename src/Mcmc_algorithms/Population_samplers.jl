"""
    init_pop_sampler_opt(sampler::PopOrnstein; acc_prob::FLOAT=0.65, n_warm_up::Int64=100)

Initialise η-parameterisation and options for NUTS-sampler. 

NUTS options are acceptance probability and number of warm-up samples. 

PopOrnstein-parameterisation is τ_i ~ Gamma(α_i, β_i), μ_i ~ Normal(μ_i0, M_i0*τ_i)
and y_i = MvNormal(μ, Σ) with Σ = Diagonal(1 / τ). Note τ = precision. 
"""
function init_pop_sampler_opt(sampler::PopOrnstein; acc_prob::FLOAT=0.65, n_warm_up::Int64=100)
    pop_sampler_opt = PopSamplerOrnsteinOpt(acc_prob, n_warm_up)
end
"""
PopNormalLKJ-parameterisation is τ_i ~ prior_τ_i, μ_i ~ prior_μ_i, corr ~ LKJ(eta) and 
y_i = MvNormal(μ, Σ) with Σ = Diagonal(τ) * corr * Diagonal(τ). Prior_i = user provided prior. 
"""
function init_pop_sampler_opt(sampler::PopNormalLKJ; acc_prob::FLOAT=0.65, n_warm_up::Int64=10)
    pop_sampler_opt = PopSamplerNormalLjkOpt(acc_prob, n_warm_up)
end
"""
PopNormalDiag-parameterisation is τ_i ~ prior_τ_i, μ_i ~ prior_μ_i, and 
y_i = MvNormal(μ, Σ) with Σ = Diagonal(τ.^2). Prior_i = user provided prior. 
"""
function init_pop_sampler_opt(sampler::PopNormalDiag; acc_prob::FLOAT=0.65, n_warm_up::Int64=10)
    pop_sampler_opt = PopSamplerNormalDiagOpt(acc_prob, n_warm_up)
end
function init_pop_sampler_opt(sampler::PopNormalLjkTwo, index_one::T1, index_two::T1, 
    tag_one::T2, tag_two::T2, n_param_ind::T3; 
    acc_prob::FLOAT=0.65, n_warm_up::Int64=10) where {T1<:Array{<:Integer, 1}, T2<:Array{<:Integer, 1}, T3<:Signed}

    pop_sampler_opt = PopSamplerNormalLjkOptTwo(acc_prob, n_warm_up, index_one, index_two, tag_one, tag_two, n_param_ind)
end


"""
    init_pop_sampler(sampler::PopSamplerOrnsteinOpt, n_individuals, n_param)

Initalise pop-sampler struct used within Gibbs-sampler from pop_sampler_opt. 

For efficiency number of individuals and parameters are stored in pop-sampler-options,
in addition to acceptance-probability and number of warm up samples. 

See also: [`init_pop_sampler_opt`](@ref)
"""
function init_pop_sampler(sampler::PopSamplerOrnsteinOpt, n_individuals, n_param)
    acc_prop = sampler.acc_prop
    n_warm_up = sampler.n_warm_up

    pop_sampler = PopSamplerOrnstein(acc_prop, n_warm_up, n_individuals, n_param)
end
function init_pop_sampler(sampler::PopSamplerNormalLjkOpt, n_individuals, n_param)
    acc_prop = sampler.acc_prop
    n_warm_up = sampler.n_warm_up

    pop_sampler = PopSamplerNormalLjk(acc_prop, n_warm_up, n_individuals, n_param)
end
function init_pop_sampler(sampler::PopSamplerNormalDiagOpt, n_individuals, n_param)
    acc_prop = sampler.acc_prop
    n_warm_up = sampler.n_warm_up

    pop_sampler = PopSamplerNormalDiag(acc_prop, n_warm_up, n_individuals, n_param)
end
function init_pop_sampler(sampler::PopSamplerNormalLjkOptTwo, n_individuals, n_param)
    acc_prop = sampler.acc_prop
    n_warm_up = sampler.n_warm_up
    index_one = sampler.index_one
    index_two = sampler.index_two
    tag_one = sampler.tag_one
    tag_two = sampler.tag_two
    n_param_ind = sampler.n_param_ind

    pop_sampler = PopSamplerNormalLjkTwo(acc_prop, n_warm_up, index_one, index_two, tag_one, tag_two, 
        n_individuals, n_param, n_param_ind)
end


"""
    turing_ornstein(prior_mean, prior_scale, n_param, ind_param, n_individuals)

Turing-library model-function for PopOrnstein-η-parameterisation. 

See sampler-parameterisation in documentation for [`init_pop_sampler_opt`](@ref)
"""
@model function turing_ornstein(prior_mean, prior_scale, n_param, ind_param, n_individuals)
    
    mean_val ~ arraydist(prior_mean)
    scale_val ~ arraydist(prior_scale)
    
    cov_mat = Diagonal(1 ./ scale_val)

    ind_param ~ filldist(MvNormal(mean_val, cov_mat), n_individuals)

    return mean_val, scale_val
end


"""
    turing_lkj_normal(prior_mean, prior_scale, prior_corr, 
        n_param, ind_param, n_individuals, ::Type{T} = Float64) where {T}

Turing-library sampler-function for PopNormalLKJ-η-parameterisation. 

Currently only works in multi-threaded environment. 

For sampler-parameterisation see documentation for [`init_pop_sampler_opt`](@ref)
"""
@model function turing_lkj_normal(prior_mean, prior_scale, prior_corr, 
    n_param, ind_param, n_individuals)
    
    # If using threads must have _varinfo.varinfo
    if :Omega in fieldnames(typeof(_varinfo.varinfo.metadata))
        if :vals in fieldnames(typeof(_varinfo.varinfo.metadata.Omega))
            if :value in fieldnames(typeof(_varinfo.varinfo.metadata.Omega.vals[1]))
                if det(Bijectors._inv_link_chol_lkj(reshape(map(x->x.value, _varinfo.varinfo.metadata.Omega.vals), n_param, n_param)))<=0
                    println("reject 1!")
                    Turing.acclogp!(_varinfo, -Inf)
                    return
                end
            end
        end
    end
    
    mean_val ~ arraydist(prior_mean)
    scale_val ~ arraydist(prior_scale)

    Omega ~ prior_corr    
    
    # Reject if covariance matrix is not semnite-positive definite 
    _Sigma = Symmetric(Diagonal(scale_val) * Omega * Diagonal(scale_val))
    if !isposdef(_Sigma)
        println("reject 2!")
        Turing.acclogp!(_varinfo, -Inf)
        return
    end
    cov_mat = PDMat(_Sigma)
    
    ind_param ~ filldist(MvNormal(mean_val, cov_mat), n_individuals)
    
    return mean, scale_val, Omega
end


@model function turing_lkj_normal_two(prior_mean, prior_scale, prior_corr, 
    n_param, ind_param, n_individuals, index_one::T1, index_two::T1, tag_one::T1, 
    tag_two::T1) where T1<:Array{<:Integer, 1}
    
    len_tag1 = length(tag_one)
    len_tag2 = length(tag_two)

    # If using threads must have _varinfo.varinfo
    if :Omega in fieldnames(typeof(_varinfo.varinfo.metadata))
        if :vals in fieldnames(typeof(_varinfo.varinfo.metadata.Omega))
            if :value in fieldnames(typeof(_varinfo.varinfo.metadata.Omega.vals[1]))
                if det(Bijectors._inv_link_chol_lkj(reshape(map(x->x.value, _varinfo.varinfo.metadata.Omega.vals), n_param, n_param)))<=0
                    println("reject 1!")
                    Turing.acclogp!(_varinfo, -Inf)
                    return
                end
            end
        end
    end
    
    mean_val ~ arraydist(prior_mean)
    scale_val ~ arraydist(prior_scale)

    Omega ~ prior_corr    
    
    # Reject if covariance matrix is not semnite-positive definite 
    _Sigma1 = Symmetric(Diagonal(scale_val[index_one]) * Omega * Diagonal(scale_val[index_one]))
    if !isposdef(_Sigma1)
        println("reject 2!")
        Turing.acclogp!(_varinfo, -Inf)
        return
    end
    _Sigma2 = Symmetric(Diagonal(scale_val[index_two]) * Omega * Diagonal(scale_val[index_two]))
    if !isposdef(_Sigma2)
        println("reject 2!")
        Turing.acclogp!(_varinfo, -Inf)
        return
    end

    cov_mat1 = PDMat(_Sigma1)
    cov_mat2 = PDMat(_Sigma2)
    
    @views ind_param[:, tag_one] ~ filldist(MvNormal(mean_val[index_one], cov_mat1), len_tag1)
    @views ind_param[:, tag_two] ~ filldist(MvNormal(mean_val[index_two], cov_mat2), len_tag2)
    
    return mean, scale_val, Omega
end


"""
    turing_diag_normal(prior_mean, prior_scale, n_param, ind_param, n_individuals)

Turing-library sampler-function for PopNormalLKJ-η-parameterisation. 

Currently only works in multi-threaded environment. 

For sampler-parameterisation see documentation for [`init_pop_sampler_opt`](@ref)
"""
@model function turing_diag_normal(prior_mean, prior_scale, n_param, ind_param, n_individuals)
        
    mean_val ~ arraydist(prior_mean)
    scale_val ~ arraydist(prior_scale)

    cov_mat = Diagonal(scale_val.^2)
    
    ind_param ~ filldist(MvNormal(mean_val, cov_mat), n_individuals)
    
    return mean, scale_val
end


"""
    gibbs_pop_param_warm_up(pop_par_current::PopParam, 
                            pop_sampler::PopSamplerOrnstein, 
                            pop_param_info::ParamInfoPop, 
                            ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}
    
Warm up NUTS-sampler using number of samples specified by pop_sampler and return Turing-chain object.

The NUTS-sampler is initalised with pop_par_current-values and sampling is 
conditoned on current individual parameters (ind_param_current)

For sampler-parameterisation see documentation for [`init_pop_sampler_opt`](@ref)
"""
function gibbs_pop_param_warm_up(pop_par_current::PopParam, 
                                 pop_sampler::PopSamplerOrnstein, 
                                 pop_param_info::ParamInfoPop, 
                                 ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Parameter required in the sampler 
    prior_scale = pop_param_info.prior_pop_param_scale
    prior_mean = pop_param_info.prior_pop_param_mean
    n_individuals = pop_sampler.n_individuals
    n_param = pop_sampler.n_param

    mod_pop_param = turing_ornstein(prior_mean, prior_scale, n_param, ind_param_current, n_individuals)

    # Set initial values for sampler 
    varinfo = Turing.VarInfo(mod_pop_param)
    mod_pop_param(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((
        mean_val = pop_par_current.mean_vec,
        scale_val= pop_par_current.scale_vec)))
    theta_init::Array{FLOAT, 1} = varinfo[Turing.SampleFromPrior()]
    
    # Sampler options 
    acc_prop::Float64 = convert(Float64, pop_sampler.acc_prop)
    n_warm_up::Int64 = convert(Int64, pop_sampler.n_warm_up)

    Turing.setadbackend(:forwarddiff)

    # If domain error is thrown by NUTS, don't update parameters
    local iter = 0
    while iter < 10
        try
            local sampled_chain
            @suppress begin
            sampled_chain = sample(mod_pop_param, NUTS(n_warm_up, acc_prop), 1, init_theta=theta_init, save_state=true)
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
function gibbs_pop_param_warm_up(pop_par_current::PopParam, pop_sampler::PopSamplerNormalLjk, 
    pop_param_info::ParamInfoPop, ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Parameter required in the sampler 
    prior_scale = pop_param_info.prior_pop_param_scale
    prior_mean = pop_param_info.prior_pop_param_mean
    prior_corr = pop_param_info.prior_pop_param_corr
    n_individuals = pop_sampler.n_individuals
    n_param = pop_sampler.n_param

    mod_pop_param = turing_lkj_normal(prior_mean, prior_scale, prior_corr, 
        n_param, ind_param_current, n_individuals)

    # Set initial values for sampler 
    varinfo = Turing.VarInfo(mod_pop_param)
    mod_pop_param(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((
        mean_val = pop_par_current.mean_vec,
        scale_val= pop_par_current.scale_vec, 
        Omega=pop_par_current.corr_mat)))
    theta_init = varinfo[Turing.SampleFromPrior()]
    
    # Sampler options 
    acc_prop = pop_sampler.acc_prop
    n_warm_up = pop_sampler.n_warm_up

    Turing.setadbackend(:forwarddiff)

    # If domain error is thrown by NUTS, don't update parameters
    local iter = 0
    while iter < 10
        try
            local sampled_chain
            @suppress begin
            sampled_chain = sample(mod_pop_param, NUTS(n_warm_up, acc_prop), 1, init_theta=theta_init, save_state=true)
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
function gibbs_pop_param_warm_up(pop_par_current::PopParam, pop_sampler::PopSamplerNormalDiag, 
    pop_param_info::ParamInfoPop, ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Parameter required in the sampler 
    prior_scale = pop_param_info.prior_pop_param_scale
    prior_mean = pop_param_info.prior_pop_param_mean
    n_individuals = pop_sampler.n_individuals
    n_param = pop_sampler.n_param

    mod_pop_param = turing_diag_normal(prior_mean, prior_scale, n_param, ind_param_current, n_individuals)

    # Set initial values for sampler 
    varinfo = Turing.VarInfo(mod_pop_param)
    mod_pop_param(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((
        mean_val = pop_par_current.mean_vec,
        scale_val= pop_par_current.scale_vec)))
    theta_init = varinfo[Turing.SampleFromPrior()]
    
    # Sampler options 
    acc_prop = pop_sampler.acc_prop
    n_warm_up = pop_sampler.n_warm_up

    Turing.setadbackend(:forwarddiff)

    # If domain error is thrown by NUTS, don't update parameters
    local iter = 0
    while iter < 10
        try
            local sampled_chain
            @suppress begin
            sampled_chain = sample(mod_pop_param, NUTS(n_warm_up, acc_prop), 1, init_theta=theta_init, save_state=true)
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
function gibbs_pop_param_warm_up(pop_par_current::PopParam, pop_sampler::PopSamplerNormalLjkTwo, 
    pop_param_info::ParamInfoPop, ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Parameter required in the sampler 
    prior_scale = pop_param_info.prior_pop_param_scale
    prior_mean = pop_param_info.prior_pop_param_mean
    prior_corr = pop_param_info.prior_pop_param_corr
    n_individuals = pop_sampler.n_individuals
    n_param = pop_sampler.n_param
    index_one = pop_sampler.index_one
    index_two = pop_sampler.index_two
    tag_one = pop_sampler.tag_one
    tag_two = pop_sampler.tag_two
    n_param_ind = pop_sampler.n_param_ind

    mod_pop_param = turing_lkj_normal_two(prior_mean, prior_scale, prior_corr, 
        n_param_ind, ind_param_current, n_individuals, index_one, index_two, tag_one, 
        tag_two)

    # Set initial values for sampler 
    varinfo = Turing.VarInfo(mod_pop_param)
    mod_pop_param(varinfo, Turing.SampleFromPrior(), Turing.PriorContext((
        mean_val = pop_par_current.mean_vec,
        scale_val= pop_par_current.scale_vec, 
        Omega=pop_par_current.corr_mat)))
    theta_init = varinfo[Turing.SampleFromPrior()]
    
    # Sampler options 
    acc_prop = pop_sampler.acc_prop
    n_warm_up = pop_sampler.n_warm_up

    Turing.setadbackend(:forwarddiff)
    
    # If domain error is thrown by NUTS, don't update parameters
    local iter = 0
    while iter < 10
        try
            local sampled_chain
            @suppress begin
            sampled_chain = sample(mod_pop_param, NUTS(n_warm_up, acc_prop), 1, init_theta=theta_init, save_state=true)
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

Update η-parameters based on ind_param_current and store in pop_par_current. Return Turing-chain from update. 

η-update is performed via Turing-resume function to avoid allocating new Chains-struct. 
Upon updating the chain generated by Turing-resume is returned.

For sampler-parameterisation see documentation for [`init_pop_sampler_opt`](@ref)
"""
function gibbs_pop_param!(pop_par_current::PopParam, 
                          pop_sampler::PopSamplerOrnstein, 
                          current_chain::Chains,
                          ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Change the individual parameters current value 
    n_param = pop_sampler.n_param
    current_chain.info.model.args.ind_param .= ind_param_current

    @suppress begin
    current_chain = resume(current_chain, 1, save_state=true)
    end

    name_scale_val = ["scale_val["*string(i)*"]" for i in 1:n_param]
    name_mean_val = ["mean_val["*string(i)*"]" for i in 1:n_param]

    @views pop_par_current.mean_vec .= current_chain[name_mean_val].value[1:end]
    @views pop_par_current.scale_vec .= current_chain[name_scale_val].value[1:end]

    return current_chain

end
function gibbs_pop_param!(pop_par_current::PopParam, 
                          pop_sampler::PopSamplerNormalLjk, 
                          current_chain::Chains,
                          ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Change the individual parameters current value 
    n_param = pop_sampler.n_param
    current_chain.info.model.args.ind_param .= ind_param_current

    @suppress begin
    current_chain = resume(current_chain, 1, save_state=true)
    end

    name_cor_val = ["Omega["*string(i)*","* string(j)*"]" for i in 1:n_param for j in 1:n_param]
    name_scale_val = ["scale_val["*string(i)*"]" for i in 1:n_param]
    name_mean_val = ["mean_val["*string(i)*"]" for i in 1:n_param]

    @views pop_par_current.corr_mat .= reshape(current_chain[name_cor_val].value, (n_param, n_param))
    @views pop_par_current.mean_vec .= current_chain[name_mean_val].value[1:end]
    @views pop_par_current.scale_vec .= current_chain[name_scale_val].value[1:end]

    return current_chain
end
function gibbs_pop_param!(pop_par_current::PopParam, 
                          pop_sampler::PopSamplerNormalLjkTwo, 
                          current_chain::Chains,
                          ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Change the individual parameters current value 
    n_param = pop_sampler.n_param
    n_param_ind = pop_sampler.n_param_ind
    current_chain.info.model.args.ind_param .= ind_param_current

    @suppress begin
    current_chain = resume(current_chain, 1, save_state=true)
    end

    name_cor_val = ["Omega["*string(i)*","* string(j)*"]" for i in 1:n_param_ind for j in 1:n_param_ind]
    name_scale_val = ["scale_val["*string(i)*"]" for i in 1:n_param]
    name_mean_val = ["mean_val["*string(i)*"]" for i in 1:n_param]

    @views pop_par_current.corr_mat .= reshape(current_chain[name_cor_val].value, (n_param_ind, n_param_ind))
    @views pop_par_current.mean_vec .= current_chain[name_mean_val].value[1:end]
    @views pop_par_current.scale_vec .= current_chain[name_scale_val].value[1:end]

    return current_chain
end
function gibbs_pop_param!(pop_par_current::PopParam, 
                          pop_sampler::PopSamplerNormalDiag, 
                          current_chain::Chains,
                          ind_param_current::T)::Chains where T<:Array{<:AbstractFloat, 2}

    # Change the individual parameters current value 
    n_param = pop_sampler.n_param
    current_chain.info.model.args.ind_param .= ind_param_current

    @suppress begin
    current_chain = resume(current_chain, 1, save_state=true)
    end

    name_scale_val = ["scale_val["*string(i)*"]" for i in 1:n_param]
    name_mean_val = ["mean_val["*string(i)*"]" for i in 1:n_param]

    @views pop_par_current.mean_vec .= current_chain[name_mean_val].value[1:end]
    @views pop_par_current.scale_vec .= current_chain[name_scale_val].value[1:end]

    return current_chain
end


"""
    calc_dist_ind_param(pop_par_current::PopParam, pop_sampler::PopSamplerNormalLjk)

Based on current population parameters calculate distribuion-struct for individual parameters. 
"""
function calc_dist_ind_param(pop_par_current::PopParam, pop_sampler::PopSamplerNormalLjk, ind_id::T1) where T1<:Signed

    scale_use = pop_par_current.scale_vec
    cov_mat = Symmetric(Diagonal(scale_use) * pop_par_current.corr_mat * Diagonal(scale_use))
    dist_ind_param = MvNormal(pop_par_current.mean_vec, cov_mat)
    
    return dist_ind_param
end
function calc_dist_ind_param(pop_par_current::PopParam, pop_sampler::PopSamplerOrnstein, ind_id::T1) where T1<:Signed

    cov_mat = Diagonal(1 ./ pop_par_current.scale_vec)
    dist_ind_param = MvNormal(pop_par_current.mean_vec, cov_mat)
    
    return dist_ind_param
end
function calc_dist_ind_param(pop_par_current::PopParam, pop_sampler::PopSamplerNormalDiag, ind_id::T1) where T1<:Signed

    cov_mat = Diagonal(pop_par_current.scale_vec.^2) 
    dist_ind_param = MvNormal(pop_par_current.mean_vec, cov_mat)
    
    return dist_ind_param
end
function calc_dist_ind_param(pop_par_current::PopParam, pop_sampler::PopSamplerNormalLjkTwo, ind_id::T1) where T1<:Signed

    if ind_id ∈ pop_sampler.tag_one
        scale_use = pop_par_current.scale_vec[pop_sampler.index_one]
        mean_vec = pop_par_current.mean_vec[pop_sampler.index_one]
        cov_mat = Symmetric(Diagonal(scale_use) * pop_par_current.corr_mat * Diagonal(scale_use))
        dist_ind_param = MvNormal(mean_vec, cov_mat)
    else
        scale_use = pop_par_current.scale_vec[pop_sampler.index_two]
        mean_vec = pop_par_current.mean_vec[pop_sampler.index_two]
        cov_mat = Symmetric(Diagonal(scale_use) * pop_par_current.corr_mat * Diagonal(scale_use))
        dist_ind_param = MvNormal(mean_vec, cov_mat)
    end
    
    return dist_ind_param
end