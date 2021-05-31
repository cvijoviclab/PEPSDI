#= 

Model-equations for Ornstein-Uhlenbeck model in mixed effects setting 

=#


function alpha_ornstein_mixed(du, u, p, t)
    c = p.c
    kappa = p.kappa
    du[1] = c[1] * (c[2] - u[1])
end
function beta_ornstein_mixed(du, u, p, t)
    c = p.c
    kappa = p.kappa
    du[1, 1] = kappa[1]
end
function prob_ornstein_mixed(y_obs, y_mod, error_param, t, dim_obs)
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    # Log-scale for numerical accuracy
    prob = exp(prob)
    return prob
end

function ornstein_obs_mixed(y, u, p, t)
    y[1] = u[1]
end
function ornstein_calc_x0_mixed!(x0, p)
    x0[1] = 0.0
end

function empty_func()
end