#= 

Model-equations for Ornstein-Uhlenbeck model 

=#


function alpha_ornstein_full(du, u, p, t)
    c = p.c
    du[1] = c[1] * (c[2] - u[1])
end
function beta_ornstein_full(du, u, p, t)
    c = p.c
    du[1, 1] = c[3]^2
end
function prob_ornstein_full(y_obs, y_mod, error_param, t, dim_obs)

    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    # Log-scale for numerical accuracy
    prob = exp(prob)

    return prob
end


function alpha_ornstein_one(du, u, p, t)
    c  = p.c
    du[1] = exp(-0.7) * (c[1] - u[1])
end
function beta_ornstein_one(du, u, p, t)
    du[1, 1] = exp(-0.9)
end
function prob_ornstein_one(y_obs, y_mod, error_param, t, dim_obs)
    error_dist = Normal(0.0, 0.3)
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    # Log-scale for numerical accuracy
    prob = exp(prob)
    return prob
end



function ornstein_obs(y, u, p, t)
    y[1] = u[1]
end
function calc_x0_ornstein!(x0, ind_param)
    x0[1] = 0.0
end

function empty_func()
end