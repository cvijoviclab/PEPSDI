# Function contianing the drift for the first sde-model
# Args:
#     du (output), updated drift-vector
#     u, current u-value
#     c, parameter vector for solving SDE
#     t, current time-step value
# Returns:
#     void
function alpha_mod1(du, u, p, t::FLOAT)
    c = p.c
    h1 = c[1]
    h2 = c[2] * u[1] * u[1]
    h3 = c[3] * u[2]
    h4 = c[4] * u[2]
    du[1] = h1 - 2*h2 + h3
    du[2] = h2 - h3 - h4
end
function alpha_mod1_kappa(du, u, p, t)
    c = p.c
    kappa = p.kappa
    h1 = c[1]
    h2 = c[2] * u[1] * u[1]
    h3 = c[3] * u[2]
    h4 = kappa[1] * u[2]
    du[1] = h1 - 2*h2 + h3
    du[2] = h2 - h3 - h4
end


# Function contianing the diffusion-matrix for the first sde-model
# Args:
#     du (output), updated diffusion-matrix
#     u, current u-value
#     c, parameter vector for solving SDE
#     t, current time-step value
# Returns:
#     void
function beta_mod1(du, u, p, t::FLOAT)
    c = p.c
    h1 = c[1]
    h2 = c[2] * u[1] * u[1]
    h3 = c[3] * u[2]
    h4 = c[4] * u[2]

    du[1, 1] = h1 + 4*h2 + 4*h3
    du[1, 2] = -(2*h2 + 2*h3)
    du[2, 1] = du[1, 2]
    du[2, 2] = h2 + h3 + h4
end
function beta_mod1_kappa(du, u, p, t)
    c = p.c
    kappa = p.kappa
    h1 = c[1]
    h2 = c[2] * u[1] * u[1]
    h3 = c[3] * u[2]
    h4 = kappa[1] * u[2]

    du[1, 1] = h1 + 4*h2 + 4*h3
    du[1, 2] = -(2*h2 + 2*h3)
    du[2, 1] = du[1, 2]
    du[2, 2] = h2 + h3 + h4

end


# Observation function for the first sde-model.
# Args:
#     y (output), updated observed values from the model
#     u, current u-value
#     c, parameter vector for solving SDE
#     t, current time-step value
# Returns:
#     void
function sde_mod1_obs(y, u, p, t)
    y[1] = u[2]
end


# Probability to observe a specific observation vector
# at a time-point t.
# Args:
#     y_obs, observation vector for the data
#     y_mod, the model produced y-values
#     error_dist, error-distribution list (for each observeble)
#     t, current time-point
#     dim_obs, dimension of observation model
# Returns:
#     prob, probability of observing y_obs at time t
function prob_obs_mod1(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob += logpdf(error_dist, diff)
    
    # Log-scale for numerical accuracy
    prob = exp(prob)
    return prob
end


# Function calculating the initial values for model 1 
# Args:
#     x0, (output) current initial values 
#     ind_param, individual parameters 
# Returns:
#     void 
function calc_x0_mod1!(x0, p)
    x0[1] = 110.0
    x0[2] = 70.0
end

function empty_func()
end