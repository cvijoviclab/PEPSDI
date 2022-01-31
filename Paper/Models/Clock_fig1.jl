function clock_fig1_h_vec!(u, h_vec, p, t)
    c = p.c
    const_term = 0.2617993877991494
    h_vec[1] = c[1] * (2 + sin(const_term * t))
    h_vec[2] = c[2] * u[1]
    h_vec[3] = c[3] * u[1]
    h_vec[4] = c[4] * u[2]    
end
function clock_fig1_h_vec_max!(u, h_vec, p, t_start, t_end)
    c = p.c
    
    h_vec[1] = c[1] * 2
    h_vec[2] = c[2] * u[1]
    h_vec[3] = c[3] * u[1]
    h_vec[4] = c[4] * u[2]    
end

function clock_fig1_x0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 10
    x0[2] = 10
end

function clock_fig1_obs(y, u, p, t)
    y[1] = u[2]
end
function clock_fig1_prob_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)
    
    return exp(prob)
end
