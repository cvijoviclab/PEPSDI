function calc_h_vec_osc(u, h_vec, p, t)

    c = p.c
    alpha_0 = c[1]
    alpha = c[2]
    beta = c[3]
    gamma = c[4]

    h_vec[1] = alpha_0 + alpha / (1 + u[6]^2)
    h_vec[2] = alpha_0 + alpha / (1 + u[4]^2)
    h_vec[3] = alpha_0 + alpha / (1 + u[5]^2)
    h_vec[4] = beta * u[1]
    h_vec[5] = beta * u[2]
    h_vec[6] = beta * u[3]
    h_vec[7] = gamma * u[1]
    h_vec[8] = gamma * u[2]
    h_vec[9] = gamma * u[3]
    h_vec[10] = beta * u[4]
    h_vec[11] = beta * u[5]
    h_vec[12] = beta * u[6]

end

function osc_x0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 0
    x0[2] = 0
    x0[3] = 0
    x0[4] = 2
    x0[5] = 1
    x0[6] = 3
end
function osc_x0!(x0, p)
    x0[1] = 0.0
    x0[2] = 0.0
    x0[3] = 0.0
    x0[4] = 2.0
    x0[5] = 1.0
    x0[6] = 3.0
end



function osc_obs(y, u, p, t)
    y[1] = u[4]
    y[2] = u[5]
    y[3] = u[6]
end


function osc_prob_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0

    error_dist1 = Normal(0.0, error_param[1])

    diff1 = y_obs[1] - y_mod[1]
    diff2 = y_obs[2] - y_mod[2]
    diff3 = y_obs[3] - y_mod[3]

    prob = logpdf(error_dist1, diff1) + logpdf(error_dist1, diff2) + logpdf(error_dist1, diff3)

    return exp(prob)
end


function alpha_osc(du, u, p, t)

    c = p.c
    alpha_0 = c[1]
    alpha = c[2]
    beta = c[3]
    gamma = c[4]

    h1 = alpha_0 + alpha / (1 + u[6]^2)
    h2 = alpha_0 + alpha / (1 + u[4]^2)
    h3 = alpha_0 + alpha / (1 + u[5]^2)
    h4 = beta * u[1]
    h5 = beta * u[2]
    h6 = beta * u[3]
    h7 = gamma * u[1]
    h8 = gamma * u[2]
    h9 = gamma * u[3]
    h10 = beta * u[4]
    h11 = beta * u[5]
    h12 = beta * u[6]

    du[1] = -h1 + h7
    du[2] = -h2 + h8 
    du[3] = -h3 + h9
    du[4] = h10 - h4
    du[5] = h11 - h5 
    du[6] = h12 - h6
    du .*= -1
end


function beta_osc(du, u, p, t)

    c = p.c
    alpha_0 = c[1]
    alpha = c[2]
    beta = c[3]
    gamma = c[4]

    h1 = alpha_0 + alpha / (1 + u[6]^2)
    h2 = alpha_0 + alpha / (1 + u[4]^2)
    h3 = alpha_0 + alpha / (1 + u[5]^2)
    h4 = beta * u[1]
    h5 = beta * u[2]
    h6 = beta * u[3]
    h7 = gamma * u[1]
    h8 = gamma * u[2]
    h9 = gamma * u[3]
    h10 = beta * u[4]
    h11 = beta * u[5]
    h12 = beta * u[6]

    #du .= 0.0
    du[1, 1] = h1 + h7
    du[2, 2] = h2 + h8
    du[3, 3] = h3 + h9
    du[4, 4] = h10 + h4
    du[5, 5] = h11 + h5
    du[6, 6] = h12 + h6

end


function calc_s_mat_osc()
    S_left = [0 0 0 0 0 0;
            0 0 0 0 0 0;
            0 0 0 0 0 0;
            1 0 0 0 0 0;
            0 1 0 0 0 0;
            0 0 1 0 0 0;
            1 0 0 0 0 0;
            0 1 0 0 0 0;
            0 0 1 0 0 0;
            0 0 0 1 0 0;
            0 0 0 0 1 0;
            0 0 0 0 0 1]
    S_right = [1 0 0 0 0 0;
            0 1 0 0 0 0;
            0 0 1 0 0 0;
            1 0 0 1 0 0;
            0 1 0 0 1 0;
            0 0 1 0 0 1;
            0 0 0 0 0 0;
            0 0 0 0 0 0;
            0 0 0 0 0 0;
            0 0 0 0 0 0;
            0 0 0 0 0 0;
            0 0 0 0 0 0]

    return convert(Array{Int16, 2}, S_left - S_right)
end
