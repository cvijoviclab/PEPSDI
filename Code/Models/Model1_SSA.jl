function mod1_obs(y, u, p, t)
    y[1] = u[2]
end


function mode1_prop_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])
    for i in 1:dim_obs
        if !isnan(y_obs[i])
            diff = y_obs[i] - y_mod[i]
            prob += logpdf(error_dist, diff)
        else
            continue
        end 
    end
    # Log-scale for numerical accuracy
    prob = exp(prob)
    return prob
end


function mod1_h_vec!(u, h_vec, p, t)
    c = p.c
    h_vec[1] = c[1]
    h_vec[2] = c[2] * u[1] * (u[1] - 1)
    h_vec[3] = c[3] * u[2]
    h_vec[4] = c[4] * u[2]
end


function mod1_x0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 110
    x0[2] = 70
end