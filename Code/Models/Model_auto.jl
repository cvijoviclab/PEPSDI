function auto_obs(y, u, p, t)
    y[1] = log(u[1])
end


function auto_prob_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)
    
    return exp(prob)
end


function auto_h_vec!(u, h_vec, p, t)
    c = p.c
    h_vec[1] = c[1]
    h_vec[2] = c[2]
    h_vec[3] = c[3] * u[1]
    h_vec[4] = c[4] * u[2]
    h_vec[5] = c[5] * u[1] * u[2]
end


function auto_x0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 5
    x0[2] = 5
end
function auto_x0!(x0::T1, p) where T1<:Array{<:Int32, 1}
    x0[1] = 5
    x0[2] = 5
end
function auto_x0!(x0::T1, p) where T1<:Array{<:Int64, 1}
    x0[1] = 5
    x0[2] = 5
end

