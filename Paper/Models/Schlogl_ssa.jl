function schlogl_obs_sde(y, u, p, t)
    y[1] = u[1]
end
function schlogl_obs_ssa(y, u, p, t)
    y[1] = u[1]
end


function schlogl_prop_obs(y_obs, y_mod, error_param, t, dim_obs)
    prob::FLOAT = 0.0
    noise = error_param[1]
    error_dist = Normal(0.0, error_param[1])
    diff = y_obs[1] - y_mod[1]
    prob = logpdf(error_dist, diff)

    return exp(prob)
end


function schlogl_h_vec!(u, h_vec, p, t)
    c = p.c
    kappa = p.kappa
    kappa3 = 23.1
    h_vec[1] = kappa[1] * u[1] * (u[1] - 1)
    h_vec[2] = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec[3] = c[1]
    h_vec[4] = kappa3 * u[1]
end

function schlogl_h_vec_full!(u, h_vec, p, t)
    c = p.c
    c4 = 37.5
    h_vec[1] = c[1] * u[1] * (u[1] - 1)
    h_vec[2] = c[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec[3] = c[3]
    h_vec[4] = c4 * u[1]
end


function schlogl_x0!(x0::T1, p) where T1<:Array{<:UInt16, 1}
    x0[1] = 0
end
function schlogl_x0!(x0::T1, p) where T1<:Array{<:AbstractFloat, 1}
    x0[1] = 0.0
end


# When using a SDE-variant 
function schlogl_alpha(du, u, p, t)
    c = p.c
    c4 = 37.5
    h_vec1 = c[1] * u[1] * (u[1] - 1)
    h_vec2 = c[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[3]
    h_vec4 = c4 * u[1]

    @views du[1] = h_vec1 - h_vec2 + h_vec3 - h_vec4
end
# When using a SDE-variant 
function schlogl_alpha_kappa(du, u, p, t)
    c = p.c
    kappa = p.kappa
    kappa3 = 23.1

    h_vec1 = kappa[1] * u[1] * (u[1] - 1)
    h_vec2 = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[1]
    h_vec4 = kappa3 * u[1]

    @views du[1] = h_vec1 - h_vec2 + h_vec3 - h_vec4
    
end


function schlogl_beta(du, u, p, t)
    c = p.c
    c4 = 37.5
    h_vec1 = c[1] * u[1] * (u[1] - 1)
    h_vec2 = c[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[3]
    h_vec4 = c4 * u[1]
    
    @views du[1, 1] = h_vec1 + h_vec2 + h_vec3 + h_vec4
end
function schlogl_beta_kappa(du, u, p, t)
    c = p.c
    kappa = p.kappa
    kappa3 = 23.1
    
    h_vec1 = kappa[1] * u[1] * (u[1] - 1)
    h_vec2 = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[1]
    h_vec4 = kappa3 * u[1]

    @views du[1, 1] = h_vec1 + h_vec2 + h_vec3 + h_vec4
end



function schlogl_alpha_kappa_test(du, u, p, t)
    c = p.c
    kappa = p.kappa

    h_vec1 = kappa[1] * u[1] * (u[1] - 1)
    h_vec2 = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[1]
    h_vec4 = kappa[3] * u[1]

    @views du[1] = h_vec1 - h_vec2 + h_vec3 - h_vec4
    
end
function schlogl_beta_kappa_test(du, u, p, t)
    c = p.c
    kappa = p.kappa
    kappa3 = 23.1
    
    h_vec1 = kappa[1] * u[1] * (u[1] - 1)
    h_vec2 = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec3 = c[1]
    h_vec4 = kappa[3] * u[1]

    @views du[1, 1] = h_vec1 + h_vec2 + h_vec3 + h_vec4
end

function schlogl_h_vec_test!(u, h_vec, p, t)
    c = p.c
    kappa = p.kappa
    h_vec[1] = kappa[1] * u[1] * (u[1] - 1)
    h_vec[2] = kappa[2] * u[1] * (u[1] - 1) * (u[1] - 2)
    h_vec[3] = c[1]
    h_vec[4] = kappa[3] * u[1]
end
