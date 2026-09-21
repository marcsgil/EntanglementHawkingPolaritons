function dispersion(ks, param)
    param.ħ * sum(abs2, ks) / 2param.m - param.δ₀
end

gaussian(x, center, width) = exp(-((x - center) / width)^2)

function loss(xs, param)
    param.γ / 2 + param.V_damp * (gaussian(xs[1], zero(xs[1]), param.w_damp) +
                                  gaussian(xs[1], param.L, param.w_damp))
end

function potential(xs, param)
    param.V_def * gaussian(xs[1], param.x_def, param.w_def) - im * loss(xs, param)
end

function half_pump(x, Fmax, Fmin, k, w, x0)
    ((Fmax - Fmin) * sech((x - x0) / w) + Fmin) * cis(k * x)
end

time_dependence(t, param) = (param.extra_intensity * exp(-t / param.decay_time) + 1)

function pump(x, param, t)
    if x[1] < param.divide
        k = param.k_up
        F = param.F_up
        x0 = 0
    else
        k = param.k_down
        F = param.F_down
        x0 = param.L
    end

    ((param.F_max - F) * sech((x[1] - x0) / param.w_pump) + F) * cis(k * x[1]) * time_dependence(t, param)
end

nonlinearity(ψ, param) = param.g * (abs2(first(ψ)) - 1 / param.dx)

position_noise_func(ψ, xs, param) = √(loss(xs, param) / param.dx)