function uncentered_normal_correlation(statistics, field_1, field_2)
    index = CorrelationIndex(field_1, field_2, false)
    centered = Array(statistics.correlation[index])
    mean_1 = Array(statistics.mean[field_1])
    mean_2 = Array(statistics.mean[field_2])
    return centered + mean_1 * mean_2'
end

function calculate_g2_minus_one(
    statistics,
    field_1,
    field_2,
    density_1,
    density_2,
    c11,
    c22,
    c12,
)
    statistics.num_samples > 0 || throw(ArgumentError("statistics contain no samples"))

    mean_density_1 = real.(Array(statistics.mean[density_1]))
    mean_density_2 = real.(Array(statistics.mean[density_2]))
    density_covariance = real.(Array(
        statistics.correlation[CorrelationIndex(density_1, density_2, false)],
    ))
    first_order = uncentered_normal_correlation(statistics, field_1, field_2)

    numerator = density_covariance .- real.(conj.(c12) .* first_order) .+ abs2.(c12) ./ 4
    density_1_normal = mean_density_1 .- c11 / 2
    density_2_normal = mean_density_2 .- c22 / 2
    return numerator ./ (density_1_normal * density_2_normal')
end

"""
    position_g2_minus_one(statistics, dx)

Calculate the normally ordered position-space `g² - 1` from persisted,
centered ensemble statistics using the truncated-Wigner correction.
"""
function position_g2_minus_one(statistics::EnsembleStatistics, dx)
    field = PositionField()
    density = PositionDensity()
    N = length(statistics.mean[field])
    c12 = Matrix{ComplexF64}(I, N, N) ./ dx
    return calculate_g2_minus_one(
        statistics,
        field,
        field,
        density,
        density,
        inv(dx),
        inv(dx),
        c12,
    )
end

function momentum_commutators(first_window::WindowSpec, second_window::WindowSpec, dx)
    weights_1 = get_window_weights(first_window)
    weights_2 = get_window_weights(second_window)
    c11 = sum(abs2, weights_1) / dx
    c22 = sum(abs2, weights_2) / dx
    c12 = zeros(ComplexF64, length(weights_1), length(weights_2))

    offset = second_window.first_idx - first_window.first_idx
    for second_idx in eachindex(weights_2)
        first_idx = second_idx + offset
        if first_idx in eachindex(weights_1)
            c12[first_idx, second_idx] =
                weights_1[first_idx] * conj(weights_2[second_idx]) / dx
        end
    end

    fft!(c12, 1)
    bfft!(c12, 2)
    return c11, c22, c12
end

"""
    momentum_g2_minus_one(statistics, window_pair, dx)

Calculate the normally ordered cross-window momentum-space `g² - 1`. The
first member of `window_pair` is the first matrix dimension and the horizontal
plot axis; by convention it is the downstream window.
"""
function momentum_g2_minus_one(statistics::EnsembleStatistics, window_pair::Pair, dx)
    first_field = WindowedFourierField(window_pair.first)
    second_field = WindowedFourierField(window_pair.second)
    first_density = WindowedFourierDensity(window_pair.first)
    second_density = WindowedFourierDensity(window_pair.second)
    commutators = momentum_commutators(window_pair.first, window_pair.second, dx)

    return calculate_g2_minus_one(
        statistics,
        first_field,
        second_field,
        first_density,
        second_density,
        commutators...,
    )
end
