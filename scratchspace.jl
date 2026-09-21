using EntanglementHawkingPolaritons, CairoMakie

window_spec = WindowSpec(1, 64)
weights = get_window_weights(window_spec, Vector{Float32})

lines(weights)