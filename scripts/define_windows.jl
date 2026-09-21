using EntanglementHawkingPolaritons, JLD2

saving_dir = "data/test"

steady_state, param = jldopen(joinpath(saving_dir, "steady_state.jld2")) do file
    file["steady_state"], file["param"]
end
xs = StepRangeLen(0, param.dx, param.N) .- param.x_def
##
for param ∈ ((-250, 500), (-180, 200))
    xmin, window_length = param
    xmax = xmin + window_length
    window1 = WindowSpec(-xmax, -xmin, xs)
    window2 = WindowSpec(xmin, xmax, xs)
    save_window_pair(saving_dir, Pair(window1, window2))
end
##
plot_all_windows(saving_dir, savefig=true, xlims=(-300, 300))