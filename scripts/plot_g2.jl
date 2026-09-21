using EntanglementHawkingPolaritons

saving_dir = "data/test"
momentum_window_indices = nothing # Use nothing to plot every saved pair.
savefig = true

position_power = 5
position_colorrange = (-6, 6)
position_limits = (-150, 150)

momentum_power = 3
momentum_colorrange = (-2, 2)
momentum_xlims = nothing
momentum_ylims = nothing

begin
    fig = plot_position_g2_minus_one(
        saving_dir;
        power=position_power,
        colorrange=position_colorrange,
        xlims=position_limits,
        ylims=position_limits,
        savefig,
    )
    isinteractive() && display(fig)
end
##
window_indices = isnothing(momentum_window_indices) ?
                 eachindex(read_window_pairs(saving_dir)) : momentum_window_indices

for window_index in window_indices
    fig = plot_momentum_g2_minus_one(
        saving_dir,
        window_index;
        power=momentum_power,
        colorrange=momentum_colorrange,
        xlims=momentum_xlims,
        ylims=momentum_ylims,
        savefig,
    )
    isinteractive() && display(fig)
end
