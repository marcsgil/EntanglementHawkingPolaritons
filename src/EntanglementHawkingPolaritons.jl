module EntanglementHawkingPolaritons

using Statistics, LinearAlgebra, FFTW, Roots, ForwardDiff, GeneralizedGrossPitaevskii, JLD2, CairoMakie, DSP
using CUDA, Dates, Logging, ProgressMeter, Random

include("windows.jl")
export WindowKind, Hann, WindowSpec, hann, get_window_weights

include("observables.jl")
export ObservableTransform, PositionField, PositionDensity
export WindowedFourierField, WindowedFourierDensity
export CompiledObservableTransform, CompiledObservables
export compile_transform, compile_observables, evaluate!

include("ensemble_statistics.jl")
export CorrelationIndex, EnsembleStatistics, initialize_statistics
export batch_statistics!, merge_statistics!

include("io.jl")
export save_steady_state, read_steady_state, save_window_pair, read_window_pairs
export save_statistics, read_statistics

include("ggpe.jl")
export dispersion, loss, potential, pump, nonlinearity, position_noise_func

include("correlations.jl")
export update_correlations!

include("g2.jl")
export position_g2_minus_one, momentum_g2_minus_one

include("tracing.jl")

include("polariton_funcs.jl")

include("plot_funcs.jl")
export plot_velocities, plot_density, plot_bistability, plot_dispersion, plot_window_pair, plot_all_windows
export plot_position_g2_minus_one, plot_momentum_g2_minus_one

end
