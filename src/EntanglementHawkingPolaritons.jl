module EntanglementHawkingPolaritons

using Statistics, FFTW

include("windows.jl")
export WindowKind, Hann, WindowSpec, hann

include("observables.jl")
export ObservableTransform, PositionField, PositionDensity
export WindowedFourierField, WindowedFourierDensity
export CompiledObservableTransform, CompiledObservables
export compile_transform, compile_observables, evaluate!

include("ensemble_statistics.jl")
export CorrelationIndex, EnsembleStatistics, initialize_statistics
export batch_statistics!, merge_statistics!

include("ggpe.jl")
export dispersion, loss, potential, pump, nonlinearity, position_noise_func

end
