"""Requested observables and correlations for the noise-stage workflow."""
function correlation_requests(window_pairs)
    position_field = PositionField()
    position_density = PositionDensity()
    observables = ObservableTransform[position_field, position_density]
    correlations = CorrelationIndex[
        CorrelationIndex(position_field, position_field, false),
        CorrelationIndex(position_field, position_field, true),
        CorrelationIndex(position_density, position_density, false),
    ]

    for pair in window_pairs
        first_window, second_window = pair.first, pair.second
        first_field = WindowedFourierField(first_window)
        second_field = WindowedFourierField(second_window)
        first_density = WindowedFourierDensity(first_window)
        second_density = WindowedFourierDensity(second_window)
        append!(observables, (first_field, second_field, first_density, second_density))

        # Autocorrelations for each member, followed by the pair cross-correlation.
        append!(correlations, (
            CorrelationIndex(first_field, first_field, false),
            CorrelationIndex(first_field, first_field, true),
            CorrelationIndex(second_field, second_field, false),
            CorrelationIndex(second_field, second_field, true),
            CorrelationIndex(first_density, first_density, false),
            CorrelationIndex(second_density, second_density, false),
            CorrelationIndex(first_field, second_field, false),
            CorrelationIndex(first_field, second_field, true),
            CorrelationIndex(first_density, second_density, false),
        ))
    end
    return unique(observables), unique(correlations)
end

function _randn!(rng, values)
    isnothing(rng) ? randn!(values) : randn!(rng, values)
    return values
end

function check_statistics_requests(statistics, observables, correlations)
    saved_observables = Set(keys(statistics.mean))
    requested_observables = Set(observables)
    saved_correlations = Set(keys(statistics.correlation))
    requested_correlations = Set(correlations)

    saved_observables == requested_observables &&
        saved_correlations == requested_correlations && return nothing

    throw(ArgumentError(
        "statistics.jld2 does not match the currently requested observables " *
            "and correlations; the missing statistics cannot be reconstructed " *
            "from previous samples",
    ))
end

"""
    update_correlations!(saving_dir, batchsize, nbatches; kwargs...)

Run stochastic batches from independently sampled truncated-Wigner vacuum
states and accumulate centered field and density correlations. Position-space
field normal and anomalous autocorrelations are always included, as is the
density autocorrelation. For every saved window pair, both members' field
normal/anomalous and density autocorrelations, plus their corresponding
cross-correlations, are included.
"""
function update_correlations!(
    saving_dir,
    batchsize::Integer,
    nbatches::Integer;
    backend=CUDA.CuArray,
    nsaves::Integer=1,
    dt=nothing,
    show_progress::Bool=true,
    max_datetime=typemax(DateTime),
    rng=nothing,
    noise_eltype=nothing,
    log_path="log.txt",
)
    batchsize >= 1 || throw(ArgumentError("batchsize must be positive"))
    nbatches >= 0 || throw(ArgumentError("nbatches must be nonnegative"))
    nsaves >= 1 || throw(ArgumentError("nsaves must be positive"))

    steady_state, param, t_steady_state = read_steady_state(saving_dir, backend)
    field = only(steady_state)
    windows = read_window_pairs(saving_dir)
    observables, correlations = correlation_requests(windows)
    statistics_path = joinpath(saving_dir, "statistics.jld2")
    statistics = if isfile(statistics_path)
        saved = read_statistics(saving_dir, backend)
        check_statistics_requests(saved, observables, correlations)
        saved
    else
        initialize_statistics(field, observables, correlations)
    end
    compiled = compile_observables(field, batchsize, keys(statistics.mean))
    batch = initialize_statistics(field, keys(statistics.mean), keys(statistics.correlation))

    tspan = (zero(t_steady_state), t_steady_state)
    dt = isnothing(dt) ? param.dt : dt
    initial_batch = similar(field, (length(field), batchsize))
    noise_eltype = isnothing(noise_eltype) ? eltype(initial_batch) : noise_eltype
    noise_prototype = similar(initial_batch, noise_eltype)
    problem = GrossPitaevskiiProblem(
        (initial_batch,),
        (param.L,);
        dispersion,
        potential,
        nonlinearity,
        pump,
        position_noise_func,
        noise_prototype=(noise_prototype,),
        param,
    )
    solver = StrangSplitting()
    steps_per_save = GeneralizedGrossPitaevskii.resolve_fixed_timestepping(dt, tspan, nsaves)[3]
    progress = show_progress && nbatches > 0 ? Progress(steps_per_save * nsaves * nbatches) : nothing

    completed = 0
    open(log_path, "w+") do io
        logger = SimpleLogger(io)
        while completed < nbatches && (isnothing(max_datetime) || now() <= max_datetime)
            with_logger(logger) do
                @info "Batch $(completed + 1)"
            end
            flush(io)

            # Each column is an independent truncated-Wigner vacuum realization.
            _randn!(rng, initial_batch)
            initial_batch ./= sqrt(2 * param.dx)
            _, solution = solve(
                problem,
                solver,
                tspan;
                dt,
                nsaves,
                save_start=false,
                show_progress,
                progress,
                rng,
            )
            batch_statistics!(batch, compiled, @view solution[1][:, :, end])
            merge_statistics!(statistics, batch)
            completed += 1
        end
    end
    !isnothing(progress) && finish!(progress)

    save_statistics(saving_dir, statistics)
    return statistics
end
