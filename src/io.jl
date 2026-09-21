function save_steady_state(saving_dir, steady_state, param, tspan)
    path = joinpath(saving_dir, "steady_state.jld2")

    jldopen(path, "a+") do file
        file["steady_state"] = steady_state
        file["param"] = param
        file["t_steady_state"] = tspan[end]
    end
end

function read_steady_state(saving_dir, ::Type{T}=Array) where {T}
    path = joinpath(saving_dir, "steady_state.jld2")

    jldopen(path) do file
        file["steady_state"] .|> T,
        file["param"],
        file["t_steady_state"]
    end
end

function save_window_pair(saving_dir, pair)
    path = joinpath(saving_dir, "windows.jld2")

    jldopen(path, "a+") do file
        n = length(keys(file)) + 1
        file["window_pair_$n"] = pair
    end
end


function read_window_pairs(saving_dir, ::Type{T}=Vector{Float64}) where {T}
    path = joinpath(saving_dir, "windows.jld2")
    isfile(path) || return []
    jldopen(path) do file
        [file[key] for key ∈ keys(file)]
    end
end

"""Convert all numeric work arrays in `statistics` to the requested backend."""
function _convert_statistics(statistics::EnsembleStatistics, ::Type{T}) where {T}
    mean_entries = [(key, T(value)) for (key, value) in statistics.mean]
    mean_value_type = isempty(mean_entries) ? AbstractVector : typeof(first(mean_entries)[2])
    mean = Dict{ObservableTransform,mean_value_type}(mean_entries)

    correlation_entries = [(key, T(value)) for (key, value) in statistics.correlation]
    correlation_value_type = isempty(correlation_entries) ? AbstractMatrix : typeof(first(correlation_entries)[2])
    correlation = Dict{CorrelationIndex,correlation_value_type}(correlation_entries)
    return EnsembleStatistics(mean, correlation, statistics.num_samples)
end

"""
    save_statistics(saving_dir, statistics)

Persist ensemble statistics as CPU arrays. The previous file is retained as
`previous_statistics.jld2`; a newer save replaces that backup.
"""
function save_statistics(saving_dir, statistics::EnsembleStatistics)
    mkpath(saving_dir)
    path = joinpath(saving_dir, "statistics.jld2")
    temporary_path = joinpath(saving_dir, "statistics.jld2.tmp")
    previous_path = joinpath(saving_dir, "previous_statistics.jld2")

    rm(temporary_path; force=true)
    jldopen(temporary_path, "w") do file
        file["statistics"] = _convert_statistics(statistics, Array)
    end

    rm(previous_path; force=true)
    isfile(path) && mv(path, previous_path)
    mv(temporary_path, path)
    return path
end

"""
    read_statistics(saving_dir, Array)

Load the complete saved statistics object, materializing its arrays on `T`.
"""
function read_statistics(saving_dir, ::Type{T}=Array) where {T}
    path = joinpath(saving_dir, "statistics.jld2")
    jldopen(path) do file
        _convert_statistics(file["statistics"], T)
    end
end
