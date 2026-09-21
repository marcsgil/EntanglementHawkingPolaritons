using LinearAlgebra


struct CorrelationIndex{O1<:ObservableTransform,O2<:ObservableTransform}
    first::O1
    second::O2
    anomalous::Bool
end


mutable struct EnsembleStatistics{
    V<:AbstractVector,
    M<:AbstractMatrix,
}
    mean::Dict{ObservableTransform,V}
    correlation::Dict{CorrelationIndex,M}
    num_samples::Int64
end


"""Allocate a zero vector with the storage type and shape of `v`."""
function zero_vector_like(v::AbstractVector)
    out = similar(v)
    fill!(out, zero(eltype(out)))
    return out
end

"""Allocate a zero matrix suitable for the outer product `a * b'`."""
function zero_outer_like(a::AbstractVector, b::AbstractVector)
    T = promote_type(eltype(a), eltype(b))
    out = similar(a, T, (length(a), length(b)))
    fill!(out, zero(T))
    return out
end


"""
    initialize_statistics(field_prototype, observables, correlations)

Allocate empty statistics for the explicitly requested observable transforms
and correlations. Every correlation endpoint must be present in `observables`.
"""
function initialize_statistics(
    field_prototype::AbstractVector,
    observables,
    correlations,
)
    V = typeof(field_prototype)
    mean = Dict{ObservableTransform,V}()

    for observable in observables
        observable isa ObservableTransform ||
            throw(ArgumentError("all observables must be ObservableTransform values"))
        haskey(mean, observable) &&
            throw(ArgumentError("duplicate observable: $observable"))
        mean[observable] = zero_vector_like(observable_prototype(observable, field_prototype))
    end

    correlation_prototype = zero_outer_like(field_prototype, field_prototype)
    M = typeof(correlation_prototype)
    correlation = Dict{CorrelationIndex,M}()

    for index in correlations
        index isa CorrelationIndex ||
            throw(ArgumentError("all correlations must be CorrelationIndex values"))
        haskey(mean, index.first) ||
            throw(ArgumentError("correlation endpoint is not a requested observable: $(index.first)"))
        haskey(mean, index.second) ||
            throw(ArgumentError("correlation endpoint is not a requested observable: $(index.second)"))
        haskey(correlation, index) &&
            throw(ArgumentError("duplicate correlation: $index"))
        correlation[index] = zero_outer_like(mean[index.first], mean[index.second])
    end

    return EnsembleStatistics(mean, correlation, 0)
end


function update_correlation!(destination::AbstractArray{T}, values_1, values_2, anomalous) where {T}
    tB = anomalous ? 'T' : 'C'
    mul!(
        destination,
        'N',
        tB,
        values_1,
        values_2,
        T(inv(size(values_1, 2))),
        false,
    )
    return destination
end


"""
    batch_statistics!(batch, compiled, samples)

Overwrite `batch` with normalized population statistics for the columns of
`samples`. Each requested observable is evaluated once, then centered in its
compiled output workspace before correlations are calculated.
"""
function batch_statistics!(
    batch::EnsembleStatistics,
    compiled::CompiledObservables,
    samples::A,
) where {A<:AbstractMatrix}
    # `samples` may be a view of a saved solution while compiled transforms
    # own independent matrix workspaces, so their concrete matrix types need
    # not match.
    evaluated = Dict{ObservableTransform,AbstractMatrix}()
    for (observable, destination) in batch.mean
        values = evaluate!(compiled.transforms[observable], samples)
        evaluated[observable] = values
        mean!(destination, values)
    end

    for (observable, values) in evaluated
        values .-= batch.mean[observable]
    end

    for (index, destination) in batch.correlation
        update_correlation!(
            destination,
            evaluated[index.first],
            evaluated[index.second],
            index.anomalous,
        )
    end

    batch.num_samples = compiled.batchsize
    return batch
end


# ---------------------------------------------------------------------------
# Merge operations
# ---------------------------------------------------------------------------

"""Overwrite a batch mean with `μ_batch - μ_accumulated`."""
function write_δ!(μ_batch, μ_accumulated)
    μ_batch .-= μ_accumulated
    return μ_batch
end

"""Merge a batch mean using a precomputed mean difference `δμ`."""
function accumulate_mean!(μ_accumulated::AbstractArray{T}, δμ, num_samples, batchsize) where {T}
    N = num_samples + batchsize
    α = T(batchsize / N)
    @. μ_accumulated += α * δμ
    return μ_accumulated
end

"""
Merge normalized population correlations. Normal correlations use
`δμ1 * δμ2'`; anomalous correlations use `δμ1 * transpose(δμ2)`.
"""
function accumulate_correlation!(
    C_accumulated::AbstractArray{T},
    C_batch,
    δμ1,
    δμ2,
    num_samples,
    batchsize,
    anomalous,
) where {T}
    N = num_samples + batchsize
    α_old = T(num_samples / N)
    α_batch = T(batchsize / N)
    α_cross = T(α_old * α_batch)

    @. C_accumulated = α_old * C_accumulated + α_batch * C_batch

    tB = anomalous ? 'T' : 'C'
    mul!(
        C_accumulated,
        'N',
        tB,
        δμ1,
        δμ2,
        α_cross,
        one(α_cross),
    )

    return C_accumulated
end

"""
    merge_statistics!(accumulated, batch)

Destructively merge `batch` into `accumulated`. `batch.mean` is consumed as
workspace for the mean differences.
"""
function merge_statistics!(
    accumulated::EnsembleStatistics,
    batch::EnsembleStatistics,
)
    n = accumulated.num_samples
    m = batch.num_samples

    for key in keys(accumulated.mean)
        write_δ!(batch.mean[key], accumulated.mean[key])
    end

    for (index, C_accumulated) in accumulated.correlation
        accumulate_correlation!(
            C_accumulated,
            batch.correlation[index],
            batch.mean[index.first],
            batch.mean[index.second],
            n,
            m,
            index.anomalous,
        )
    end

    for key in keys(accumulated.mean)
        accumulate_mean!(accumulated.mean[key], batch.mean[key], n, m)
    end

    accumulated.num_samples += m
    return accumulated
end
