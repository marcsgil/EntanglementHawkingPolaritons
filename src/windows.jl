abstract type WindowKind end

"""Hann window kind for use in a [`WindowSpec`](@ref)."""
struct Hann <: WindowKind end

"""
    WindowSpec(first_idx, length[, kind=Hann()])

Immutable description of a spatial window. Numeric weights are materialized
for a particular array backend when an observable transform is compiled.
"""
struct WindowSpec{K<:WindowKind}
    first_idx::Int
    length::Int
    kind::K

    function WindowSpec(first_idx::Integer, length::Integer, kind::K=Hann()) where {K<:WindowKind}
        first_idx >= 1 || throw(ArgumentError("first_idx must be positive"))
        length >= 1 || throw(ArgumentError("window length must be positive"))
        return new{K}(Int(first_idx), Int(length), kind)
    end
end

function WindowSpec(x_begin, x_end, positions, kind::WindowKind=Hann())
    first_idx = position2idx(x_begin, positions)
    last_idx = position2idx(x_end, positions)
    return WindowSpec(first_idx, last_idx - first_idx, kind)
end

Base.length(spec::WindowSpec) = spec.length

selected_idxs(spec::WindowSpec) =
    spec.first_idx:(spec.first_idx+spec.length-1)

position2idx(position, positions) =
    argmin(index -> abs(positions[index] - position), eachindex(positions))

function check_window_bounds(spec::WindowSpec, source_length)
    last(selected_idxs(spec)) <= source_length ||
        throw(DimensionMismatch("window extends beyond a field of length $source_length"))
    return nothing
end

hann(n, N) = sinpi((n-1) / N)^2


"""Materialize a window specification using the storage backend of `prototype`."""
get_window_weights(spec::WindowSpec, type=Vector{Float64}) =
    get_window_weights(spec.kind, length(spec), type)

get_window_weights(spec::WindowSpec, prototype::AbstractArray) =
    get_window_weights(spec.kind, length(spec), prototype)

function get_window_weights(::Hann, N, ::Type{T}=Vector{Float64}) where {T<:AbstractArray}
    weights = T(undef, N)
    map!(n->hann(n, N), weights, eachindex(weights))
    return weights
end

function get_window_weights(::Hann, N, prototype::AbstractArray)
    T = typeof(real(zero(eltype(prototype))))
    weights = similar(prototype, T, N)
    map!(n -> hann(n, N), weights, eachindex(weights))
    return weights
end

"""Copy and window the first dimension of `src` into `dest`."""
function apply_window!(dest, src, spec::WindowSpec, weights)
    trailing = ntuple(_ -> Colon(), ndims(src) - 1)
    weight_shape = (length(spec), ntuple(_ -> 1, ndims(src) - 1)...)
    source = view(src, selected_idxs(spec), trailing...)
    dest .= source .* reshape(weights, weight_shape)
    return dest
end
