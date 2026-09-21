abstract type ObservableTransform end

"""The complex field in position space."""
struct PositionField <: ObservableTransform end

"""The position-space density `abs2(field)`, stored in a complex array."""
struct PositionDensity <: ObservableTransform end

"""The complex Fourier transform of a windowed field."""
struct WindowedFourierField{W<:WindowSpec} <: ObservableTransform
    window::W
end

"""The density `abs2(fft(window .* field))`, stored in a complex array."""
struct WindowedFourierDensity{W<:WindowSpec} <: ObservableTransform
    window::W
end


observable_length(::Union{PositionField,PositionDensity}, field_prototype) =
    length(field_prototype)

observable_length(transform::Union{WindowedFourierField,WindowedFourierDensity}, field_prototype) =
    length(transform.window)

function observable_prototype(transform::ObservableTransform, field_prototype::AbstractVector)
    if transform isa Union{WindowedFourierField,WindowedFourierDensity}
        check_window_bounds(transform.window, length(field_prototype))
    end
    return similar(field_prototype, eltype(field_prototype), observable_length(transform, field_prototype))
end


abstract type CompiledObservableTransform end

struct CompiledPositionField{A<:AbstractMatrix} <: CompiledObservableTransform
    output::A
end

struct CompiledPositionDensity{A<:AbstractMatrix} <: CompiledObservableTransform
    output::A
end

struct CompiledWindowedFourierField{
    W<:WindowSpec,
    V<:AbstractVector,
    A<:AbstractMatrix,
    P,
} <: CompiledObservableTransform
    window::W
    weights::V
    output::A
    plan::P
end

struct CompiledWindowedFourierDensity{
    W<:WindowSpec,
    V<:AbstractVector,
    A<:AbstractMatrix,
    P,
} <: CompiledObservableTransform
    window::W
    weights::V
    output::A
    plan::P
end

struct CompiledObservables
    transforms::Dict{ObservableTransform,CompiledObservableTransform}
    batchsize::Int
end


function batch_workspace(field_prototype::AbstractVector, output_length, batchsize)
    return similar(
        field_prototype,
        eltype(field_prototype),
        (output_length, batchsize),
    )
end

compile_transform(::PositionField, field_prototype::AbstractVector, batchsize) =
    CompiledPositionField(batch_workspace(field_prototype, length(field_prototype), batchsize))

function compile_transform(::PositionDensity, field_prototype::AbstractVector, batchsize)
    output = batch_workspace(field_prototype, length(field_prototype), batchsize)
    return CompiledPositionDensity(output)
end

function compile_transform(
    transform::WindowedFourierField,
    field_prototype::AbstractVector,
    batchsize,
)
    check_window_bounds(transform.window, length(field_prototype))
    output = batch_workspace(field_prototype, length(transform.window), batchsize)
    weights = get_window_weights(transform.window, field_prototype)
    plan = plan_fft!(output, 1)
    return CompiledWindowedFourierField(transform.window, weights, output, plan)
end

function compile_transform(
    transform::WindowedFourierDensity,
    field_prototype::AbstractVector,
    batchsize,
)
    check_window_bounds(transform.window, length(field_prototype))
    output = batch_workspace(field_prototype, length(transform.window), batchsize)
    weights = get_window_weights(transform.window, field_prototype)
    plan = plan_fft!(output, 1)
    return CompiledWindowedFourierDensity(transform.window, weights, output, plan)
end


function evaluate!(compiled::CompiledPositionField, samples)
    compiled.output .= samples
    return compiled.output
end

function evaluate!(compiled::CompiledPositionDensity, samples)
    compiled.output .= abs2.(samples)
    return compiled.output
end

function evaluate!(compiled::CompiledWindowedFourierField, samples)
    apply_window!(compiled.output, samples, compiled.window, compiled.weights)
    compiled.plan * compiled.output
    return compiled.output
end

function evaluate!(compiled::CompiledWindowedFourierDensity, samples)
    apply_window!(compiled.output, samples, compiled.window, compiled.weights)
    compiled.plan * compiled.output
    compiled.output .= abs2.(compiled.output)
    return compiled.output
end


"""
    compile_observables(field_prototype, batchsize, observables)

Compile a fixed collection of observable transforms for repeated batches with
the given number of samples.
"""
function compile_observables(field_prototype::AbstractVector, batchsize::Integer, observables)
    batchsize >= 1 || throw(ArgumentError("batchsize must be positive"))

    transforms = Dict{ObservableTransform,CompiledObservableTransform}()
    for observable in observables
        observable isa ObservableTransform ||
            throw(ArgumentError("all observables must be ObservableTransform values"))
        haskey(transforms, observable) &&
            throw(ArgumentError("duplicate observable: $observable"))
        transforms[observable] = compile_transform(observable, field_prototype, batchsize)
    end
    isempty(transforms) && throw(ArgumentError("at least one observable is required"))

    return CompiledObservables(transforms, Int(batchsize))
end
