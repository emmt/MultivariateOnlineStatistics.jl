"""

Module `MultivariateOnlineStatistics` deals with statistics of multi-variate
data.  Computation are performed *on-line* that is in one pass, as the data is
obtained.

"""
module MultivariateOnlineStatistics

export
    IndependentStatistics,
    mean,
    nobs,
    order,
    std,
    var

using Base: OneTo

using Statistics
import Statistics: mean, std, var

using StatsBase
import StatsBase: nobs

"""
    IndependentStatistics{L,T,N}(dims)

yields an object to store the `L` first statistical moments of independent
`N`-dimensional data of dimensions `dims`.  Type parameter `T` is the
floating-point type of stored information.  The number `N` of dimensions may be
omitted as it is equal to `length(dims)`.

For fine tuning the type of storage used by the object, the arrays `s1`, `s2`,
..., and `sL` storing the statistics may be provided as an `L`-tuple:

    IndependentStatistics((s1, s2, ..., sL))

where `s1`, `s2`, ..., and `sL` are arrays having the same floating-point
element type and the same indices.  The storage arrays will be zero-filled.

If storage arrays already contain statistical moments collected from a number
of independent samples, then call the constructor with the number `n` of
samples specified as the last argument:

    IndependentStatistics((s1, s2, ..., sL), n)

In that case, input arrays `s1`, `s2`, ..., and `sL` must have been correctly
intialized, typically as follows (∀i):

    s1[i] = (x_1[i] + ... + x_n[i])/n
    s2[i] = (x_1[i] - s1[i])^2 + ... + (x_n[i] - s1[i])^2
    ...
    sL[i] = (x_1[i] - s1[i])^L + ... + (x_n[i] - s1[i])^L

with `x_j` the `j`-th data sample and where index `i` may be multi-dimensional.
That is to say that `s1` is the element-wise empirical mean of the samples
while `s2`, ..., and `sL` are the sum over the samples of the element-wise
sample difference with their element-wise empirical mean raised to the
corresponding power.

Assuming `A` is an instance of `IndependentStatistics`, then collecting
statistics from more samples is done by:

    push!(A, x...) -> A
    merge!(A, itr) -> A

where each `x...` is a single data sample, an (abstract) array of suitable
dimension, while `itr` is an iterable object which yields independent data
samples.  The recurrence formula of Welford (1962) is used to avoid loss of
precision due to rounding errors.

!!! note
    It is assumed that data samples are mutually independent.

If `B` is another instance of `IndependentStatistics`, the statistics collected
by `B` can be merged into `A` by:

    merge!(A, B) -> A

In this case, the recurrence formula of Chan, Golub and LeVeque (1979) is used
to avoid loss of precision due to rounding errors.

To retrieve statistics, a number of methods from the `Statistics` and
`StatsBase` package are re-exported:

    nobs(A)                # the number of independent samples
    mean(A)                # the element-wise sample mean
    var(A; corrected=true) # the element-wise sample variance
    std(A; corrected=true) # the element-wise sample standard deviation

If keyword `corrected` is true (the default) then an unbiased estimator
is returned; otherwise, the maximum-likelihood estimator is returned.

The following basic methods are also applicable to an instance of
`IndependentStatistics`:

    ndims(A)   # the number of dimensions of a data sample
    size(A)    # the dimensions of a data sample
    size(A, k) # the k-th dimension of a data sample
    axes(A)    # the axes of a data sample
    axes(A, k) # the k-th axis of a data sample
    eltype(A)  # the floating-point type of the collected statistics
    order(A)   # the maximum order of statistical moments

"""
mutable struct IndependentStatistics{L,T<:AbstractFloat,N,
                                     A<:AbstractArray{T,N}}
    s::NTuple{L,A} # collected statistical moments
    n::Int         # number of samples
    function IndependentStatistics{L,T,N,A}(
        s::NTuple{L,A},
        n::Integer) where {T<:AbstractFloat,N,L,
                           A<:AbstractArray{T,N}}
        have_same_axes(s...) ||
            dimension_mismatch("storage arrays have different indices")
        n ≥ 0 || throw(ArgumentError("bad number of samples"))
        return new{L,T,N,A}(s, n)
    end
end

# When number of samples is not provided, assume 0 samples and zero-fill after
# building with the inner constructor (to check correctness of arguments).
function IndependentStatistics{L,T,N,A}(
    s::NTuple{L,A}) where {T<:AbstractFloat,N,L,A<:AbstractArray{T,N}}
    empty!(IndependentStatistics{L,T,N,A}(s, 0))
end

# For basic constructors, just provide the missing type parameters and call one
# of the fully qualified constructors.
for f in (:(IndependentStatistics{L,T,N}),
          :(IndependentStatistics{L,T}),
          :(IndependentStatistics{L}),
          :(IndependentStatistics))
    @eval begin
        function $f(s::NTuple{L,A}) where {T<:AbstractFloat,N,L,
                                           A<:AbstractArray{T,N}}
            IndependentStatistics{L,T,N,A}(s)
        end
        function $f(s::NTuple{L,A}, n::Integer) where {T<:AbstractFloat,N,L,
                                                       A<:AbstractArray{T,N}}
            IndependentStatistics{L,T,N,A}(s, n)
        end
    end
end

IndependentStatistics{L,T}(dims::Integer...) where {L,T} =
    IndependentStatistics{L,T}(to_size(dims))
IndependentStatistics{L,T}(dims::Dims{N}) where {L,T,N} =
    IndependentStatistics(ntuple(x -> zeros(T, dims), L), 0)

IndependentStatistics{L,T,N}(dims::Integer...) where {L,T,N} =
    IndependentStatistics{L,T,N}(dims)
IndependentStatistics{L,T,N}(dims::NTuple{N,Integer}) where {L,T,N} =
    IndependentStatistics{L,T}(to_size(dims))
IndependentStatistics{L,T,N}(dims::NTuple{Np,Integer}) where {L,T,N,Np} =
    throw(ArgumentError("number of dimensions is not equal to type-parameter"))

"""
    MultivariateOnlineStatistics.storage(A)

yields a tuple of the arrays storing the statistics collected by `A`.

---
    MultivariateOnlineStatistics.storage(A, k)

yields the `k`-th array storing the statistics collected by `A`.

"""
storage(A::IndependentStatistics) = A.s
storage(A::IndependentStatistics, k) = A.s[k]

"""
    order(A)

yields the highest order of the statistical moments collected by `A`.

"""
order(A::IndependentStatistics) = order(typeof(A))
order(::Type{<:IndependentStatistics{L,T,N}}) where {L,T,N} = L

# Extend simple Base methods.
Base.eltype(A::IndependentStatistics) = eltype(typeof(A))
Base.ndims(A::IndependentStatistics) = ndims(typeof(A))
Base.eltype(::Type{<:IndependentStatistics{L,T}}) where {L,T} = T
Base.ndims(::Type{<:IndependentStatistics{L,T,N}}) where {L,T,N} = N
Base.size(A::IndependentStatistics) = size(storage(A, 1))
Base.size(A::IndependentStatistics, k) = size(storage(A, 1), k)
Base.axes(A::IndependentStatistics) = axes(storage(A, 1))
Base.axes(A::IndependentStatistics, k) = axes(storage(A, 1), k)

"""
     MultivariateOnlineStatistics.checked_axes(A) -> I

yields the indices of the data integrated by `A`.  Compared to `axes(A)`, this
method also checks the number of samples and the indices of all the storage
arrays used by `A`.

"""
@inline function checked_axes(A::IndependentStatistics{L}) where {L}
    A.n ≥ 0 || throw_bad_number_of_samples()
    I = axes(storage(A, 1))
    L < 2 || all_match(I, axes, storage(A, 2:L)...) ||
        throw_bad_storage_indices()
    return I
end

function Base.empty!(A::IndependentStatistics{L,T}) where {L,T}
    for s in storage(A)
        fill!(s, zero(eltype(s)))
    end
    A.n = 0
    return A
end

# Extend StatsBase methods.
nobs(A::IndependentStatistics) = A.n

# Extend Statistics methods.
mean(A::IndependentStatistics) = storage(A, 1)
var(A::IndependentStatistics{L,T,N}; corrected::Bool=true) where {L,T,N} =
    var!(Array{T,N}(undef, size(A)), A; corrected=corrected)
std(A::IndependentStatistics{L,T,N}; corrected::Bool=true) where {L,T,N} =
    std!(Array{T,N}(undef, size(A)), A; corrected=corrected)

function var!(dst::AbstractArray{<:AbstractFloat,N},
              A::IndependentStatistics{<:Any,<:AbstractFloat,N};
              corrected::Bool=true) where {N}
    mapvar!(identity, dst, A; corrected = corrected)
end

function std!(dst::AbstractArray{<:AbstractFloat,N},
              A::IndependentStatistics{<:Any,<:AbstractFloat,N};
              corrected::Bool=true) where {N}
    mapvar!(sqrt, dst, A; corrected = corrected)
end

# Map the variance.
function mapvar!(f,
                 dst::AbstractArray{<:AbstractFloat,N},
                 A::IndependentStatistics{L,T,N};
                 corrected::Bool=true) where {L,T,N}
    L ≥ 2 || error("2nd order statistical moment is not available")
    s2 = storage(A, 2)
    axes(dst) == axes(s2) ||
        dimension_mismatch("destination has incompatible indices")
    n = nobs(A)
    n ≥ 0 || throw_bad_number_of_samples()
    if corrected
        n > 1 || error("not enough samples")
        n -= 1
    end
    if n > 1
        a = one(T)/n
        @inbounds @simd for i in eachindex(dst, s2)
            dst[i] = f(a*s2[i])
        end
    elseif n == 1
        @inbounds @simd for i in eachindex(dst, s2)
            dst[i] = f(s2[i])
        end
    else
        fill!(dst, f(zero(eltype(dst))))
    end
    return dst
end

function Base.push!(A::IndependentStatistics{L,T,N},
                    x::AbstractArray{<:Real,N}) where {L,T<:AbstractFloat,N}
    axes(x) == checked_axes(A) ||
        dimension_mismatch("data sample has incompatible indices")
    return unsafe_push!(A, x)
end

# Push without checking arguments.
function unsafe_push!(A::IndependentStatistics{L,T,N},
                      x::AbstractArray{<:Real,N}) where {L,T<:AbstractFloat,N}
    n = A.n
    if n == 0
        # No data have yet been collected by A.
        s1 = storage(A, 1)
        @inbounds @simd for i in eachindex(s1, x)
            s1[i] = x[i]
        end
    elseif L == 1
        s1, = storage(A)
        w1 = T(1)/(n + 1)
        @inbounds @simd for i in eachindex(s1, x)
            # Apply the recurrence given in Welford (1962).
            s1[i] += w1*(x[i] - s1[i])
        end
    elseif L == 2
        s1, s2 = storage(A)
        w1 = T(1)/(n + 1)
        wn = T(n)/(n + 1)
        @inbounds @simd for i in eachindex(s1, s2, x)
            # Apply the recurrence given in Welford (1962).  Note that this
            # guarantees that s2[i] ≥ O always hold.
            u = x[i] - s1[i]
            s1[i] += w1*u
            s2[i] += wn*u^2
        end
    else
        error("statistical moments of order higher than 2 not yet implemented")
    end
    A.n += 1
    return A
end

function Base.copyto!(dst::IndependentStatistics{L,<:AbstractFloat,N},
                      src::IndependentStatistics{L,<:AbstractFloat,N}) where {L,N}
    copy!(dst, src)
end

function Base.copy!(dst::IndependentStatistics{L,<:AbstractFloat,N},
                    src::IndependentStatistics{L,<:AbstractFloat,N}) where {L,N}
    checked_axes(dst) == checked_axes(src) ||
        dimension_mismatch("statistics have incompatible indices")
    return unsafe_copy!(dst, src)
end

# Copy without checking arguments.
function unsafe_copy!(dst::IndependentStatistics{L,<:AbstractFloat,N},
                      src::IndependentStatistics{L,<:AbstractFloat,N}) where {L,N}
    for k in 1:L
        copy!(storage(dst, k), storage(src, k))
    end
    dst.n = src.n
    return dst
end

function Base.merge!(A::IndependentStatistics, itr)
    I = checked_axes(A)
    for x in itr
        axes(x) == I ||
            dimension_mismatch("data sample has incompatible indices")
        unsafe_push!(A, x)
    end
    return A
end

function Base.merge!(A::IndependentStatistics{L,<:AbstractFloat,N},
                     B::IndependentStatistics{L,<:AbstractFloat,N}) where {L,N}
    checked_axes(A) == checked_axes(B) ||
        dimension_mismatch("statistics have incompatible indices")
    if A.n == 0
        # No data have been collected in A, just copy B if any data there.
        if B.n > 0
            unsafe_copy!(A, B)
        end
    elseif B.n == 1
        # A single sample has been collected by B, it is sufficient to push it
        # into A.
        unsafe_push!(A, storage(B, 1))
    else
        # Some samples have been integrated in A and B.
        _merge!(A, B)
    end
    return A
end

function _merge!(A::IndependentStatistics{1,T,N},
                 B::IndependentStatistics{1,<:AbstractFloat,N}) where {T,N}
    # Apply the formula given in Chan, Golub and LeVeque (1979) at the
    # numerical precision of the destination to combine the collected
    # statistics.
    As1, = storage(A)
    Bs1, = storage(B)
    n = A.n + B.n
    wa = T(A.n)/n
    wb = T(B.n)/n
    wab = wa*B.n
    @inbounds @simd for i in eachindex(As1, As2, Bs1, Bs2)
        As1[i] = wa*As1[i] + wb*T(Bs1[i])
    end
    A.n = n
    nothing
end

function _merge!(A::IndependentStatistics{2,T,N},
                 B::IndependentStatistics{2,<:AbstractFloat,N}) where {T,N}
    # Apply the formulae given in Chan, Golub and LeVeque (1979) at the
    # numerical precision of the destination to combine the collected
    # statistics.
    As1, As2 = storage(A)
    Bs1, Bs2 = storage(B)
    n = A.n + B.n
    wa = T(A.n)/n
    wb = T(B.n)/n
    wab = wa*B.n
    @inbounds @simd for i in eachindex(As1, As2, Bs1, Bs2)
        a, b = As1[i], T(Bs1[i])
        As1[i] = wa*a + wb*b
        As2[i] += T(Bs2[i]) + wab*(a - b)^2
    end
    A.n = n
    nothing
end

function _merge!(A::IndependentStatistics{L,T,N},
                 B::IndependentStatistics{L,<:AbstractFloat,N}) where {L,T,N}
    error("statistical moments of order higher than 2 not yet implemented")
end

# This method borrowed from ArrayTools.
"""
    MultivariateOnlineStatistics.all_match(val, f, args...) -> bool

yields as soon as possible (short-circuit) whether `f(arg) == val` for each
argument `arg` in `args...`.  The returned value is `true` if there are no
arguments after `f`.

"""
all_match(val, f::Function) = true
all_match(val, f::Function, A) = f(A) == val
@inline all_match(val, f::Function, A, B...) =
    all_match(val, f, A) && all_match(val, f::Function, B...)

"""
    MultivariateOnlineStatistics.same_axes(A...) -> inds

yields the axes of arrays `A...` throwing an error if these arrays have
not the same axes.

"""
same_axes(A::AbstractArray) = axes(A)
@inline same_axes(A::AbstractArray, B::AbstractArray...) = begin
    I = axes(A)
    all_match(I, axes, B...) ||
        dimension_mismatch("arguments have different indices")
    return I
end

"""
    MultivariateOnlineStatistics.have_same_axes(A...) -> bool

yields whether arrays `A...` have the same axes.

"""
have_same_axes(A::AbstractArray) = true
@inline have_same_axes(A::AbstractArray, B::AbstractArray...) =
    all_match(axes(A), axes, B...)

"""
    MultivariateOnlineStatistics.to_int(x)

yields integer `x` converted to an `Int`.

"""
to_int(x::Int) = x
to_int(x::Integer) = Int(x)

"""
    MultivariateOnlineStatistics.to_size(dims)

yields `dims` converted to an array size, that is an instance of `Dims`.

"""
to_size(x::Dims) = x
to_size(x::NTuple{N,Integer}) where {N} = map(to_int, x)
to_size(x::Integer...) = to_size(x)
to_size(x::Integer) = to_size(to_int(x),)

@noinline dimension_mismatch(args...) = dimension_mismatch(string(args...))
@noinline dimension_mismatch(msg::AbstractString) =
    throw(DimensionMismatch(msg))

@noinline assertion_error(args...) = assertion_error(string(args...))
@noinline assertion_error(msg::AbstractString) = throw(AssertionError(msg))

# Throw an AssertionError when contents of a structure is inconsistent.
@noinline throw_bad_number_of_samples() =
    assertion_error("bad number of samples")
@noinline throw_bad_storage_indices() =
    assertion_error("storage arrays have different indices")

end # module
