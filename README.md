# Multi-variate online statistics

[![Documentation][doc-dev-img]][doc-dev-url]
[![License][license-img]][license-url]
[![Build Status][github-ci-img]][github-ci-url]
[![Build Status][appveyor-img]][appveyor-url]
[![Coverage][coveralls-img]][coveralls-url]

`MultivariateOnlineStatistics` is a [Julia](http://julialang.org/) package to
estimate statistical moments of multi-variate data.  Computation are performed
*on-line* that is in one pass, as the data is obtained.


## Documentation

### Construction

To create an object `A` to collect the `L` first statistical moments of
independent `N`-dimensional data of dimensions `dims`, call:

```julia
A = IndependentStatistics{L,T,N}(dims)
```

Type parameter `T` is the floating-point type for computed statistcs.  The
number `N` of dimensions may be omitted as it must be equal to `length(dims)`:

```julia
A = IndependentStatistics{L,T}(dims)
```

For fine tuning the type of storage used by the object, the arrays `s1`, `s2`,
..., and `sL` storing the statistics may be provided as an `L`-tuple:

```julia
A = IndependentStatistics((s1, s2, ..., sL))
```

where `s1`, `s2`, ..., and `sL` are arrays having the same floating-point
element type and the same indices.  The storage arrays will be zero-filled.

If storage arrays already contain statistical moments collected from a number
of independent samples, then call the constructor with the number `n` of
samples specified as the last argument:

```julia
A = IndependentStatistics((s1, s2, ..., sL), n)
```

In that case, input arrays `s1`, `s2`, ..., and `sL` must have been correctly
intialized, typically as follows (∀i):

 ```julia
s1[i] = (x_1[i] + ... + x_n[i])/n
s2[i] = (x_1[i] - s1[i])^2 + ... + (x_n[i] - s1[i])^2
...
sL[i] = (x_1[i] - s1[i])^L + ... + (x_n[i] - s1[i])^L
```

with `x_j` the `j`-th data sample and where index `i` may be multi-dimensional.
That is to say that `s1` is the element-wise empirical mean of the samples
while `s2`, ..., and `sL` are the sum over the samples of the element-wise
sample difference with their element-wise empirical mean raised to the
corresponding power.


### Uaage

Assuming `A` is an instance of `IndependentStatistics`, then collecting
statistics from more samples is done by:

```julia
push!(A, x...) -> A
merge!(A, itr) -> A
```

where each `x...` is a single data sample, an (abstract) array of suitable
dimension, while `itr` is an iterable object which yields independent data
samples.  The recurrence formula of Welford (1962) is used to avoid loss of
precision due to rounding errors.

:warning: It is assumed that data samples are mutually independent.

If `B` is another instance of `IndependentStatistics`, the statistics collected
by `B` can be merged into `A` by:

```julia
merge!(A, B) -> A
```

In this case, the recurrence formula of Chan, Golub and LeVeque (1979) is used
to avoid loss of precision due to rounding errors.

To retrieve statistics, a number of methods from the `Statistics` and
`StatsBase` package are re-exported:

```julia
nobs(A)                # the number of independent samples
mean(A)                # the element-wise sample mean
var(A; corrected=true) # the element-wise sample variance
std(A; corrected=true) # the element-wise sample standard deviation
```

If keyword `corrected` is true (the default) then an unbiased estimator is
returned; otherwise, the maximum-likelihood estimator is returned.

It is also possible to retrieve the statistical moments for a given data index:

```julia
mean(A, I...)                # sample mean at indices I...
var(A, I...; corrected=true) # sample variance at indices I...
std(A, I...; corrected=true) # sample standard deviation at indices I...
```

The following basic methods are also applicable to an instance of
`IndependentStatistics`:

```julia
ndims(A)   # the number of dimensions of a data sample
size(A)    # the dimensions of a data sample
size(A, k) # the k-th dimension of a data sample
axes(A)    # the axes of a data sample
axes(A, k) # the k-th axis of a data sample
eltype(A)  # the floating-point type of the collected statistics
order(A)   # the maximum order of statistical moments
```


## Installation

The easiest way to install `MultivariateOnlineStatistics` is via Julia registry
[`EmmtRegistry`](https://github.com/emmt/EmmtRegistry):

```julia
using Pkg
pkg"registry add https://github.com/emmt/EmmtRegistry"
pkg"add MultivariateOnlineStatistics"
```


[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/MultivariateOnlineStatistics.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/MultivariateOnlineStatistics.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[github-ci-img]: https://github.com/emmt/MultivariateOnlineStatistics.jl/actions/workflows/CI.yml/badge.svg?branch=master
[github-ci-url]: https://github.com/emmt/MultivariateOnlineStatistics.jl/actions/workflows/CI.yml?query=branch%3Amaster

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/MultivariateOnlineStatistics.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/MultivariateOnlineStatistics-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/MultivariateOnlineStatistics.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/MultivariateOnlineStatistics.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/MultivariateOnlineStatistics.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/MultivariateOnlineStatistics.jl?branch=master
