# Multi-variate online statitics

| **Documentation**               | **License**                     | **Build Status**                                                | **Code Coverage**                   |
|:--------------------------------|:--------------------------------|:----------------------------------------------------------------|:------------------------------------|
| [![][doc-dev-img]][doc-dev-url] | [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] |

**MultivariateOnlineStatistics** is a [Julia](http://julialang.org/) package to
estimate statistical moments of multi-variate data.  Computation are performed
*on-line* that is in one pass, as the data is obtained.

## Installation

`MultivariateOnlineStatistics` is not yet an [offical Julia
package](https://pkg.julialang.org/) but its installation can be as easy as:

```julia
… pkg> add https://github.com/emmt/MultivariateOnlineStatistics.jl
```

where `… pkg>` stands for the package manager prompt (the ellipsis `…` denotes
your current environment).  To start Julia's package manager, launch Julia and,
at the [REPL of
Julia](https://docs.julialang.org/en/stable/manual/interacting-with-julia/),
hit the `]` key; you should get the above `… pkg>` prompt.  To revert to
Julia's REPL, just hit the `Backspace` key at the `… pkg>` prompt.

To install `MultivariateOnlineStatistics` in a Julia script, write:

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/emmt/MultivariateOnlineStatistics.jl",
                    rev="master"));
```

This also works from the Julia REPL.

In any cases, you may use the URL
`git@github.com:emmt/MultivariateOnlineStatistics.jl` if you want to use `ssh`
instead of HTTPS.


[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/MultivariateOnlineStatistics.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/MultivariateOnlineStatistics.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/MultivariateOnlineStatistics.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/MultivariateOnlineStatistics.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/MultivariateOnlineStatistics.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/MultivariateOnlineStatistics-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/MultivariateOnlineStatistics.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/MultivariateOnlineStatistics.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/MultivariateOnlineStatistics.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/MultivariateOnlineStatistics.jl?branch=master
