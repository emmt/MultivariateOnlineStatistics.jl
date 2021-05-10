module TestingMultivariateOnlineStatistics

using MultivariateOnlineStatistics, Test, Statistics, StatsBase

using MultivariateOnlineStatistics: storage

@testset "MultivariateOnlineStatistics" begin
    T = Float64
    dims = (2, 3, 4)
    N = length(dims)
    numb = 17
    X = [rand(T, dims) for t in 1:numb]
    # Constructors and initialization
    A = IndependentStatistics{2,T}(dims)
    B = IndependentStatistics{2,T}(dims...)
    C = IndependentStatistics{2,T,length(dims)}(map(Int16, dims)...)
    @test_throws ArgumentError IndependentStatistics{2,T,length(dims)+1}(dims)
    @test eltype(A) === T
    @test ndims(A) === length(dims)
    @test nobs(A) == 0
    @test extrema(mean(A)) === (zero(T), zero(T))
    @test extrema(std(A; corrected = false)) === (zero(T), zero(T))
    @test extrema(var(A; corrected = false)) === (zero(T), zero(T))
    @test_throws ErrorException std(A; corrected = true)
    @test_throws ErrorException var(A; corrected = true)
    push!(A, X[1])
    @test nobs(A) == 1
    @test mean(A) == X[1]
    @test extrema(std(A; corrected = false)) === (zero(T), zero(T))
    @test extrema(var(A; corrected = false)) === (zero(T), zero(T))
    @test_throws ErrorException std(A; corrected = true)
    @test_throws ErrorException var(A; corrected = true)
    push!(A, X[2])
    m1 = (X[1] + X[2])./2
    m2 = (X[1] - m1).^2 + (X[2] - m1).^2
    @test nobs(A) == 2
    @test mean(A) ≈ m1
    @test std(A; corrected = false) ≈ sqrt.(m2./2)
    @test var(A; corrected = false) ≈ m2./2
    @test std(A; corrected = true)  ≈ sqrt.(m2)
    @test var(A; corrected = true)  ≈ m2
    push!(A, X[3])
    m1 = (X[1] + X[2] + X[3])./3
    m2 = (X[1] - m1).^2 + (X[2] - m1).^2 + (X[3] - m1).^2
    @test nobs(A) == 3
    @test mean(A) ≈ m1
    @test std(A; corrected = false) ≈ sqrt.(m2./3)
    @test var(A; corrected = false) ≈ m2./3
    @test std(A; corrected = true)  ≈ sqrt.(m2./2)
    @test var(A; corrected = true)  ≈ m2./2
    push!(A, X[4])
    m1 = (X[1] + X[2] + X[3] + X[4])./4
    m2 = (X[1] - m1).^2 + (X[2] - m1).^2 + (X[3] - m1).^2 + (X[4] - m1).^2
    @test nobs(A) == 4
    @test mean(A) ≈ m1
    @test std(A; corrected = false) ≈ sqrt.(m2./4)
    @test var(A; corrected = false) ≈ m2./4
    @test std(A; corrected = true)  ≈ sqrt.(m2./3)
    @test var(A; corrected = true)  ≈ m2./3
    # Push remaining terms in two different ways.
    push!(A, X[5:7]...)
    push!(A, X[8:end])
    @test nobs(A) == length(X)
    # Push all terms at once.
    push!(B, X)
    @test nobs(B) == nobs(A)
    for (a, b) in zip(storage(A), storage(B))
        @test a == b
    end
    # Push in reverse order.
    for i in reverse(1:length(X))
       push!(C, X[i])
    end
    @test nobs(C) == nobs(A)
    for (a, c) in zip(storage(A), storage(C))
        @test a ≈ c
    end
    # Empty statistics.
    empty!(B)
    @test eltype(B) === T
    @test ndims(B) === length(dims)
    @test nobs(B) == 0
    for b in storage(B)
        @test extrema(b) === (zero(T), zero(T))
    end
    empty!(C)
    # Collect statistics by parts.
    n1 = length(X)÷3
    n2 = length(X)÷2
    for i in 1:n1
        push!(B, X[i])
    end
    for i in n1+1:n1+n2
        push!(C, X[i])
    end
    for i in n1+n2+1:length(X)
        push!(B, X[i])
    end
    merge!(B, C)
    @test nobs(B) == nobs(A)
    for (a, b) in zip(storage(A), storage(B))
        @test a ≈ b
    end
    # Check merge! into an empty statistics.
    empty!(B)
    merge!(B, A)
    @test nobs(B) == nobs(A)
    for (a, b) in zip(storage(A), storage(B))
        @test a == b
    end
    # Check merge! from a single sample statistics.
    empty!(A)
    empty!(B)
    for x in X
        push!(A, x)
        merge!(B, push!(empty!(C), x))
        if nobs(A) ≥ 5
            break
        end
    end
    @test nobs(B) == nobs(A)
    for (a, b) in zip(storage(A), storage(B))
        @test a == b
    end
end

end # module
