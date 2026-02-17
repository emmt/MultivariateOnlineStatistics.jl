module TestingMultivariateOnlineStatistics

using MultivariateOnlineStatistics, Test, Statistics, StatsBase, AstroFITS

using MultivariateOnlineStatistics: storage, STAT_HDU_KWD, STAT_NB_SAMPLES_KWD, isa_stat_hdu

# Dummy private function used to "label" the tests.
check(flag::Bool, comment) = flag

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
    avg = IndependentStatistics((Array{T}(undef, dims),))
    @test_throws ArgumentError IndependentStatistics{2,T,length(dims)+1}(dims)
    @test order(A) === 2
    @test order(avg) === 1
    @test eltype(A) === T
    @test ndims(A) === length(dims)
    @test size(A) === dims
    @test length(A) === prod(dims)
    @test ntuple(k -> size(A, k), ndims(A)) === size(A)
    @test axes(A) === map(Base.OneTo, size(A))
    @test ntuple(k -> axes(A, k), ndims(A)) === axes(A)
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
    # Check element-by-element mean.
    arr = mean(A)
    flag = true
    for i in eachindex(arr)
        flag &= (arr[i] ≈ mean(A, i))
    end
    @test check(flag, "mean(A, i::Integer) == mean(A)[i]")
    flag = true
    for i in CartesianIndices(arr)
        flag &= (arr[i] ≈ mean(A, i))
    end
    @test check(flag, "mean(A, i::CartesianIndex) == var(A)[i]")
    flag = true
    for i in CartesianIndices(arr)
        flag &= (arr[i] ≈ mean(A, Tuple(i)))
    end
    @test check(flag, "mean(A, i::Tuple{Vararg{Integer}}) == var(A)[i]")
    # Check element-by-element variance.
    arr = var(A)
    flag = true
    for i in eachindex(arr)
        flag &= (arr[i] ≈ var(A, i))
    end
    @test check(flag, "var(A, i::Integer) == var(A)[i]")
    flag = true
    for i in CartesianIndices(arr)
        flag &= (arr[i] ≈ var(A, i))
    end
    @test check(flag, "var(A, i::CartesianIndex) == var(A)[i]")
    flag = true
    for i in CartesianIndices(arr)
        flag &= (arr[i] ≈ var(A, Tuple(i)))
    end
    @test check(flag, "var(A, i::Tuple{Vararg{Integer}}) == var(A)[i]")
    arr = var(A; corrected=false)
    flag = true
    for i in eachindex(arr)
        flag &= (arr[i] ≈ var(A, i; corrected=false))
    end
    @test check(flag, "var(A, i::Integer; corrected=false) == var(A; corrected=false)[i]")
    # Check element-by-element standard deviation.
    arr = std(A)
    flag = true
    for i in eachindex(arr)
        flag &= (arr[i] ≈ std(A, i))
    end
    @test check(flag, "std(A, i::Integer) == std(A)[i]")
    flag = true
    for i in CartesianIndices(arr)
        flag &= (arr[i] ≈ std(A, i))
    end
    @test check(flag, "std(A, i::CartesianIndex) == std(A)[i]")
    flag = true
    for i in CartesianIndices(arr)
        flag &= (arr[i] ≈ std(A, Tuple(i)))
    end
    @test check(flag, "std(A, i::Tuple{Vararg{Integer}}) == std(A)[i]")
    arr = std(A; corrected=false)
    flag = true
    for i in eachindex(arr)
        flag &= (arr[i] ≈ std(A, i; corrected=false))
    end
    @test check(flag, "std(A, i::Integer; corrected=false) == std(A; corrected=false)[i]")

    # Push remaining terms in two different ways.
    push!(A, X[5:7]...)
    merge!(A, X[8:end])
    @test nobs(A) == length(X)
    # Push all terms at once.
    merge!(B, X)
    @test nobs(B) == length(X)
    for (a, b) in zip(storage(A), storage(B))
        @test a == b
    end
    merge!(avg, X)
    @test nobs(avg) == length(X)
    @test mean(avg) == mean(B)
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

    # Partially qualified constructors.
    merge!(empty!(A), X)
    for B in (IndependentStatistics(storage(A), nobs(A)),
              IndependentStatistics{order(A)}(storage(A), nobs(A)),
              IndependentStatistics{order(A),eltype(A)}(storage(A), nobs(A)),
              IndependentStatistics{order(A),eltype(A),ndims(A)}(storage(A), nobs(A)),)
        @test nobs(B) == length(X)
        @test order(B) == order(A)
        @test eltype(B) == eltype(A)
        @test size(B) == size(A)
    end
    for B in (IndependentStatistics(storage(A)),
              IndependentStatistics{order(A)}(storage(A)),
              IndependentStatistics{order(A),eltype(A)}(storage(A)),
              IndependentStatistics{order(A),eltype(A),ndims(A)}(storage(A)),)
        @test nobs(B) == 0
        @test order(B) == order(A)
        @test eltype(B) == eltype(A)
        @test size(B) == size(A)
    end

    # AstroFITS extension (input/output)
    mktempdir() do dir

        # prepare some initial stat to work with
        p = normpath(dir, "test.fits")
        (W,H,S) = (4,7,15)
        rawdata = rand(W,H,S)
        (L,T,N) = (2, Float64, 2)
        A = Array{T,N}
        stat = merge!(IndependentStatistics{L,T,N}(W,H), eachslice(rawdata; dims=3))
        hdr = FitsHeader("TEST" => 42)
        
        # test writing on path
        @test_nowarn writefits!(p, hdr, stat)
        
        # test writing on FitsFile
        FitsFile(p, "w!") do f
            @test write(f, hdr, stat)[1].data_eltype == T
            @test f[1].data_size == (W,H,L)
            @test isa_stat_hdu(f[1])
        end

        # test FitsImageHDU declaration
        FitsFile(p, "w!") do f
            @test FitsImageHDU(f, stat).data_eltype == T
            @test FitsImageHDU(f, stat).data_size == (W,H,L)
            @test FitsImageHDU{T}(f, stat).data_eltype == T
            @test FitsImageHDU{T}(f, stat).data_size == (W,H,L)
            @test FitsImageHDU{Float32}(f, stat).data_eltype == Float32
            @test FitsImageHDU{Float32}(f, stat).data_size == (W,H,L)
            @test FitsImageHDU{T,3}(f, stat).data_size == (W,H,L)
            msg = "HDU has 4 dimensions instead of $(N+1)"
            @test_throws DimensionMismatch(msg) FitsImageHDU{T,4}(f, stat)
        end

        # test writing on FitsImageHDU
        FitsFile(p, "w!") do f
            hdu = FitsImageHDU(f, stat)
            @test write(hdu, stat).data_eltype == T
            @test write(hdu, stat).data_size == (W,H,L)
            hdu = FitsImageHDU{T,2}(f, (W,H))
            msg = "HDU has 2 dimensions instead of $(N+1)"
            @test_throws DimensionMismatch(msg) write(hdu, stat)
            hdu = FitsImageHDU{T,3}(f, (W,H,3))
            msg = "HDU size is $((W,H,3)) instead of $((W,H,L))"
            @test_throws DimensionMismatch(msg) write(hdu, stat)
        end
        
        # test reading of statistical HDU and equality to our initial stat 
        writefits!(p, hdr, stat)
        @test readfits(IndependentStatistics, p).s == stat.s
        @test readfits(IndependentStatistics, p).n == stat.n
        FitsFile(p, "r") do f
            hdu = f[1]
            @test hdu["TEST"].integer == 42
            @test read(IndependentStatistics, hdu).n == stat.n
            @test read(IndependentStatistics{L}, hdu).n == stat.n
            @test read(IndependentStatistics{L,T}, hdu).n == stat.n
            @test read(IndependentStatistics{L,T,N}, hdu).n == stat.n
            @test read(IndependentStatistics{L,T,N,A}, hdu).n == stat.n
            msg = "HDU has $L statistical moments instead of 3"
            @test_throws DimensionMismatch(msg) read(IndependentStatistics{3}, hdu)
            msg = "HDU has $(N+1) dimensions instead of 4"
            @test_throws DimensionMismatch(msg) read(IndependentStatistics{L,T,3}, hdu)
            @test_throws TypeError read(IndependentStatistics{L,T,N,Vector{T}}, hdu)
        end
        
        # test absence or wrongness of STAT HDU keywords
        writefits!(p, hdr, stat)
        FitsFile(p, "rw") do f
            hdu = f[1]
            delete!(hdu, STAT_HDU_KWD)
            msg = "HDU is not declared as IndependentStatistics data"
            @test_warn msg read(IndependentStatistics, hdu)
            
            hdu[STAT_NB_SAMPLES_KWD] = "wrong"
            msg = "HDU keyword \"$STAT_NB_SAMPLES_KWD\" is not of type integer"
            @test_throws ArgumentError(msg) read(IndependentStatistics, hdu)
            
            delete!(hdu, STAT_NB_SAMPLES_KWD)
            msg = "HDU miss keyword \"$STAT_NB_SAMPLES_KWD\""
            @test_throws ArgumentError(msg) read(IndependentStatistics, hdu)
        end
    end
end

end # module
