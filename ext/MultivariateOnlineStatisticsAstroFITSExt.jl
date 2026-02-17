module MultivariateOnlineStatisticsAstroFITSExt

if isdefined(Base, :get_extension)
    using MultivariateOnlineStatistics, AstroFITS, FITSHeaders
    using MultivariateOnlineStatistics:
        argument_error, dimension_mismatch, STAT_HDU_KWD, STAT_NB_SAMPLES_KWD
    import MultivariateOnlineStatistics: isa_stat_hdu
    import AstroFITS: FitsImageHDU, OptionalHeader
    import Base: read, write
else
    using ..MultivariateOnlineStatistics, ..AstroFITS, ..FITSHeaders
    using ..MultivariateOnlineStatistics:
        argument_error, dimension_mismatch, STAT_HDU_KWD, STAT_NB_SAMPLES_KWD
    import ..MultivariateOnlineStatistics: isa_stat_hdu
    import ..AstroFITS: FitsImageHDU, OptionalHeader
    import ..Base: read, write
end

function isa_stat_hdu(hdu)
    (hdu isa AstroFITS.FitsImageHDU
     && haskey(hdu, STAT_HDU_KWD)
     && hdu[STAT_HDU_KWD].type == FITSHeaders.FITS_LOGICAL
     && hdu[STAT_HDU_KWD].logical)
end

function write(file::FitsFile, header::OptionalHeader, data::IndependentStatistics, args...)
    write(file, header, data)
    write(file, args...)
end

function write(file::FitsFile, hdr::OptionalHeader, stat::IndependentStatistics)
    write(merge!(FitsImageHDU(file, stat), hdr), stat)
    return file # returns the file not the HDU
end

FitsImageHDU(file::FitsFile, stat::IndependentStatistics{<:Any,T}) where {T} =
    FitsImageHDU{T}(file, stat)

FitsImageHDU{T}(file::FitsFile, stat::IndependentStatistics{<:Any,<:Any,N}) where {T,N} =
    FitsImageHDU{T,N+1}(file, stat)

function FitsImageHDU{T,N1}(file::FitsFile,
                            stat::IndependentStatistics{<:Any,<:Any,N}) where {T,N1,N}
    N1 == N+1 || dimension_mismatch("HDU has $N1 dimensions instead of $(N+1)")
    return FitsImageHDU{T,N1}(file, (size(stat)..., order(stat)))
end

function write(hdu::FitsImageHDU{<:Any,N1},
               stat::IndependentStatistics{L,T,N}) where {N1,L,T,N}
    N1 == N+1 || dimension_mismatch("HDU has $N1 dimensions instead of $(N+1)")

    hdu.data_size == (size(stat)..., order(stat)) || dimension_mismatch(
        "HDU size is $(hdu.data_size) instead of $((size(stat)..., order(stat)))")

    push!(hdu, STAT_HDU_KWD        => (true,   "image is IndependentStatistics data"),
               STAT_NB_SAMPLES_KWD => (stat.n, "number of statistical samples"))

    i = 1
    for l in 1:L
        write(hdu, stat.s[l]; first=i)
        i += length(stat)
    end

    return hdu
end



read(::Type{IndependentStatistics}, hdu::FitsImageHDU; kwds...) =
    read(IndependentStatistics{hdu.data_size[end]}, hdu; kwds...)

read(::Type{IndependentStatistics{L}}, hdu::FitsImageHDU{T}; kwds...) where {L,T} =
    read(IndependentStatistics{L,T}, hdu; kwds...)

read(::Type{IndependentStatistics{L,T}}, hdu::FitsImageHDU{<:Any,N1}; kwds...) where {L,T,N1} =
    read(IndependentStatistics{L,T,N1-1}, hdu; kwds...)

read(::Type{IndependentStatistics{L,T,N}}, hdu::FitsImageHDU; kwds...) where {L,T,N} =
    read(IndependentStatistics{L,T,N,Array{T,N}}, hdu; kwds...)

function read(::Type{IndependentStatistics{L,T,N,A}},
              hdu::FitsImageHDU{<:Any,N1}; kwds...) where {L,T,N,A,N1}
    N1 == N+1 || dimension_mismatch("HDU has $N1 dimensions instead of $(N+1)")

    L == hdu.data_size[N1] || dimension_mismatch(
        "HDU has $(hdu.data_size[N1]) statistical moments instead of $L")
    
    haskey(hdu, STAT_NB_SAMPLES_KWD) || argument_error(
        "HDU miss keyword \"$STAT_NB_SAMPLES_KWD\"")

    hdu[STAT_NB_SAMPLES_KWD].type == FITSHeaders.FITS_INTEGER || argument_error(
        "HDU keyword \"$STAT_NB_SAMPLES_KWD\" is not of type integer")

    isa_stat_hdu(hdu) || @warn("HDU is not declared as IndependentStatistics data")

    n = hdu[STAT_NB_SAMPLES_KWD].integer
    data = read(Array{T,N1}, hdu; kwds...)
    s = NTuple{L,A}( A(sl) for sl in eachslice(data; dims=N1))

    return IndependentStatistics{L,T,N,A}(s,n)
end

end