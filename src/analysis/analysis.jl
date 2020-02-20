"""
    detect_stationary(ta, μ, σ)
`ta` is a TimeArray, `μ` and `σ` is global mean and std of time array
"""
function detect_stationary(ta::TimeArray, μ::AbstractFloat, σ::AbstractFloat)

end

function compute_inttscale(mw)
    #window = length(mw)
    #_autocor = StatsBase.autocor(mw, 1:window-1)
    #_autocor = StatsBase.autocor(mw)
    _autocor = StatsBase.autocor(mw, 0:50)

    if isnan(sum(_autocor))
        return 0.0
    end
    
    # summation autocor
    x = 1:length(_autocor)
    integrate(x, _autocor, SimpsonEven())
end

function diff_mean(mw)
    #window = length(mw)
    #_autocor = StatsBase.autocor(mw, 1:window-1)
    #_autocor = StatsBase.autocor(mw)
    _autocor = StatsBase.autocor(mw, 0:50)

    if isnan(sum(_autocor))
        return 0.0
    end
    
    # summation autocor
    x = 1:length(_autocor)
    integrate(x, _autocor, SimpsonEven())
end

function mean_aucotor(ta::TimeArray, window::Integer; padding::Bool = false)
    A = values(ta)
    # moving window
    mw = map(1:length(ta)-window+1) do i 
        A[i:i+(window-1)]
    end
    padding && (mw = [fill(zeros(window), window - 1); mw])

    _autocors = []
    for w in mw
        #_auto = StatsBase.autocor(w, 1:window-1)
        #_auto = StatsBase.autocor(w)
        _auto = StatsBase.autocor(w, 0:50)

        if !isnan(sum(_auto))
            push!(_autocors, _auto)
        end
    end

    mean(hcat(_autocors...), dims=[2])
end

"""
    pdf(data)

estimate probability density function 
    by kernel density estimation
"""
function pdf(data::AbstractVector, npts::Integer; method::Symbol=:kernel)
    if method == :kernel
        _pdfKDE = KernelDensity.kde(data, npoints = npts)
        x = _pdfKDE.x
        y = _pdfKDE.density
    elseif method == :linear
        _pdfKDE = KernelDensity.kde_lscv(data, npoints = npts)
        x = _pdfKDE.x
        y = _pdfKDE.density
    elseif method == :gibbs
        #bd = kde!(float.(data))
        #bmin, bmax = getKDERange(bd) 
        #x = range(bmin, stop=bmax, length=npts)
        #y = evaluateDualTree(bd, x)
    elseif method == :manual
        # Freedman-Diaconis’s Rule
        binsize = 2.0 * iqr(data) * length(data)^(-1//3) * 3
        # Scott’s Rule
        # binsize = 3.49 * std(data) * length(data)^(-1//3) * 3
        # Rice's Rule
        # binsize = length(data)^(-1//3) * 2
        @show binsize
        minx = float.(minimum(data))
        maxx = float.(maximum(data))
        nbins = Int(round((maxx - minx) / float(binsize)))

        _x = minx:binsize:maxx
        x = minx:binsize:(maxx - binsize)
        y = map(1:length(x)) do i
            count(_i -> (_x[i] <= _i < _x[i + 1]), data) / float(length(data))
        end
    end

    x, y
end
