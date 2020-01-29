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
    mean_by_time(df, target)

Filter DataFrame df by time directive and get averages

time directive must be in :hour (24 hour), :day (365 day), :quarter (4 quarter)
"""
function time_mean(df::DataFrame, target::Symbol, tdir::Symbol)
    if tdir == :hour
        time_len = 24
    elseif tdir == :week
        time_len = 53
    elseif tdir == :month
        time_len = 12
    elseif tdir == :quarter
        time_len = 4
    else
        error("Time directive must be between :hour, :day, :month, :quarter")
    end

    avgs = zeros(time_len)

    for time=1:time_len
        if tdir == :hour
            _df = @from i in df begin
                @where hour(i.date) == time - 1
                @orderby i.date
                @select i
                @collect DataFrame
            end
        elseif tdir == :week
            _df = @from i in df begin
                @where week(i.date) == time
                @orderby i.date
                @select i
                @collect DataFrame
            end
        elseif tdir == :month
            _df = @from i in df begin
                @where month(i.date) == time
                @orderby i.date
                @select i
                @collect DataFrame
            end
        elseif tdir == :quarter
            _df = @from i in df begin
                @where quarterofyear(i.date) == time
                @orderby i.date
                @select i
                @collect DataFrame
            end
        end

        avgs[time] = mean(_df[!, target])
    end

    avgs
end

"""
    pdf(data)

estimate probability density function 
    by kernel density estimation
"""
function pdf(data::AbstractArray, npts::Integer)
    KernelDensity.kde_lscv(data, npoints = npts)
end
