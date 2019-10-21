"""
function evolve_OU(df, ycol, norm_prefix, norm_feas,
    μσs, default_FloatType,
    test_wd_idxs, test_idxs,
    filename, test_dates) where I <: Integer

https://pdfs.semanticscholar.org/dfdd/ff87a20a918e7ac101808881b926bb298e95.pdf

The Ornstein-Uhlembeck position process
dXₜ = θ(μ - Xₜ)dt + σdWₜ 
OU dynamics = "mean reverting" term + "white noise" term
"""
function evolve_OU(df::DataFrame, ycol::Symbol, features::Array{Symbol, 1},
    μσs::AbstractNDSparse, default_FloatType::DataType,
    input_size::Integer,
    test_dates::Array{ZonedDateTime,1}) where I <: Integer

    days = 72
    lags = collect(-days:1:days)
    # compute correlation and plot
    crosscorr_input(df, Symbol("norm_", ycol), features, lags)
    #autocorr_input(df, Symbol("norm_", ycol))
    cor = crosscor(df[:, ycol], df[:, ycol], lags)
    # ycol's mean / std
    μ_ycol = μσs[String(ycol), "μ"].value
    σ_ycol = μσs[String(ycol), "σ"].value

    # OU Process parameter
    X₀ = df[1, ycol]
    # https://en.wikipedia.org/wiki/Vasicek_model
    # "speed of reversion". a characterizes the velocity at which such trajectories will regroup around μ in time;
    θ = 0.01
    # "long term mean level". All future trajectories of r will evolve around a mean level μ in the long run;
    μ = μ_ycol
    # "long term variance". All future trajectories of r will regroup around the long term mean with such variance after a long time.
    σ = σ_ycol
    # time interval, sqrt(σ * 2.0 * h) is "instantaneous volatility", measures instant by instant the amplitude of randomness entering the system
    # higher sqrt(σ * 2.0 * h), more randomnesss
    h = 0.001
    #idx_b = findall(x -> x == 0, lags)[1]
    #idx_e = findall(x -> x == 120, lags)[1]
    idx_b = days + 1
    idx_e = 2 * days + 1
    #@show idx_b, idx_e
    T = compute_tau(cor, idx_b, idx_e)
    θ = 1/T
    μ = 0.0
    σ = 1.0
    @show T

    # Wiener Process
    rng = MersenneTwister()
    dW = Normal(0.0, 1.0)

    T = length(test_dates)
    Xₜ = zeros(T)
    X_org = df[!, ycol]
    Xₜ[1] = X₀

    rand_arr = zeros(T)

    # Vasicek Model
    # https://en.wikipedia.org/wiki/Vasicek_model
    for next in 2:T
        Distributions.rand!(rng, dW, rand_arr)
        #μ, σ = mean_and_std(df[next - input_size:next, ycol])
        if next != 2
            X₀ = df[next-1, ycol]
        end
        Xₜ[next] = X₀ + (μ - Xₜ[next-1]) / T + sum(sqrt(σ^2 * 2.0 / T) .* rand_arr[1:next-1])
    end

    Base.Filesystem.mkpath("/mnt/OU/")

    plot_line_OU(DateTime.(test_dates), X_org, Xₜ, ycol, "/mnt/OU/")
end

function compute_tau(vec::Array{F, 1}, idx_b, idx_e) where F<:AbstractFloat
    dlag = 1.0

    M = 0.0
    cnt = 0
    for i = idx_b:idx_e
        if (cnt == 0 || i == idx_e)
            c = 1.0
        elseif (cnt % 2 == 0)
            c = 4.0
        else
            c = 2.0
        end
        cnt = cnt + 1
        M += c * vec[i]
    end

    M / 3.0
end

function plot_line_OU(dates::Array{DateTime, 1}, obs_series, model_series,
    ycol::Symbol, output_dir::String)

    ENV["GKSwstype"] = "nul"

    line_path = output_dir * "OU_MA_$(string(ycol))"
    dates_h = dates .+ Dates.Hour(1)

    BG_COLOR = ColorTypes.RGB(255/255, 255/255, 255/255)
    LN_COLOR = ColorTypes.RGB(67/255, 75/255, 86/255)
    MK_COLOR = ColorTypes.RGB(67/255, 75/255, 86/255)
    LN01_COLOR = ColorTypes.RGB(202/255,0/255,32/255)
    LN02_COLOR = ColorTypes.RGB(5/255,113/255,176/255)
    FL01_COLOR = ColorTypes.RGB(239/255, 138/255, 98/255)
    FL02_COLOR = ColorTypes.RGB(103/255, 169/255, 207/255)

    cor = Statistics.cor(obs_series, model_series)
    pl = Plots.plot(dates, obs_series,
        line=:solid, linewidth=5, label="OBS",
        size = (2560, 1080),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=LN01_COLOR,
        title="OBS & Stochastic, $(ycol) Corr :  $(cor)",
        xlabel="date", ylabel=string(ycol), fillalpha=0.5, legend=:best)

    Plots.plot!(dates, model_series,
        line=:solid, linewidth=2, color=LN02_COLOR, fillalpha=0.5, label="Stochastic")

    Plots.png(pl, line_path * ".png")

    nothing
end
