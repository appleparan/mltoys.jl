#const BG_COLOR = RGB(248/255, 247/255, 247/255)
# BG: 243 250 255 - LN: 67 75 86
#const BG_COLOR = RGB(243/255, 250/255, 255/255)
const BG_COLOR = RGB(255/255, 255/255, 255/255)
const LN_COLOR = RGB(67/255, 75/255, 86/255)
const MK_COLOR = RGB(67/255, 75/255, 86/255)
const LN01_COLOR = RGB(202/255,0/255,32/255)
const LN02_COLOR = RGB(5/255,113/255,176/255)
const FL01_COLOR = RGB(239/255, 138/255, 98/255)
const FL02_COLOR = RGB(103/255, 169/255, 207/255)

function plot_totaldata(df::DataFrame, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    hist_path = output_dir * String(ycol) * "_total_hist.png"
    plot_path = output_dir * String(ycol) * "_total_plot.png"

    # plot histogram
    ht = Plots.histogram(float.(df[!, ycol]), title="Histogram of " * String(ycol),
        size = (2560, 1080),
        xlabel=String(ycol), ylabel="#", bins=200, legend=false,
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, fillcolor = FL01_COLOR, fillalpha=0.2)
    png(ht, hist_path)

    # plot in dates
    dates = DateTime.(df[!, :date])
    pl = Plots.plot(dates, float.(df[!, ycol]),
        size = (2560, 1080),
        title=String(ycol) * " in dates", xlabel="date", ylabel=String(ycol),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linecolor = LN01_COLOR, legend=false)
    png(pl, plot_path)
end

function plot_pcorr(_df::DataFrame, feas::AbstractArray, label_feas::AbstractArray, output_dir::String)
    ENV["GKSwstype"] = "100"

    crpl_path = output_dir * "pearson_corr"
    crcsv_path = output_dir * "pearson_corr.csv"

    dfM = convert(Matrix, _df[:, feas])

    dfm_cor = Statistics.cor(dfM)

    ann = []
    for i in 1:length(feas)
    for j in 1:length(feas)
        _ann = (i - 0.5, j - 0.5, Plots.text(Formatting.format(dfm_cor[i, j], precision=2), 18, :white))
        push!(ann, _ann)
    end
    end

    crpl = Plots.heatmap(string.(label_feas), string.(label_feas), float.(dfm_cor),
        size = (2560, 1080),
        clim = (-1.0, 1.0), c=:blues, legend=true, annotations = ann,
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        title="CORR", background_color = BG_COLOR)

    Plots.png(crpl, crpl_path * ".csv")
    Plots.svg(crpl, crpl_path * ".svg")

    df_cor = DataFrame(dfm_cor)
    names!(df_cor, Symbol.(label_feas))
    CSV.write(crcsv_path, df_cor, writeheader = true)
end

function plot_corr(df_corr::DataFrame, output_size::Integer, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    plot_path = output_dir * "$(output_prefix)_corr_hourly"

    pl = Plots.plot(float.(df_corr[!, :hour]), float.(df_corr[!, :corr]),
        size = (2560, 1080),
        title="Correlation on hourly prediction", xlabel="hour", ylabel="corr",
        line=:solid, linewidth=5, label="OBS",
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linecolor = LN01_COLOR, legend=false)

    png(pl, plot_path)

    nothing
end

function plot_corr_input(corr_arr::Array{F, 1}, title::String, output_dir::String, output_prefix::String) where F<:AbstractFloat
    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix)_corr.png"

    pl = Plots.plot(float.(corr_arr),
        size = (2560, 1080),
        line=:solid, linewidth=5,
        ylim=(min(0.0, minimum(corr_arr)), 1.0),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color= :black,
        title = title,
        xlabel = "lag", ylabel = "corr", legend=false)
    plot!([72], color= :black, linewidth=5, line=:dash, seriestype="vline")

    Plots.png(pl, line_path)

    nothing
end

function plot_corr_OU(lag_arr::Array{I, 1}, corr_arr::Array{F, 1},
    title::String, output_dir::String, output_prefix::String) where {F<:AbstractFloat, I<:Integer}
    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix).png"

    pl = Plots.plot(float.(lag_arr), float.(corr_arr),
        size = (2560, 1080),
        line=:solid, linewidth=5,
        ylim=(min(0.0, minimum(corr_arr)), 1.0),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=30PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color= :black,
        title = title,
        xlabel = "time", ylabel = "corr", legend=false)
    # plot 72 hour line to check correlation is zero at 72 hour.
    plot!([72], color= :black, linewidth=5, line=:dash, seriestype="vline")

    Plots.png(pl, line_path)

    nothing
end

function plot_corr_OU(corr_arr::Array{F, 1},
    title::String, output_dir::String, output_prefix::String) where {F<:AbstractFloat, I<:Integer}
    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix).png"

    pl = Plots.plot(float.(corr_arr),
        size = (2560, 1080),
        line=:solid, linewidth=5,
        ylim=(min(0.0, minimum(corr_arr)), 1.0),
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=30PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color= :black,
        title = title,
        xlabel = "time", ylabel = "corr", legend=false)
    # plot 72 hour line to check correlation is zero at 72 hour.
    plot!([72], color= :black, linewidth=5, line=:dash, seriestype="vline")

    Plots.png(pl, line_path)

    nothing
end


function plot_DNN_scatter(dnn_table::Array{IndexedTable, 1}, ycol::Symbol,
    output_size::Integer, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        sc_path = output_dir * "$(i_pad)/" * "$(output_prefix)_scatter_$(i_pad)h.png"
        lim = max(maximum(JuliaDB.select(dnn_table[i], :y)), 
            maximum(JuliaDB.select(dnn_table[i], :ŷ)))

        sc = Plots.scatter(float.(JuliaDB.select(dnn_table[i], :y)), float.(JuliaDB.select(dnn_table[i], :ŷ)),
            size = (1080, 1080),
            xlim = (0, lim), ylim = (0, lim), legend=false,
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            title="DNN/OBS ($(i_pad)h)", xlabel="OBS", ylabel="DNN",
            background_color = BG_COLOR, markercolor = MK_COLOR)

        # diagonal line (corr = 1)
        Plots.plot!(collect(0:0.1:lim), collect(0:0.1:lim), 
            xlim = (0, lim), ylim = (0, lim), legend=false,
            background_color = BG_COLOR, linecolor = LN_COLOR)
        Plots.png(sc, sc_path)

        sc_csvpath = output_dir * "$(i_pad)/" * "$(output_prefix)_scatter_$(i_pad)h.csv"
        df = DataFrame(obs = float.(JuliaDB.select(dnn_table[i], :y)),
            model = float.(JuliaDB.select(dnn_table[i], :ŷ)))
        CSV.write(sc_csvpath, df, writeheader = true)
    end

    nothing
end

function plot_DNN_histogram(dnn_table::Array{IndexedTable, 1}, ycol::Symbol,
    output_size::Integer, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        hs_OBS_path = output_dir * "$(i_pad)/" * "$(output_prefix)_hist(obs)_$(i_pad)h.png"
        hs_DNN_path = output_dir * "$(i_pad)/" * "$(output_prefix)_hist(dnn)_$(i_pad)h.png"

        ht = Plots.histogram(float.(JuliaDB.select(dnn_table[i], :y)), label="OBS",
            size = (2560, 1080),
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            title="Histogram of OBS ($(i_pad)h)", ylabel="#",
            background_color = BG_COLOR, fillalpha=0.5)
        Plots.png(ht, hs_OBS_path)

        ht = Plots.histogram(float.(JuliaDB.select(dnn_table[i], :ŷ)), label="DNN",
            size = (2560, 1080),
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            title="Histogram of DNN ($(i_pad)h)", ylabel="#",
            background_color = BG_COLOR, fillalpha=0.5)
        Plots.png(ht, hs_DNN_path)
    end

    nothing
end

"""
    plot_DNN_lineplot(dates, dnn_table, ycol, output_size, output_dir, output_prefix)

Plot prediction by dates (full length). 
"""
function plot_DNN_lineplot(dates::Array{DateTime, 1}, dnn_table::Array{IndexedTable, 1}, ycol::Symbol,
    output_size::Integer, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        line_path = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(i_pad)h.png"
        line_csvpath = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(i_pad)h.csv"
        dates_h = dates .+ Dates.Hour(i)

        pl = Plots.plot(dates_h, float.(JuliaDB.select(dnn_table[i], :y)),
            size = (2560, 1080),
            ylim = (0.0,
                max(maximum(JuliaDB.select(dnn_table[i], :y)), maximum(JuliaDB.select(dnn_table[i], :ŷ)))),
            line=:solid, linewidth=5, label="OBS",
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            background_color = BG_COLOR, color=LN01_COLOR,
            title=String(ycol) * " by dates ($(i_pad)h)",
            xlabel="date", ylabel=String(ycol), legend=:best)

        pl = Plots.plot!(dates_h, float.(JuliaDB.select(dnn_table[i], :ŷ)),
            line=:solid, linewidth=5, color=LN02_COLOR, label="DNN")
        Plots.png(pl, line_path)

        line_csvpath = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(i_pad)h.csv"
        df = DataFrame(dates = dates,
            obs = float.(JuliaDB.select(dnn_table[i], :y)),
            model = float.(JuliaDB.select(dnn_table[i], :ŷ)))
        CSV.write(line_csvpath, df, writeheader = true)
    end

    nothing
end

"""
    plot_DNN_lineplot(dates, dnn_table, s_date, f_date, ycol, output_size, output_dir, output_prefix)

Plot prediction by dates (given date range). 
"""
function plot_DNN_lineplot(dates::Array{DateTime, 1}, dnn_table::Array{IndexedTable, 1},
    s_date::DateTime, f_date::DateTime, ycol::Symbol,
    output_size::Integer, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        line_path = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(i_pad)h.png"
        dates_h = dates .+ Dates.Hour(i)

        pl = Plots.plot(dates_h, float.(JuliaDB.select(dnn_table[i], :y)),
            size = (2560, 1080),
            ylim = (0.0,
                max(maximum(JuliaDB.select(dnn_table[i], :y)), maximum(JuliaDB.select(dnn_table[i], :ŷ)))),
            line=:solid, linewidth=5, label="OBS",
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            background_color = BG_COLOR, color=LN01_COLOR,
            title=String(ycol) * " by dates ($(i_pad)h)",
            xlabel="date", ylabel=String(ycol), legend=:best)

        pl = Plots.plot!(dates_h, float.(JuliaDB.select(dnn_table[i], :ŷ)),
            line=:solid, linewidth=5, color=LN02_COLOR, label="DNN")
        Plots.png(pl, line_path)

        line_csvpath = output_dir * "$(i_pad)/" * "$(output_prefix)_line_$(i_pad)h.png"
        df = DataFrame(dates = dates,
            obs = float.(JuliaDB.select(dnn_table[i], :y)),
            model = float.(JuliaDB.select(dnn_table[i], :ŷ)))
        CSV.write(line_csvpath, df, writeheader = true)
    end
    
    nothing
end

function plot_evaluation(df::DataFrame, ycol::Symbol, eval_syms, output_dir::String)
    ENV["GKSwstype"] = "100"

    last_epoch = df[end, :epoch]

    for eval_sym in eval_syms
        gr(size = (1920, 1080))

        plot_path = output_dir * String(ycol) * "_eval_$(String(eval_sym)).png"

        pl = Plots.plot(float.(df[:, :epoch]), float.(df[:, eval_sym]),
            size = (2560, 1080),
            color=:black, linewidth=6,
            guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15PlotMeasures.px,
            guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
            background_color = BG_COLOR,
            title="$(String(eval_sym)) " * String(ycol),
            xlabel="epoch", ylabel="$(String(eval_sym))", legend=false)
        annotate!([(last_epoch, df[last_epoch, eval_sym], text("Value: " *  string(df[last_epoch, eval_sym]), 18, :black, :right))])
        Plots.png(pl, plot_path)
    end

    eval_path = output_dir * String(ycol) * "_evals.csv"
    CSV.write(eval_path, df, writeheader = true)
    nothing
end
