const BG_COLOR = RGB(255/255, 255/255, 255/255)
const LN_COLOR = RGB(67/255, 75/255, 86/255)
const MK_COLOR = RGB(67/255, 75/255, 86/255)
const LN01_COLOR = RGB(202/255,0/255,32/255)
const LN02_COLOR = RGB(5/255,113/255,176/255)
const FL01_COLOR = RGB(239/255, 138/255, 98/255)
const FL02_COLOR = RGB(103/255, 169/255, 207/255)
# Opencolor : https://yeun.github.io/open-color/
const GRAY4 = colorant"#CED4dA"
const GRAY7 = colorant"#495057"
const RED4 = colorant"#FF8787"
const RED7 = colorant"#F03E3E"
const PINK4 = colorant"#F783AC"
const GRAPE4 = colorant"#DA77F2"
const VIOLET4 = colorant"#9775FA"
const INDIGO4 = colorant"#748FFC"
const BLUE4 = colorant"#4DABF7"
const CYAN4 = colorant"#3BC9DB"
const TEAL4 = colorant"#38D9A9"
const GREEN4 = colorant"#69DB7C"
const LIME4 = colorant"#A9E34b"
const YELLOW4 = colorant"#FFD43B"
const ORANGE4 = colorant"#FFA94D"

function plot_anal_lineplot(dates::Array{DateTime, 1}, arr::AbstractArray, ycol::Symbol,
    order::Integer, lag::Integer,
    output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix)_anal_timediff_o$(order)l$(lag).png"

    pl = Plots.plot(dates, arr,
        size = (2560, 1080),
        line=:solid, linewidth=5,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=15PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title="Time difference with lag: $(lag), order: $(order)",
        xlabel="date", ylabel=String(ycol), legend=false)
    Plots.png(pl, line_path)

    line_csvpath = output_dir * "$(output_prefix)_anal_timediff_o$(order)l$(lag).csv"
    df = DataFrame(date = dates,
        arr = arr)
    CSV.write(line_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_autocor(arr::AbstractArray, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    line_path = output_dir * "$(output_prefix)_autocor.png"

    pl = Plots.plot(0:length(arr)-1, arr,
        size = (1920, 1080),
        line=:solid, linewidth=5,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title=title_string,
        xlabel="time", ylabel=String(ycol), legend=false)
    # hline for zero
    Plots.plot!([0.0], seriestype=:hline, line=:dash, linewidth=5, color=:black)
    Plots.png(pl, line_path)

    line_csvpath = output_dir * "$(output_prefix)_autocor.csv"
    df = DataFrame(arr = arr)
    CSV.write(line_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_correlogram(arr::AbstractArray, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    bar_path = output_dir * "$(output_prefix)_correlogram_bar.png"
    pl = Plots.bar(0:length(arr)-1, arr,
        size = (1920, 1080),
        bar_width = 0.5,
        markershape = :circle, markersize = 5, markercolor = LN_COLOR,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title=title_string,
        xlabel="time", ylabel=String(ycol), legend=false)
    # hline for zero
    Plots.plot!([0.0], seriestype=:hline, line=:dash, linewidth=5, color=:black)
    Plots.png(pl, bar_path)

    line_path = output_dir * "$(output_prefix)_correlogram_line.png"
    pl = Plots.plot(0:length(arr)-1, arr,
        size = (1920, 1080),
        bar_width = 0.5,
        markershape = :circle, markersize = 5, markercolor = LN_COLOR,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        title=title_string,
        xlabel="time", ylabel=String(ycol), legend=false)
    # hline for zero
    Plots.plot!([0.0], seriestype=:hline, line=:dash, linewidth=5, color=:black)
    Plots.png(pl, line_path)

    bar_csvpath = output_dir * "$(output_prefix)_correlogram.csv"
    df = DataFrame(arr = arr)
    CSV.write(bar_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_violin(df::DataFrame, ycol::Symbol, tdir::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    violin_path = output_dir * "$(output_prefix)_violin.png"

    pl = StatsPlots.@df df violin(df[!, tdir], df[!, ycol],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(0.2, :blue, stroke(0)), legend=false)
    Plots.png(pl, violin_path)

    violin_csvpath = output_dir * "$(output_prefix)_violin.csv"
    df = DataFrame(tdir => df[!, tdir], :arr => df[!, ycol])
    CSV.write(violin_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_violin(df::DataFrame, ycol::Symbol, tdir::Symbol, means::AbstractArray,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    violin_path = output_dir * "$(output_prefix)_violin_$(string(tdir))ly.png"

    pl = StatsPlots.@df df violin(df[!, tdir], df[!, ycol],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(0.2, :blue, stroke(0)), legend=false)
    if tdir == :hour
        x_means = 0:23
    elseif tdir == :week
        x_means = 1:53
    elseif tdir == :month
        x_means = 1:12
    elseif tdir == :quarter
        x_means = 1:4
    else
        error("Time directive must be between :hour, :week, :month, :quarter")
    end
    Plots.plot!(x_means, means, color=:red, marker=(2.0, :red, stroke(1)), legend=:false)
    Plots.png(pl, violin_path)

    violin_csvpath = output_dir * "$(output_prefix)_violin_$(string(tdir))ly.csv"
    df = DataFrame(tdir => df[!, tdir], :arr => df[!, ycol])
    CSV.write(violin_csvpath, df, writeheader = true)

    nothing
end
#=
function plot_anal_pdf(k::UnivariateKDE, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    pdf_path = output_dir * "$(output_prefix).png"

    pl = Plots.plot(k.x, k.density,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        ylabel=String(ycol),
        marker=(1, :blue, stroke(0)), legend=false)
    Plots.png(pl, pdf_path)

    pdf_df = DataFrame(x = k.x, density = k.density)
    pdf_csvpath = output_dir * "$(output_prefix).csv"
    CSV.write(pdf_csvpath, pdf_df, writeheader = true)

    nothing
end
=#
function plot_anal_pdf(x::AbstractVector, y::AbstractVector, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    pdf_path = output_dir * "$(output_prefix).png"

    pl = Plots.plot(x, y,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        ylabel=String(ycol),
        marker=(1, :blue, stroke(0)), legend=false)
    Plots.png(pl, pdf_path)

    pdf_df = DataFrame(x = x, density = y)
    pdf_csvpath = output_dir * "$(output_prefix).csv"
    CSV.write(pdf_csvpath, pdf_df, writeheader = true)

    nothing
end


function plot_anal_periodic_mean(df::DataFrame, ycol::Symbol, tdir::Symbol, means::AbstractArray,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    time_mean_path = output_dir * "$(output_prefix)_$(string(tdir))ly.png"

    if tdir == :hour
        x_means = 0:23
    elseif tdir == :week
        x_means = 1:53
    elseif tdir == :month
        x_means = 1:12
    elseif tdir == :quarter
        x_means = 1:4
    else
        error("Time directive must be between :hour, :week, :month, :quarter")
    end

    pl = Plots.plot(x_means, means,
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(1.0, :blue, stroke(0)), legend=false)
    Plots.png(pl, time_mean_path)

    time_mean_csvpath = output_dir * "$(output_prefix)_$(string(tdir))ly.csv"
    df = DataFrame(tdir => x_means, :mean => means)
    CSV.write(time_mean_csvpath, df, writeheader = true)

    nothing
end

function plot_anal_periodic_fluc(df::DataFrame, tdir::Symbol, ycol::Symbol,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    time_fluc_path = output_dir * "$(output_prefix)_time_fluc_$(string(tdir))ly.png"
    pl = StatsPlots.@df df violin(df[!, tdir], df[!, Symbol(tdir, "_fluc")],
        size = (1920, 1080),
        title = title_string,
        guidefontsize = 32, titlefontsize = 48, tickfontsize = 24, legendfontsize = 24, margin=36PlotMeasures.px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=:black,
        xlabel=String(tdir), ylabel=String(ycol),
        marker=(0.2, :blue, stroke(0)), legend=false)
    Plots.png(pl, time_fluc_path)

    time_fluc_csvpath = output_dir * "$(output_prefix)_time_fluc_$(string(tdir))ly.csv"
    df = DataFrame(tdir => df[!, tdir], :fluc => df[!, Symbol(tdir, "_", :fluc)])
    CSV.write(time_fluc_csvpath, df, writeheader = true)

    nothing
end

function plot_seasonality(raw_data, df, ycol::Symbol, _year::Integer, f_date::DateTime,
    title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    season_path1 = output_dir * "$(output_prefix)_1.png"

    year_dr = DateTime(_year, 1, 1, 0):Dates.Hour(1):DateTime(_year, 12, 31, 23)
    day_dr = DateTime(_year, 1, 1, 0):Dates.Hour(1):DateTime(_year, 1, 1, 23)
    l1 = @layout [a ; b ; c ; d]

    b_offset = Hour(DateTime(_year, 1, 1, 0) - f_date).value

    # l1 : raw + daily season + (annual_season + smoothed_season) + annual_residual_smoothed
    p1 = Plots.plot(year_dr, df[!, :raw][(b_offset+1):(b_offset+length(year_dr))],
        title = "Raw Data", titlefontsize = 54, titlefontcolor = LN_COLOR, 
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel="data", legend=false)
    # Daily Seasonality
    p2 = Plots.plot(collect(0:23), df[!, :day_sea][(b_offset+1):(b_offset+length(day_dr))],
        title = "Daily Seasonality", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32, linewidth=5,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel="daily seasonality", legend=false)
    # Annual Seasonality
    p3 = Plots.plot(year_dr, df[!, :year_sea][(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Seasonality", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY4,
        ylabel="annual seasonality", legend=false)
    Plots.plot!(year_dr, df[!, :year_sea_s][(b_offset+1):(b_offset+length(year_dr))],
        guidefontsize = 32, tickfontsize = 32, color=RED7, linewidth=7,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, legend=false)

    # Annual Residual
    p4 = Plots.plot(year_dr,  df[!, :year_res][(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Residual", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel="annual residual", legend=false)

    pl = plot(p1, p2, p3, p4,
        size = (1920, 4320),
        margin=36PlotMeasures.px,
        layout = l1)
    Plots.png(pl, season_path1)

    #season_csvpath = output_dir * "$(output_prefix).csv"
    season_path2 = output_dir * "$(output_prefix)_2.png"
    # l2 : daily residual + annual_residual_smothed + annaul_residual_raw
    l2 = @layout [a ; b ; c ; d]
    max_p1p2 = max(maximum(df[!, :year_sea][(b_offset+1):(b_offset+length(year_dr))]), maximum(df[!, :year_sea_s][(b_offset+1):(b_offset+length(day_dr))]))
    min_p1p2 = min(minimum(df[!, :year_sea][(b_offset+1):(b_offset+length(year_dr))]), minimum(df[!, :year_sea_s][(b_offset+1):(b_offset+length(day_dr))]))
    p1 = Plots.plot(year_dr, df[!, :year_sea][(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Seasonality", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32, ylim=(min_p1p2, max_p1p2),
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel="annual seasonality", legend=false)
    p2 = Plots.plot(year_dr, df[!, :year_sea_s][(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Seasonality (smoothed)", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32, linewidth=5, ylim=(min_p1p2, max_p1p2),
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7, 
        ylabel="annual seasonality (smoothed)", legend=false)
    max_p3p4 = max(maximum(df[!, :year_res][(b_offset+1):(b_offset+length(year_dr))]), maximum(df[!, :year_res_s][(b_offset+1):(b_offset+length(day_dr))]))
    min_p3p4 = min(minimum(df[!, :year_res][(b_offset+1):(b_offset+length(year_dr))]), minimum(df[!, :year_res_s][(b_offset+1):(b_offset+length(day_dr))]))
    p3 = Plots.plot(year_dr, df[!, :year_res][(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Residual (Raw)", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32, ylim=(min_p3p4, max_p3p4),
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel="annual residual", legend=false)
    p4 = Plots.plot(year_dr, df[!, :year_res_s][(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Residual (Smooth)", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32, ylim=(min_p3p4, max_p3p4),
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel="annual residual", legend=false)
        
    pl = plot(p1, p2, p3, p4,
        size = (1920, 4320),
        margin=36PlotMeasures.px,
        layout = l2)
    Plots.png(pl, season_path2)
    nothing
end

function plot_seasonality(raw_data, year_sea1, year_sea2, day_sea1, day_res2, ycol::Symbol, b_year::Integer, e_year::Integer, b_date::DateTime,
    _ylabels::Array{String, 1}, title_string::String, output_dir::String, output_prefix::String)

    ENV["GKSwstype"] = "100"

    season_path = output_dir * "$(output_prefix).png"
    #748ffc

    year_dr = DateTime(b_year, 1, 1, 0):Dates.Hour(1):DateTime(e_year, 12, 31, 23)
    day_dr = DateTime(b_year, 1, 1, 0):Dates.Hour(1):DateTime(b_year, 1, 1, 23)
    l = @layout [a ; b ; c ; d]
    b_offset = Hour(DateTime(b_year, 1, 1, 0) - b_date).value
    #Raw
    p1 = Plots.plot(year_dr, raw_data[(b_offset+1):(b_offset+length(year_dr))],
        title = "Raw Data", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel=_ylabels[1], legend=false)
    # Annual Seasonality
    p2 = Plots.plot(year_dr, year_sea1[(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Seasonality", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY4,
        ylabel=_ylabels[2], legend=false)
    # smoothed
    Plots.plot!(year_dr, year_sea2[(b_offset+1):(b_offset+length(year_dr))],
        color=RED7, linewidth=5, legend=false)
    # Daily Seasonality
    p3 = Plots.plot(hour.(day_dr), day_sea1[(b_offset+1):(b_offset+length(day_dr))],
        title = "Daily Seasonality", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel=_ylabels[3], legend=false)
    # Annual Residual
    p4 = Plots.plot(year_dr, day_res2[(b_offset+1):(b_offset+length(year_dr))],
        title = "Annual Residual", titlefontsize = 54, titlefontcolor = LN_COLOR,
        guidefontsize = 32, tickfontsize = 32,
        guidefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, color=GRAY7,
        ylabel=_ylabels[4], legend=false)

    pl = plot(p1, p2, p3, p4,
        size = (1920, 4320),
        margin=36PlotMeasures.px,
        layout = l)
    Plots.png(pl, season_path)

    season_csvpath = output_dir * "$(output_prefix).csv"

    nothing
end
