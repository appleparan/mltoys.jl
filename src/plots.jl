#const BG_COLOR = RGB(248/255, 247/255, 247/255)
const BG_COLOR = RGB(255/255, 255/255, 255/255)
const LN_COLOR = RGB(56/255, 44/255, 80/255)
const MK_COLOR = RGB(109/255, 117/255, 126/255)

function plot_totaldata(df::DataFrame, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    hist_path = output_dir * String(ycol) * "_total_hist.png"
    plot_path = output_dir * String(ycol) * "_total_plot.png"

    # plot histogram
    gr(size=(1920, 1080))
    ht = Plots.histogram(df[ycol], title="Histogram of " * String(ycol),
        xlabel=String(ycol), ylabel="#", bins=200,
        margin=15px, legend=false,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, 
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, fillcolor=[:red], fillalpha=0.2)
    png(ht, hist_path)

    # plot in dates
    dates = DateTime.(df[:date])
    gr(size=(1920, 1080))
    pl = Plots.plot(dates, df[ycol],
        title=String(ycol) * " in dates", xlabel="date", ylabel=String(ycol),
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12, 
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        margin=15px, background_color = BG_COLOR, linecolor = LN_COLOR, legend=false)
    png(pl, plot_path)
end

function plot_corr(_df::DataFrame, feas, label_feas, output_dir::String)
    ENV["GKSwstype"] = "100"

    crpl_path = output_dir * "pearson_corr.png"
    crcsv_path = output_dir * "pearson_corr.csv"

    dfM = convert(Matrix, _df[:, feas])

    dfm_cor = Statistics.cor(dfM)

    crpl = Plots.heatmap(string.(label_feas), string.(label_feas), dfm_cor,
        clim = (-1.0, 1.0), c=:blues, legend=true,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        title="CORR",
        margin=15px, background_color = BG_COLOR)

    Plots.png(crpl, crpl_path)

    df_cor = DataFrame(dfm_cor)
    names!(df_cor, Symbol.(label_feas))
    CSV.write(crcsv_path, df_cor, writeheader = true)
end

function get_prediction_table(df, dataset, model, ycol::Symbol, μ::AbstractFloat, σ::AbstractFloat, output_dir::String)
    table_01h_path = output_dir * String(ycol) * "_01h_table.csv"
    table_24h_path = output_dir * String(ycol) * "_24h_table.csv"

    dnn_01h_table = table((y = [], ŷ = [],))
    dnn_24h_table = table((y = [], ŷ = [],))

    for (x, y) in dataset
        ŷ = model(x |> gpu)

        cpu_y = y |> cpu
        cpu_ŷ = ŷ |> cpu

        org_y = cpu_y .* σ .+ μ
        org_ŷ = Flux.Tracker.data(cpu_ŷ) .* σ .+ μ
        
        tmp_table = table((y = [org_y[1]], ŷ = [org_ŷ[1]],))
        dnn_01h_table = merge(dnn_01h_table, tmp_table)

        tmp_table = table((y = [org_y[24]], ŷ = [org_ŷ[24]],))
        dnn_24h_table = merge(dnn_24h_table, tmp_table)
    end

    JuliaDB.save(dnn_01h_table, table_01h_path)
    JuliaDB.save(dnn_24h_table, table_24h_path)

    dnn_01h_table, dnn_24h_table
end

function plot_DNN_scatter(dnn_01h_table, dnn_24h_table, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    sc_01h_path = output_dir * String(ycol) * "_01h_scatter.png"
    sc_24h_path = output_dir * String(ycol) * "_24h_scatter.png"

    lim = max(maximum(JuliaDB.select(dnn_01h_table, :y)), maximum(JuliaDB.select(dnn_01h_table, :ŷ)))
    gr(size=(1080, 1080))
    sc = Plots.scatter(JuliaDB.select(dnn_01h_table, :y), JuliaDB.select(dnn_01h_table, :ŷ), 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        title="OBS/DNN", xlabel="Observation", ylabel="DNN",
        margin=15px, background_color = BG_COLOR, markercolor = MK_COLOR)
    Plots.plot!(0:0.1:lim, 0:0.1:lim, 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        background_color = BG_COLOR, linecolor = LN_COLOR)
    Plots.png(sc, sc_01h_path)

    gr(size=(1080, 1080))
    sc = Plots.scatter(JuliaDB.select(dnn_24h_table, :y), JuliaDB.select(dnn_24h_table, :ŷ), 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        title="OBS/DNN", xlabel="Observation", ylabel="DNN",
        margin=15px, background_color = BG_COLOR, markercolor = MK_COLOR)
    Plots.plot!(0:0.1:lim, 0:0.1:lim, 
        xlim = (0, lim), ylim = (0, lim), legend=false,
        background_color = BG_COLOR, linecolor = LN_COLOR)
    Plots.png(sc, sc_24h_path)   
end

function plot_DNN_histogram(dnn_01h_table, dnn_24h_table, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    hs_01h_path = output_dir * String(ycol) * "_01h_hist.png"
    hs_24h_path = output_dir * String(ycol) * "_24h_hist.png"

    gr(size=(2560, 1080))
    ht = Plots.histogram(JuliaDB.select(dnn_01h_table, :y), label="obs", 
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        margin=15px, title="Histogram of data", ylabel="#",
        background_color = BG_COLOR, fillalpha=0.5)
    ht = Plots.histogram!(JuliaDB.select(dnn_01h_table, :ŷ), label="model")
    Plots.png(ht, hs_01h_path)

    gr(size=(2560, 1080))
    ht = Plots.histogram(JuliaDB.select(dnn_24h_table, :y), label=["obs", "model"],
        guidefontsize = 12, titlefontsize = 18, tickfontsize = 12, legendfontsize = 12,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        margin=15px, title="Histogram of data", ylabel="#",
        background_color = BG_COLOR, fillalpha=0.5)
    ht = Plots.histogram!(JuliaDB.select(dnn_24h_table, :ŷ), label="model")
    Plots.png(ht, hs_24h_path)
end

function plot_DNN_lineplot(dates, dnn_01h_table, dnn_24h_table, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    line_01h_path = output_dir * String(ycol) * "_01h_line.png"
    line_24h_path = output_dir * String(ycol) * "_24h_line.png"
    dates_01h = dates .+ Dates.Hour(1)
    dates_24h = dates .+ Dates.Hour(24)
    len_model = length(JuliaDB.select(dnn_01h_table, :y))

    # plot in dates
    gr(size = (2560, 1080))
    pl = Plots.plot(dates_01h[1:len_model], JuliaDB.select(dnn_01h_table, :y),
        ylim = (0.0, maximum(JuliaDB.select(dnn_01h_table, :y))),
        line=:dash, color=:black, label="obs.",
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR,
        title=String(ycol) * " in dates (1h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    pl = Plots.plot!(dates_01h[1:len_model], JuliaDB.select(dnn_01h_table, :ŷ),
        ylim = (0.0, maximum(JuliaDB.select(dnn_01h_table, :ŷ))),
        line=:solid, color=:red, label="model")
    Plots.png(pl, line_01h_path)

    @info "Correlation in 01H results: ", Statistics.cor(JuliaDB.select(dnn_01h_table, :y), JuliaDB.select(dnn_01h_table, :ŷ))

    gr(size = (2560, 1080))
    pl = Plots.plot(dates_24h[1:len_model], JuliaDB.select(dnn_24h_table, :y),
        ylim = (0.0, maximum(JuliaDB.select(dnn_24h_table, :y))),
        line=:dash, color=:black, label="obs.",
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR,
        title=String(ycol) * " in dates (24h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    pl = Plots.plot!(dates_24h[1:len_model], JuliaDB.select(dnn_24h_table, :ŷ),
        ylim = (0.0, maximum(JuliaDB.select(dnn_01h_table, :ŷ))),
        line=:solid, color=:red, label="model")
    Plots.png(pl, line_24h_path)

    @info "Correlation in 24H results: ", Statistics.cor(JuliaDB.select(dnn_24h_table, :y), JuliaDB.select(dnn_24h_table, :ŷ))
end

function plot_DNN_lineplot(dates, dnn_01h_table, dnn_24h_table, s_date::DateTime, f_date::DateTime, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    fmt = @dateformat_str "yyyymmddHH"

    s_date_str = Dates.format(s_date, fmt)
    f_date_str = Dates.format(f_date, fmt)
    line_01h_path = output_dir * String(ycol) * "_" * s_date_str * "_" * f_date_str * "_01h_line.png"
    line_24h_path = output_dir * String(ycol) * "_" * s_date_str * "_" * f_date_str * "_24h_line.png"
    dates_01h = dates .+ Dates.Hour(1)
    dates_24h = dates .+ Dates.Hour(24)
    len_model = length(JuliaDB.select(dnn_01h_table, :y))

    # slice between s_date and e_date
    y_01h_vals = JuliaDB.select(dnn_01h_table, :y)
    ŷ_01h_vals = JuliaDB.select(dnn_01h_table, :ŷ)
    y_24h_vals = JuliaDB.select(dnn_24h_table, :y)
    ŷ_24h_vals = JuliaDB.select(dnn_24h_table, :ŷ)

    s_01h_idx = findfirst(isequal.(dates_01h, s_date))
    f_01h_idx = findfirst(isequal.(dates_01h, f_date))
    s_24h_idx = findfirst(isequal.(dates_24h, s_date))
    f_24h_idx = findfirst(isequal.(dates_24h, f_date))

    # plot in dates
    gr(size = (2560, 1080))
    pl = Plots.plot(dates_01h[s_01h_idx:f_01h_idx], y_01h_vals[s_01h_idx:f_01h_idx],
        ylim = (0.0, maximum(y_01h_vals[s_01h_idx:f_01h_idx])),
        line=:dash, color=:black, label="obs.",
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR,
        title=String(ycol) * " in dates (1h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    pl = Plots.plot!(dates_01h[s_01h_idx:f_01h_idx], ŷ_01h_vals[s_01h_idx:f_01h_idx],
        ylim = (0.0, maximum(ŷ_01h_vals[s_01h_idx:f_01h_idx])),
        line=:solid, color=:red, label="model")
    Plots.png(pl, line_01h_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(dates_24h[s_24h_idx:f_24h_idx], y_01h_vals[s_24h_idx:f_24h_idx],
        ylim = (0.0, maximum(y_24h_vals[s_24h_idx:f_24h_idx])),
        line=:dash, color=:black, label="obs.",
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR,
        title=String(ycol) * " in dates (24h)", 
        xlabel="date", ylabel=String(ycol), legend=true)
    pl = Plots.plot!(dates_24h[s_24h_idx:f_24h_idx], ŷ_01h_vals[s_24h_idx:f_24h_idx],
        ylim = (0.0, maximum(ŷ_24h_vals[s_24h_idx:f_24h_idx])),
        line=:solid, color=:red, label="model")
    Plots.png(pl, line_24h_path)
end

function plot_evaluation(df::DataFrame, ycol::Symbol, output_dir::String)
    ENV["GKSwstype"] = "100"
    rsme_path = output_dir * String(ycol) * "_eval_RSME.png"
    rsr_path = output_dir * String(ycol) * "_eval_RSR.png"
    nse_path = output_dir * String(ycol) * "_eval_NSE.png"
    pbias_path = output_dir * String(ycol) * "_eval_PBIAS.png"
    learn_rate_path = output_dir * String(ycol) * "_eval_learning_rate.png"
    loss_path = output_dir * String(ycol) * "_eval_loss.png"

    last_epoch = df[end, :epoch]

    gr(size = (2560, 1080))
    pl = Plots.plot(df[:, :epoch], df[:, :RSME],
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linewidth=4,
        title="RMSE of " * String(ycol),
        xlabel="epoch", ylabel="RSR", legend=false)
    annotate!([(last_epoch, df[last_epoch, :RSME], text("Value: " *  string(df[last_epoch, :RSME]), 18, :black, :right))])
    Plots.png(pl, rmse_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(df[:, :epoch], df[:, :RSR],
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linewidth=4,
        title="RSR of " * String(ycol),
        xlabel="epoch", ylabel="RSR", legend=false)
    annotate!([(last_epoch, df[last_epoch, :RSR], text("Value: " *  string(df[last_epoch, :RSR]), 18, :black, :right))])
    Plots.png(pl, rsr_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(df[:, :epoch], df[:, :NSE],
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linewidth=4,
        title="NSE of " * String(ycol),
        xlabel="epoch", ylabel="NSE", legend=false)
    annotate!([(last_epoch, df[last_epoch, :NSE], text("Value: " *  string(df[last_epoch, :NSE]), 18, :black, :right))])
    Plots.png(pl, nse_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(df[:, :epoch], df[:, :PBIAS],
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linewidth=4,
        title="PBIAS of " * String(ycol),
        xlabel="epoch", ylabel="PBIAS", legend=false)
    annotate!([(last_epoch, df[last_epoch, :PBIAS], text("Value: " *  string(df[last_epoch, :PBIAS]), 18, :black, :right))])
    Plots.png(pl, pbias_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(df[:, :epoch], df[:, :learn_rate],
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linewidth=4,
        title="LEARNING RATE of " * String(ycol),
        xlabel="epoch", ylabel="Learning Rate", yscale=:log10, legend=false)
    Plots.png(pl, learn_rate_path)

    gr(size = (2560, 1080))
    pl = Plots.plot(df[:, :epoch], df[:, :loss],
        guidefontsize = 18, titlefontsize = 24, tickfontsize = 18, legendfontsize = 18, margin=15px,
        guidefontcolor = LN_COLOR, titlefontcolor = LN_COLOR, tickfontcolor = LN_COLOR, legendfontcolor = LN_COLOR,
        background_color = BG_COLOR, linewidth=4,
        title="LOSS of " * String(ycol),
        xlabel="epoch", ylabel="Loss", legend=false)
    Plots.png(pl, loss_path)

    nothing
end