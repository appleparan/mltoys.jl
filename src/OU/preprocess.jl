
function crosscorr_input(df::DataFrame, ycol::Symbol, features::Array{Symbol, 1}, lags)
    output_dir = "/mnt/crosscorr/"
    Base.Filesystem.mkpath(output_dir)

    for fea in features
        cor = crosscor(df[:, ycol], df[:, fea], lags)
        plot_corr_OU(lags, cor, "$(string(ycol)) & $(string(fea))", output_dir, "crosscor_$(string(ycol))_$(string(fea))")
    end
end

function autocorr_input(df::DataFrame, ycol::Symbol, lags)
    output_dir = "/mnt/autocorr/"
    Base.Filesystem.mkpath(output_dir)

    cor = autocor(df[:, ycol], lags)
    plot_corr_OU(cor, "$(string(ycol))", output_dir, "autocor_$(string(ycol))")

end
