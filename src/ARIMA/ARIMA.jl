function predict_ARIMA(test_df::DataFrame, ycol::Symbol, scaled_ycol::Symbol,
    acf_dres1::AbstractArray, pacf_dres1::AbstractArray, season_table::AbstractNDSparse, μσs::AbstractNDSparse,
    train_fdate::ZonedDateTime, train_tdate::ZonedDateTime, test_fdate::ZonedDateTime, test_tdate::ZonedDateTime,
    _eltype::DataType, input_size::Integer, output_size::Integer, output_dir::String, sim_name::Symbol) where I <: Integer

    DateTime(test_fdate, Local):Dates.Hour(1):DateTime(test_tdate, Local)
    acf_df = DataFrame(lags = collect(0:(length(acf_dres1) - 1)), acf = acf_dres1, pacf = pacf_dres1)
    org_intT = compute_inttscale(acf_df[:, :lags], acf_dres1)
    # not to log negative value when curve fitting
    posIntT = max(1, Int(round(org_intT)) - 1)
    org_fit = exp_fit(acf_df[1:posIntT, :lags], acf_df[1:posIntT, :acf])
    #@info "Coef a and b in a * exp(b * x) for Annual Residual.. to 0:$(posIntT)", fit[1], fit[2]

    fitted_fit = exp_fit(acf_df[1:posIntT, :lags], acf_df[1:posIntT, :acf])
    fit_x = acf_df[1:posIntT, :lags]
    fit_y = fitted_fit[1] .* exp.(fit_x .* fitted_fit[2])
    intT = compute_inttscale(fit_x, fit_y)

    # ycol's mean / std
    μ_ycol = μσs[String(ycol) * "_res", "μ"].value
    σ_ycol = μσs[String(ycol) * "_res", "σ"].value

    CSV.write(output_dir * "ARIMA_acf_$(String(sim_name)).csv", acf_df)

    # plot acf and pacf
    plot_ARIMA_acf(acf_df, output_dir, "ARIMA_$(string(ycol))", String(sim_name))

    # Vasicek Model
    # https://en.wikipedia.org/wiki/Vasicek_model

    # ARIMA with StateSpaceModels.jl
    # Section 8.3 in Brockwell and Davis 2002,
    # https://github.com/JuliaStats/TimeModels.jl/blob/master/src/ARIMA.jl
    arima_res_df = DataFrame(date = DateTime[], offset = _eltype[], y = _eltype[], yhat = _eltype[])

    p = Progress(size(test_df, 1), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    for (i, row) in enumerate(eachrow(test_df))
        ProgressMeter.next!(p)

        if i < input_size || row[:date] + Dates.Hour(output_size) > test_tdate
            continue
        end

        model = construct_model(reshape(test_df[(i - input_size + 1):i, scaled_ycol], input_size, 1))
        ss = statespace(model_raw)

        output_dates = DateTime.(test_df[(i + 1):(i + output_size), :date], Local)
        y = _eltype.(test_df[(i + 1):(i + output_size), scaled_ycol])

        _y_hat, dist_y_hat = StateSpaceModels.forecast(ss, output_size)
        y_hat = _eltype.(compose_seasonality(output_dates, unzscore(_y_hat, μ_ycol, σ_ycol), season_table))

        tmp_df = DataFrame(date = output_dates, offset = 0:output_size, y = y, yhat = y_hat)
        append!(arima_res_df, tmp_df)
    end
    @show first(arima_res_df, 5)
    arima_res_df
end

function construct_model(y, ARIMA_p = 1, ARIMA_d = 0, ARIMA_q = 1)
    # White noise
    normal = Normal(0, 1)

    ARIMA_r = max(ARIMA_p, ARIMA_q + 1)

    # n in matrix size
    n = length(y)
    # univariate so 1, p in matrix size
    #p = 1
    # dimension of state vector α
    #m = 1
    # dimension of state covariance matrix Q_t
    #r = 1

    # Initial coefficients
    # Auto Regressive
    ar = fill!(rand(ARIMA_p), NaN)
    # Moving Average
    ma = fill!(rand(ARIMA_q), NaN)

    # H and Q will be estimated
    #=
    H_ϵ = 1.0
    Q_η = 1.0
    η = zeros(1, 1)
    WN_ϵ = Normal(0, H_ϵ)
    WN_η = Normal(0, Q_η)
    # White Noise ~ N(0, Q_t)
    ϵ = rand(WN_ϵ, (1, 1))

    # H : white noise in observation equation (p x p)
    H = ones(p, p)
    # Q : white noise in state equation (r x r)
    Q = ones(r, r)
    =#

    # initial state vector
    a_1 = 0.0
    P_1 = 1.0
    #WN_α = Normal(a_1, Q_η)

    # State Space Represetation
    # Known : Z, T, R, H and Q
    # Observation equation
    # y_t = Z_t α_t + ϵ_t
    # y_t : observation vector (p x 1)

    # Output (measurement) matrix
    # Z = [Θ_{r-1} Θ_{r-2}  ... Θ_{0}], Brockwell eq. (8.3.7), (p x m x n)
    # Z = [1, 0, ....0] # eq. (3.19) in "Time Series Analysis by State Space Methods" (2012) by J. Durbin and S. J. Koopman.
    Z = hcat(vcat(1, zeros(ARIMA_r - 1)), zeros(ARIMA_r, n - 1))
    #Z = [1 zeros(max(0, ARIMA_r - ARIMA_q - 1)), reverse(ma), 1]'

    # State equation
    # α_{t+1} = T_t α_t + R_t η_t
    # α : state vector (m x 1)

    # T =
    # Φ_1   1   0   ... 0
    # Φ_2   0   1   ... 0
    # ...
    # Φ_r   0   0   ... 1
    # zeros(ARIMA_r - 1) diagm(ones(ARIMA_r - 1)) -> 0 and 1 part
    # zeros(max(0, ARIMA_r - ARIMA_p)), reverse(ar) -> Φ part
    # eq. (3.20) in "Time Series Analysis by State Space Methods" (2012) by J. Durbin and S. J. Koopman.
    T = hcat(vcat(zeros(max(0, ARIMA_r - ARIMA_p)), reverse(ar)), vcat(diagm(ones(ARIMA_r - 1)), zeros(ARIMA_r - 1)'))
    R = reshape(vcat(1, ma, zeros(max(0, ARIMA_r - ARIMA_q - 1))), ARIMA_r, 1)
    # R : selection matrix, usually identity matrix (m x r)

    H = zeros(1, 1)
    # Q will be esitimated
    Q = fill!(zeros(ARIMA_r), NaN)

    model_raw = StateSpaceModel(y, Z, T, R)

    model_raw
end