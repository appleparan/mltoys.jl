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
function evolve_OU_season(test_set, ycol::Symbol, scaled_ycol::Symbol,
    acf_dres1::AbstractArray, season_table::AbstractNDSparse,
    μσs::AbstractNDSparse, _eltype::DataType,
    input_size::Integer, output_size::Integer,
    test_dates::Array{ZonedDateTime,1},
    output_prefix::String) where I <: Integer

    acf_df = DataFrame(time = collect(0:(length(acf_dres1) - 1)), corr = acf_dres1)
    org_intT = compute_inttscale(acf_df[:, :time], acf_dres1)
    # not to log negative value when curve fitting
    posIntT = max(1, Int(round(org_intT)) - 1)
    org_fit = exp_fit(acf_df[1:posIntT, :time], acf_df[1:posIntT, :corr])
    #@info "Coef a and b in a * exp(b * x) for Annual Residual.. to 0:$(posIntT)", fit[1], fit[2]

    fitted_fit = exp_fit(acf_df[1:posIntT, :time], acf_df[1:posIntT, :corr])
    fit_x = acf_df[1:posIntT, :time]
    fit_y = fitted_fit[1] .* exp.(fit_x .* fitted_fit[2])
    intT = compute_inttscale(fit_x, fit_y)

    # ycol's mean / std
    μ_ycol = μσs[String(ycol) * "_res", "μ"].value
    σ_ycol = μσs[String(ycol) * "_res", "σ"].value

    # Time scale
    # T(hour), dt = 1
    T = intT
    Θ = 1.0 / T
    # becuase it's zscored, original μ and σ is 0 and 1.
    μ = 0.0
    σ = sqrt(1.0 * 2.0 / T)
    dt = 1

    # Vasicek Model
    # https://en.wikipedia.org/wiki/Vasicek_model

    # Check
    # https://github.com/SciML/DiffEqNoiseProcess.jl/blob/master/src/ornstein_uhlenbeck.jl
    ou_res_df = DataFrame(date = DateTime[], offset = _eltype[], y = _eltype[], yhat = _eltype[])

    for df_tset in test_set
        output_dates = DateTime(df_tset[1, :date], Local):Dates.Hour(1):(DateTime(df_tset[1, :date], Local) + Dates.Hour(output_size))

        X₀ = df_tset[1, scaled_ycol]
        OU = OrnsteinUhlenbeckProcess(Θ, μ, σ, 0, X₀)
        prob = NoiseProblem(OU, (0.0, output_size))
        Xₜ = solve(prob; dt=1.0)

        org_y = df_tset[1:output_size + 1, ycol]
        org_Xt = compose_seasonality(collect(output_dates[1:output_size + 1]),
            unzscore(Xₜ[1:output_size + 1], μ_ycol, σ_ycol),
            season_table)

        ou_res_df_tmp = DataFrame(date = output_dates, offset = 0:output_size, y = org_y, yhat = org_Xt)
        append!(ou_res_df, ou_res_df_tmp)
    end

    ou_res_df
end
