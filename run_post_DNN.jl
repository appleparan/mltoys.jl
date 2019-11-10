using Dates, TimeZones
using DelimitedFiles
using CSV, DataFrames
using Flux, Flux.Tracker
using MicroLogging
using ProgressMeter

using Mise

function mkpath_resdir(res_dir::String, output_size::Integer) 
    for i = 1:output_size
        i_pad = lpad(i, 2, '0')
        Base.Filesystem.mkpath(res_dir * "$(i_pad)/")
    end
end

function post_station(input_path::String, model_path::String, res_dir::String,
    sample_size::Integer, output_size::Integer,
    test_fdate::ZonedDateTime, test_tdate::ZonedDateTime,
    eltype::DataType)
    
    stn_codes = [111121, 111123, 111131, 111141, 111142,
        111151, 111152, 111161, 111171, 111181,
        111191, 111201, 111212, 111221, 111231,
        111241, 111251, 111261, 111262, 111273,
        111274, 111281, 111291, 111301, 111311]
    stn_names = ["중구", "종로구", "용산구", "광진구", "성동구",
        "중랑구", "동대문구", "성북구", "도봉구", "은평구",
        "서대문구", "마포구", "강서구", "구로구", "영등포구",
        "동작구", "관악구", "강남구", "서초구", "송파구",
        "강동구", "금천구", "강북구", "양천구", "노원구"]

    stn_stats_df = DataFrame(
        code = Int64[],
        name = String[],
        lat = Real[],
        lon = Real[],
        colname = String[],
        RMSE = Real[],
        MAE = Real[],
        MSPE = Real[],
        MAPE = Real[])
    
    #stn_codes = [ 111123, 111142, 111161, 111273]
    #stn_names = ["종로구", "성동구", "성북구", "송파구"]

    @info "    Test with other station"
    p = Progress(length(stn_codes), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    stations = (; zip(Symbol.(stn_names), stn_codes)...)
    df = load_data_DNN("/input/jongro_seoul.csv", stations)
    df = filter_raw_data(df, test_fdate, test_tdate)

    for (stn_code, stn_name) in Base.Iterators.zip(stn_codes, stn_names)
        stn_df = filter_station_DNN(df, stn_code)

        row_PM10 = test_station(model_path, stn_df, :PM10, stn_code, stn_name,
            sample_size, output_size, res_dir,
            "PM10_" * string(stn_code) * "_$(stn_name)", test_fdate, test_tdate,
            eltype, false)
        row_PM25 = test_station(model_path, stn_df, :PM25, stn_code, stn_name,
            sample_size, output_size, res_dir,
            "PM25_" * string(stn_code) * "_$(stn_name)", test_fdate, test_tdate,
            eltype, false)

        push!(stn_stats_df, row_PM10)
        push!(stn_stats_df, row_PM25)
        ProgressMeter.next!(p);
    end
    
    CSV.write(res_dir * "station_stats.csv", stn_stats_df)
end

function post_feature(input_path::String, model_path::String, res_dir::String,
    sample_size::Integer, output_size::Integer,
    test_fdate::ZonedDateTime, test_tdate::ZonedDateTime,
    eltype::DataType)

    fea_stats_df = DataFrame(
        code = Int64[],
        name = String[],
        lat = Real[],
        lon = Real[],
        colname = String[],
        RMSE = Real[],
        MAE = Real[],
        MSPE = Real[],
        MAPE = Real[],
        rm_fea = String[])
    
    stn_codes = [111123]
    stn_names = ["jongro"]
    stn_code = 111123
    stn_name = "jongro"

    stations = (; zip(Symbol.(stn_names), stn_codes)...)
    stn_df = load_data_DNN("/input/jongro_seoul.csv", stations)
    stn_df = filter_raw_data(stn_df, test_fdate, test_tdate)
    rm_features = [[:SO2], [:CO], [:O3], [:NO2],[:PM10], [:PM25],  [:temp], [:u, :v], [:pres], [:humid], [:prep, :snow]]
    rm_features_str = ["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp", "u_v", "pres", "humid", "prep_snow"]

    #rm_features = [[:PM10], [:PM25]]
    #rm_features_str = ["PM10", "PM25"]
    
    @info "    Test by removing some features"
    p = Progress(length(rm_features), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    norm_prefix = "norm_"
    _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]
    stn_df2 = copy(stn_df)
    #zscore!(stn_df2, features, _norm_feas)
    min_max_scaling!(stn_df2, features, _norm_feas)

    for (idx, r_fea) in enumerate(rm_features)
        r_fea_str = join(string.(r_fea))

        rmf_df = copy(stn_df)
        for r_f in r_fea
            rmf_df[:, r_f] .= 0.0
        end

        features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
        norm_prefix = "norm_"
        _norm_feas = [Symbol(eval(norm_prefix * String(f))) for f in features]
        #zscore!(rmf_df, features, _norm_feas)
        min_max_scaling!(rmf_df, features, _norm_feas)

        for r_f in r_fea
            norm_r_fea = Symbol(norm_prefix, string(r_f))
            # replace NaN to zero at r_f column (necessary when testing removing some features)
            rmf_df[findall(x -> isnan(x), rmf_df[:, norm_r_fea]), norm_r_fea] .= 0.0
        end

        row_PM10 = test_station(model_path, rmf_df, stn_df2, :PM10, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM10_rm_$(r_fea_str)", test_fdate, test_tdate,
            eltype, true)
        row_PM25 = test_station(model_path, rmf_df, stn_df2, :PM25, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM25_rm_$(r_fea_str)", test_fdate, test_tdate,
            eltype, true)

        push!(row_PM10, rm_features_str[idx])
        push!(row_PM25, rm_features_str[idx])

        push!(fea_stats_df, row_PM10)
        push!(fea_stats_df, row_PM25)

        ProgressMeter.next!(p);
    end

    row_PM10 = test_station(model_path, stn_df, :PM10, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM10_rm_FULL", test_fdate, test_tdate, false)
    row_PM25 = test_station(model_path, stn_df, :PM25, stn_code, stn_name,
        sample_size, output_size, res_dir, "PM25_rm_FULL", test_fdate, test_tdate, false)

    push!(row_PM10, "FULL")
    push!(row_PM25, "FULL")
    push!(fea_stats_df, row_PM10)
    push!(fea_stats_df, row_PM25)
    CSV.write(res_dir * "feature_stats.csv", fea_stats_df)

end

function post_forecast(input_path::String, model_path::String, res_dir::String,
    sample_size::Integer, output_size::Integer,
    test_fdate::ZonedDateTime, test_tdate::ZonedDateTime,
    eltype::DataType)
    
    stn_codes = [111121, 111123, 111131, 111141, 111142,
        111151, 111152, 111161, 111171, 111181,
        111191, 111201, 111212, 111221, 111231,
        111241, 111251, 111261, 111262, 111273,
        111274, 111281, 111291, 111301, 111311]
    stn_names = ["중구", "종로구", "용산구", "광진구", "성동구",
        "중랑구", "동대문구", "성북구", "도봉구", "은평구",
        "서대문구", "마포구", "강서구", "구로구", "영등포구",
        "동작구", "관악구", "강남구", "서초구", "송파구",
        "강동구", "금천구", "강북구", "양천구", "노원구"]

    fore_stats_df = DataFrame(
        code = Int64[],
        name = String[],
        lat = Real[],
        lon = Real[],
        colname = String[],
        fore_all = Real[],
        fore_high = Real[])
    
    @info "    Test with other station"
    p = Progress(length(stn_codes), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    stations = (; zip(Symbol.(stn_names), stn_codes)...)
    df = load_data_DNN("/input/jongro_seoul.csv", stations)
    df = filter_raw_data(df, test_fdate, test_tdate)

    for (stn_code, stn_name) in Base.Iterators.zip(stn_codes, stn_names)
        stn_df = filter_station_DNN(df, stn_code)

        row_PM10 = test_classification(model_path, stn_df, :PM10, stn_code, stn_name,
            sample_size, output_size, res_dir,
            "PM10_" * string(stn_code) * "_$(stn_name)", test_fdate, test_tdate,
            eltype, false)
        row_PM25 = test_classification(model_path, stn_df, :PM25, stn_code, stn_name,
            sample_size, output_size, res_dir,
            "PM25_" * string(stn_code) * "_$(stn_name)", test_fdate, test_tdate,
            eltype, false)

        push!(fore_stats_df, row_PM10)
        push!(fore_stats_df, row_PM25)
        ProgressMeter.next!(p);
    end
    
    CSV.write(res_dir * "forecast_stats.csv", fore_stats_df)
end

function run()
    input_path = "/input/input.csv"
    model_path = "/mnt/"

    sample_size = 48
    output_size = 24

    # For GPU
    eltype::DataType = Float64

    test_fdate = ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul")
    test_tdate = ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul")

    
    @info "Postprocessing per station"
    res_dir = "/mnt/post/station/"
    Base.Filesystem.mkpath(res_dir)
    mkpath_resdir(res_dir, output_size)
    post_station(input_path, model_path, res_dir,
        sample_size, output_size, test_fdate, test_tdate, eltype)
    
    #=
    @info "Postprocessing per feature removing"
    res_dir = "/mnt/post/feature/"
    Base.Filesystem.mkpath(res_dir)
    mkpath_resdir(res_dir, output_size)
    post_feature(input_path, model_path, res_dir,
        sample_size, output_size, test_fdate, test_tdate, eltype)

    @info "Postprocessing for forecasting"
    res_dir = "/mnt/post/forecast/"
    Base.Filesystem.mkpath(res_dir)
    mkpath_resdir(res_dir, output_size)
    post_forecast(input_path, model_path, res_dir,
        sample_size, output_size, test_fdate, test_tdate, eltype)
    =#
end

run()
