using Dates, TimeZones
using CSV, DataFrames
using Flux
using MicroLogging
using ProgressMeter

using MLToys

function post_station(input_path::String, model_path::String, res_dir::String,
    sample_size::Integer, output_size::Integer, test_sdate::ZonedDateTime, test_fdate::ZonedDateTime)
    
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
        colname = String[],
        RMSE = Real[],
        RSR = Real[],
        PBIAS = Real[],
        NSE = Real[],
        IOA = Real[],
        corr_1h = Real[],
        corr_24h = Real[])
    
    @info "    Test with other station"
    p = Progress(length(stn_codes), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    for (stn_code, stn_name) in Base.Iterators.zip(stn_codes, stn_names)        
        stn_df = read_station(input_path, stn_code)

        row_PM10 = test_station(model_path, stn_df, :PM10, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM10_" * string(stn_code) * "_$(stn_name)", test_sdate, test_fdate)
        row_PM25 = test_station(model_path, stn_df, :PM25, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM25_" * string(stn_code) * "_$(stn_name)", test_sdate, test_fdate)

        push!(stn_stats_df, row_PM10)
        push!(stn_stats_df, row_PM25)
        ProgressMeter.next!(p);
    end
    
    CSV.write(res_dir * "station_stats.csv", stn_stats_df)
end

function post_feature(input_path::String, model_path::String, res_dir::String,
    sample_size::Integer, output_size::Integer, test_sdate::ZonedDateTime, test_fdate::ZonedDateTime)
    fea_stats_df = DataFrame(
        code = Int64[],
        name = String[],
        colname = String[],
        RMSE = Real[],
        RSR = Real[],
        PBIAS = Real[],
        NSE = Real[],
        IOA = Real[],
        corr_1h = Real[],
        corr_24h = Real[])
    
    stn_code = 111123
    stn_name = "jongro"

    stn_df = read_station(input_path, stn_code)
    rm_features = [[:SO2], [:CO], [:O3], [:NO2], [:temp], [:u, :v], [:pres], [:humid], [:prep, :snow]]
    
    @info "    Test with removed featuers"
    p = Progress(length(rm_features), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)

    for r_fea in rm_features
        r_fea_str = join(string.(r_fea))

        rmf_df = copy(stn_df)
        for r_f in r_fea
            rmf_df[:, r_f] = 0.0
        end

        row_PM10 = test_station(model_path, rmf_df, :PM10, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM10_rm_$(r_fea_str)", test_sdate, test_fdate)
        row_PM25 = test_station(model_path, rmf_df, :PM25, stn_code, stn_name,
            sample_size, output_size, res_dir, "PM25_rm_$(r_fea_str)", test_sdate, test_fdate)
        
        push!(fea_stats_df, row_PM10)
        push!(fea_stats_df, row_PM25)
        ProgressMeter.next!(p);
    end

    CSV.write(res_dir * "feature_stats.csv", fea_stats_df)
end

function run()
    input_path = "/input/input.csv"
    model_path = "/mnt/"

    sample_size = 72
    output_size = 24

    test_sdate = ZonedDateTime(2018, 1, 1, 1, tz"Asia/Seoul")
    test_fdate = ZonedDateTime(2018, 12, 31, 23, tz"Asia/Seoul")

    res_dir = "/mnt/post/station/"
    Base.Filesystem.mkpath(res_dir)
    post_station(input_path, model_path, res_dir,
        sample_size, output_size, test_sdate, test_fdate)
    res_dir = "/mnt/post/feature/"
    Base.Filesystem.mkpath(res_dir)
    post_feature(input_path, model_path, res_dir,
        sample_size, output_size, test_sdate, test_fdate)
end

run()
