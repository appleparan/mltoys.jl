using TimeZones, CSV
using Mise

function read_data()
    input_dir = "/input"
    obs_path = joinpath(input_dir, "station", "aerosol_observatory_2017_aironly.xlsx")
    aes_dir = "aerosol"
    wea_dir = joinpath("weather", "seoul")
    start_date = ZonedDateTime(2008, 1, 1, 1, 0, tz"Asia/Seoul")
    end_date = ZonedDateTime(2020, 10, 31, 23, 0, tz"Asia/Seoul")

    join_data(input_dir, obs_path, aes_dir, wea_dir, start_date, end_date)
end

function read_raw_PM1025() 
    input_dir = "/input"
    aes_dir = "aerosol"
    start_date = ZonedDateTime(2008, 1, 1, 1, 0, tz"Asia/Seoul")
    end_date = ZonedDateTime(2020, 10, 31, 23, 0, tz"Asia/Seoul")

    stn_code = 111123
    df_aes = parse_aerosols(aes_dir, input_dir)
    df_aes_jongro = filter_station(df_aes, stn_code)

    # check path
    mkpath(input_dir)
    filename = joinpath(input_dir, "input_raw_PM1025.csv")
    CSV.write(filename, df_aes_jongro)
end

read_data()
#read_raw_PM1025()

