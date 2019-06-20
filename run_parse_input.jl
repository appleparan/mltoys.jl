using TimeZones

using MLToys

function read_data()
    input_dir = "/input"
    obs_path = joinpath(input_dir, "station", "aerosol_observatory_2017.xlsx")
    aes_dir = "aerosol"
    wea_dir = joinpath("weather", "seoul")
    start_date = ZonedDateTime(2008, 1, 1, 0, 0, tz"Asia/Seoul")
    end_date = ZonedDateTime(2018, 12, 31, 24, 0, tz"Asia/Seoul")

    join_data(input_dir, obs_path, aes_dir, wea_dir, start_date, end_date)
end

read_data()
