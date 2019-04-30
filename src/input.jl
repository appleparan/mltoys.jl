using Base.Filesystem

using CSV
using DataValues
using DataFrames
using Dates, TimeZones
using ExcelReaders
using FileIO
using Glob
using MicroLogging
using ProgressMeter

# https://github.com/vtjnash/Glob.jl/issues/11
rglob(pat, topdir) = Base.Iterators.flatten(map(d -> glob(pat, d[1]), walkdir(topdir)))

function parse_obsxlsx(obs_path::String, input_dir::String)
    ws = readxlsheet(string(obs_path), "2017년")

    re_opened = r"신규"
    re_closed = r"폐쇄"
    re_moved = r"이전"
    re_hangul = r"[가-힣]+"
    re_opened_date = r"(\d+)\.\W*(\d+)\.?"
    #re_closed_date = r"(\d+)\.\W*(\d+)\.?\W*(\d+)\."
    re_closed_date = r"(\d+)\.\W*(\d+)\.?(\W*(\d+)\.?)*"
    KST = tz"Asia/Seoul"
    df = DataFrame(stationCode=[], stationName=[], oDate=[], cDate=[], lon=[], lat=[])
    cnt = 0

    #=
        row[1] = state (시, 도)
        row[2] = city (도시)
        row[3] = stationCode (측정소코드)
        row[4] = stationName (측정소명)
        row[5] = address (주소)
        row[6] = langitude (경도)
        row[7] = latitude (위도)
        row[8] = remakrs (비고)
    =#

    for i in 1:size(ws, 1)
        # just pick old date
        opened_date = ZonedDateTime(1970, 1, 1, tz"Asia/Seoul")
        # close value to maximum value in Int64
        closed_date = ZonedDateTime(2037, 12, 31, tz"Asia/Seoul")

        row = ws[i, :]
        # prev_row = ws[i - 1, :]
        # if stationCode is blank, skip, so no move
        if isa(row[3], DataValue) && isna(row[3])
            continue
        end

        # if there is not stationCode (i.e. wrong column), skip loop
        try 
            if isa(row[3], String)
                station_code = parse(Int, row[3])
            elseif isa(row[3], AbstractFloat)
                station_code = Int(row[3])
            end
        catch e
            if isa(e, ArgumentError)
                continue
            end
        end

        # check lat / long 
        if isa(row[6], Real) && isa(row[7], Real)
            station_code = 0
            station_name = ""
            long_X = float(row[6])
            lat_Y = float(row[7])

            cnt += 1

            # parse station code and name, doesn't support moved case
            if isa(row[3], String)
                station_code = parse(Int, row[3])
            elseif isa(row[3], AbstractFloat)
                station_code = Int(row[3])
            end

            if occursin(re_hangul, string(row[4]))
                station_name = string(row[4])
            end

            # parse date of station opened
            # check there is a stationCode and "신규"(opened new station) in remarks 
            if occursin(re_opened, string(row[8]))
                # parse date (year & month only)
                m = match(re_opened_date, string(row[8]))
                opened_year = parse(Int, m.captures[1])
                opened_month = parse(Int, m.captures[2])

                opened_date = ZonedDateTime(opened_year, opened_month, 1, tz"Asia/Seoul")
            end

            # parse date of station closed
            if occursin(re_closed, string(row[8]))
                m = match(re_closed_date, string(row[8]))
                closed_year = parse(Int, m.captures[1])
                closed_month = parse(Int, m.captures[2])
                closed_day = m.captures[4]
                if closed_day == nothing
                    closed_day = day(Dates.lastdayofmonth(DateTime(closed_year, closed_month)))
                else
                    closed_day = parse(Int, m.captures[4])
                end

                closed_date = ZonedDateTime(closed_year, closed_month, closed_day, tz"Asia/Seoul")
            end

            # parse date of station moved
            #=
            if hasvalue(row[3]) == true && occursin(re_moved, string(row[8]))
                if row[3] != 
                    station_code = int(row[3])
                    station_name = string(row[4])
                else 
                    station_code = int(prev_row[3])
                    station_name = string(prev_row[4])
                end

                m = match(re_closed_date, string(row[8])
                moved_year = m.captures[1]
                moved_month = m.captures[2]
                opened_date = ZonedDateTime(int(moved_year), int(moved_month), 1, 0, 0, tz"Asia/Seoul")

                # compute closed date of modified one
                closed_date_prev = opened_date - Minute(1)

                # find exising one and modify closed_date
                # TODO: not sure how to do if there are multiple movement of station
                prev = @from i in stn_df begin
                    @where i.statioCode == station_code
                    @select i
                    @collect DataFrame
                end
                
                station_code = Msrstn_list[prev_msrstn_idx][stationCode]
                station_name = Msrstn_list[prev_msrstn_idx][stationName]

                Msrstn_list[prev_Msrstn_idx][closedDate] = closed_date_ex.strftime(strftime_fmt)
            end
            =#
            
            # check type
            @assert isa(station_code, Int)
            @assert isa(long_X, Real)
            @assert isa(lat_Y, Real)
            push!(df, (station_code, station_name, opened_date, closed_date, long_X, lat_Y))
        end
    end

    # check path
    mkpath(input_dir)
    filename = joinpath(input_dir, "msrstn_korea.csv")
    CSV.write(filename, df)

    @show first(df, 5)

    df
end

function parse_aerosols(aes_dir::String, input_dir::String)
    re_aes_fn = r"([0-9]+)년\W*([0-9]+)분기.csv"
    date_str = "yyyymmddHH"

    aes_paths = joinpath(input_dir, aes_dir)
    aes_globs = rglob([re_aes_fn], aes_paths)

    # check there is a input file
    @assert isempty(aes_globs) == false
    
    df = DataFrame(stationCode = [], date = [],
                   SO2 = [], CO = [], O3 = [], NO2 = [],
                   PM10 = [], PM25 = [])

    p = Progress(length(collect(aes_globs)), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    for _aes in aes_globs
        df_raw = CSV.read(_aes)

        rename!(df_raw, [:지역 => :region, :측정소코드 => :stationCode, :측정소명 => :stationName, :측정일시 => :date,
                   :주소 => :addr])
        dropmissing!(df_raw, :date)

        # convert to ZonedDateTime to parse time correctly (i.e. 20181231T24:00 and 20190101T00:00)
        dates = [ZonedDateTime(DateTime(string(d), date_str), tz"Asia/Seoul") for d in df_raw[:date]]

        df_tmp = DataFrame(stationCode = df_raw[:stationCode],
            date = dates,
            SO2 = df_raw[:SO2],
            CO = df_raw[:CO],
            O3 = df_raw[:O3],
            NO2 = df_raw[:NO2],
            PM10 = df_raw[:PM10],
            PM25 = df_raw[:PM25])

        df = vcat(df, df_tmp)

        ProgressMeter.next!(p; showvalues = [(:filename, _aes)])
        flush(stdout); flush(stderr)
    end

    df
end

function parse_weathers(wea_dir::String, input_dir::String, wea_stn_code::Integer)
    re_wea_fn = Regex("SURFACE_ASOS_$(string(wea_stn_code))_HR_([0-9]+)_([0-9]+)_([0-9]+).csv")
    @show re_wea_fn
    date_str = "yyyy-mm-dd HH:MM"

    wea_paths = joinpath(input_dir, wea_dir)
    wea_globs = rglob([re_wea_fn], wea_paths)
    @assert isempty(wea_globs) == false

    df = DataFrame(date = [], temp = [], u = [], v = [],
        pres = [], humid = [], prep = [], snow = [])

    p = Progress(length(collect(wea_globs)), dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=40, color=:yellow)
    for _wea in wea_globs
        df_raw = CSV.read(_wea)

        rename!(df_raw, [:일시 => :date, 
            Symbol("기온(°C)") => :temp,
            Symbol("강수량(mm)") => :prep,
            Symbol("풍속(m/s)") => :wind_vel,
            Symbol("풍향(16방위)") => :wind_dir,
            Symbol("습도(%)") => :humid,
            Symbol("현지기압(hPa)") => :pres,
            Symbol("적설(cm)") => :snow])
        dropmissing!(df_raw, :date)

        dates = [ZonedDateTime(DateTime(string(d), date_str), tz"Asia/Seoul") for d in df_raw[:date]]
        
        # missings to zero except temp, humid, and pres (pressure)
        df_raw[:wind_vel] = coalesce.(df_raw[:wind_vel], 0.0)
        df_raw[:wind_dir] = coalesce.(df_raw[:wind_dir], 0.0)
        df_raw[:prep] = coalesce.(df_raw[:prep], 0.0)
        df_raw[:snow] = coalesce.(df_raw[:snow], 0.0)

        # http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
        _u = [w[1] * Base.Math.cos(Base.Math.deg2rad(w[2] - 270)) for w in zip(df_raw[:wind_vel], df_raw[:wind_dir])]
        _v = [w[1] * Base.Math.sin(Base.Math.deg2rad(w[2] - 270)) for w in zip(df_raw[:wind_vel], df_raw[:wind_dir])]

        df_tmp = DataFrame(
            date = dates,
            temp = df_raw[:temp],
            u = _u,
            v = _v,
            pres = df_raw[:pres],
            humid = df_raw[:humid],
            prep = df_raw[:prep],
            snow = df_raw[:snow])

        df = vcat(df, df_tmp)

        ProgressMeter.next!(p; showvalues = [(:filename, _wea)])
        flush(stdout); flush(stderr)
    end

    df
end

"""
    join_data(input_dir, obs_path, aes_dir, wea_dir, start_date, end_date)
join dataframe from 3 input
input_dir: input directory  (i.e. "/input")
obs_path : station xlsx path (full path)
aes_dir : subdirectory name of aerosol data (i.e. "aerosol")
wea_dir : subdirectory name of weather data (i.e. joinpath("weather", "seoul"))
"""
function join_data(input_dir::String, obs_path::String, aes_dir::String, wea_dir::String, start_date::ZonedDateTime, end_date::ZonedDateTime)
    seoul_stn_code = 108

    @info "Parsing Station dataset..."
    flush(stdout); flush(stderr)
    df_obs = parse_obsxlsx(obs_path, input_dir)
    @info "Parsing Weather dataset..."
    flush(stdout); flush(stderr)
    df_wea = parse_weathers(wea_dir, input_dir, seoul_stn_code)
    @info "Parsing Aerosol dataset..."
    flush(stdout); flush(stderr)
    df_aes = parse_aerosols(aes_dir, input_dir)

    # filter in date range
    df_obs = df_obs[(df_obs.oDate .<= start_date) .& (df_obs.cDate .>= end_date), :]
    df_aes = df_aes[(df_aes.date .>= start_date) .& (df_aes.date .<= end_date), :]
    df_wea = df_wea[(df_wea.date .>= start_date) .& (df_wea.date .<= end_date), :]

    df1 = join(df_obs, df_aes, on = :stationCode)
    df2 = join(df1, df_wea, on = :date, kind = :inner)

    df = df2[:, [:stationCode, :date, :lat, :lon,
                   :SO2, :CO, :O3, :NO2, :PM10, :PM25,
                   :temp, :u, :v, :pres, :humid, :prep, :snow]]
    
    # sort by stationCode and date
    sort!(df, (:stationCode, :date))
    @show first(df_obs, 5)
    @show first(df_aes, 5)
    @show first(df1, 5)
    @show first(df2, 5)
    @show first(df, 5)

    # check path
    mkpath(input_dir)
    filename = joinpath(input_dir, "input.csv")
    CSV.write(filename, df)

    df
end
