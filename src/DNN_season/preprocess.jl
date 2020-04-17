function filter_station_DNN(df, stn_code)
    stn_df = @from i in df begin
        @where i.stationCode == stn_code
        @select i
        @collect DataFrame
    end

    stn_df
end

function process_raw_data_DNN!(stn_df::DataFrame)
    flush(stdout); flush(stderr)
    cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25,
        :temp, :pres, :u, :v, :humid, :prep, :snow]
    airkorea_cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25]
    weather_cols = [:temp, :u, :v, :pres, :humid, :prep, :snow]

    for col in cols
        stn_df[!, col] = Missings.coalesce.(stn_df[!, col], 0.0)
    end
    
    stn_df[!, :date] = ZonedDateTime.(stn_df[!, :date])

    # no and staitonCode must not have missing value
    @assert size(collect(skipmissing(stn_df[!, :stationCode])), 1) == size(stn_df, 1)

    DataFrames.dropmissing!(stn_df, [:stationCode])
    DataFrames.allowmissing!(stn_df, cols)
    for col in [:prep, :snow]
        stn_df[!, col] = Missings.coalesce.(stn_df[!, col], 0.0)
    end

    for col in airkorea_cols
        replace!(stn_df[!, col], -999 => missing)
    end

    # check remaining missing values
    for col in names(stn_df)
        @assert size(stn_df, 1) == size(collect(skipmissing(stn_df[!, col])), 1)
    end
    dropmissing!(stn_df, cols, disallowmissing=true)

    flush(stdout); flush(stderr)
end

"""
    load_data(pre_input_path, stns, src_input_path::String = "/input/input.csv")

Path -> DataFrame

there is a two scenario
1. I have pre-filtered data
    * load pre-filtered data
    * process data
2. I don't have pre-filtered data
    * load all data from source
    * select data by station
    * process data
    * save data
"""
function load_data_DNN(pre_input_path::String, stns::NamedTuple,
    src_input_path::String = "/input/input.csv") where I<:Integer
    
    # Check preprocessed path exists
    if Base.Filesystem.ispath(pre_input_path)
        filtered_df = CSV.read(pre_input_path, copycols=true)
        process_raw_data_DNN!(filtered_df)
        df = filtered_df
    else
        # if there is no preprocessed input, read source and create
        dfs = DataFrame[]

        # if not, read from source input path
        # note that at 2012/11 one hour have missing, needed to add empty data manually
        raw_df = CSV.read(src_input_path, copycols=true)
        for (name, code) in pairs(stns)
            @info "Preprocessing $name($code) data..."
            stn_df = filter_station_DNN(raw_df, code)

            process_raw_data_DNN!(stn_df)

            push!(dfs, stn_df)
        end

        # concatenate dataframe
        _df = vcat(dfs...)
        sort!(_df, [:stationCode, :date]);

        CSV.write(pre_input_path, _df)

        df = _df
    end

    df
end

"""
    filter_jongro(df)
Filter DataFrame by jongro station code (111123)    
"""
function filter_jongro(df)
    jongro_stn = 111123
    jongro_df = @from i in df begin
        @where i.stationCode == jongro_stn
        @select i
        @collect DataFrame
    end

    cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    for col in cols
        jongro_df[col] = Missings.coalesce.(jongro_df[col], 0.0)
    end

    return jongro_df
end

function save_jongro_df(input_path = "/input/input.csv")
    df = CSV.read(input_path, copycols=true)
    sort!(df, (:date, :stationCode))
    @show first(df, 5)
    j_df = filter_jongro(df)

    CSV.write("/input/jongro_single.csv", j_df)
end

function read_jongro(input_path="/input/jongro_single.csv")
    if Base.Filesystem.isfile(input_path) == false
        save_jongro_df()
    end

    df = DataFrame(CSV.read(input_path, copycols=true))
    
    @info "Start preprocessing..."
    flush(stdout); flush(stderr)
    if eltype(df[!, :date]) != ZonedDateTime
        df[!, :date] = ZonedDateTime.(df[!, :date])
    end

    # no and staitonCode must not have missing value
    @assert size(collect(skipmissing(df[!, :stationCode])), 1) == size(df, 1)

    DataFrames.dropmissing!(df, [:stationCode])
    cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid, :prep, :snow]
    airkorea_cols = [:SO2, :CO, :O3, :NO2, :PM10, :PM25]
    weather_cols = [:temp, :u, :v, :pres, :humid]

    plot_totaldata(df, :PM25, "/mnt/raw_")
    plot_totaldata(df, :PM10, "/mnt/raw_")
    flush(stdout); flush(stderr)

    DataFrames.allowmissing!(df, cols)
    for col in [:prep, :snow]
        df[!, col] = Missings.coalesce.(df[!, col], 0.0)
    end

    for col in airkorea_cols
        replace!(df[!, col], -999 => missing)
    end

    # check remaining missing values
    for col in names(df)
        @assert size(df, 1) == size(collect(skipmissing(df[!, col])), 1)
    end
    dropmissing!(df, cols, disallowmissing=true)

    @show first(df, 5)
    flush(stdout); flush(stderr)

    df
end

function perm_df(df, permed_idx, col, labels)
    X = df[permed_idx, labels]
    Y = df[permed_idx, col]

    X, Y
end
