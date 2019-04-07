using MicroLogging
using StatsBase: mean_and_std

using MLToys

function run_model()
    df = read_jongro("/input/jongro_single.csv")
    features = [:SO2, :CO, :O3, :NO2, :PM10, :PM25, :temp, :u, :v, :pres, :humid]
    norm_prefix="norm_"
    norm_features = [Symbol(eval(norm_prefix * String(f))) for f in features]
    
    @info "Start preprocessing..."
    flush(stdout); flush(stderr)

    PM10_mean, PM10_std = mean_and_std(df[:PM10])
    PM25_mean, PM25_std = mean_and_std(df[:PM25])
    @info "PM10 mean and std ", PM10_mean, PM10_std
    @info "PM25 mean and std ", PM25_mean, PM25_std
    standardize!(df, features, norm_features)
    
    sample_size = 72
    output_size = 24
    epoch_size = 500
    
    # split into segment
    #splitted_df, mb_idxs = split_df(size(df, 1), sample_size)
    # split into stream
    mb_idxs = window_df(df, sample_size)
    total_size, train_size, valid_size, test_size = train_test_size_split(length(mb_idxs))
    train_idx, valid_idx, test_idx = train_test_idxs_split(total_size, train_size, valid_size, test_size)
    @info "Data preprocessing complete!"
    flush(stdout); flush(stderr)

    # to use zscroed data, use norm_features
    train_all(df, norm_features, mb_idxs, sample_size * length(features), output_size, epoch_size, train_idx, valid_idx, test_idx)
    
end

run_model()