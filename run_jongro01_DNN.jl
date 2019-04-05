using MLToys

function run_model()
    df = read_jongro("/input/jongro_single.csv")
    features = ["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp", "u", "v", "pres", "humid"]
    norm_prefix="norm_"
    norm_feas = [norm_prefix * f for f in features]
    
    PM10_mean, PM10_std = mean_and_std(df[:PM10])
    standardize_df!(df, features, prefix)
    
    sample_size = 72
    output_size = 24
    
    # split by segment
    #splitted_df, mb_idxs = split_df(size(df, 1), sample_size)
    # split by stream
    splitted_df, mb_idxs = window_df(size(df, 1), sample_size)
    total_size, train_size, valid_size, test_size = train_test_size_split(length(mb_idxs))
    train_idx, valid_idx, test_idx = train_test_idxs_split(total_size, train_size, valid_size, test_size)
    train_all(df, features, mb_idxs, output_size)
    
end
