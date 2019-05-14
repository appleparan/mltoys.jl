module MLToys

# to use Plots in headless system
# https://github.com/JuliaPlots/Plots.jl/issues/1076#issuecomment-327509819
include("input.jl")
include("utils.jl")
include("plots.jl")
include("evaluation.jl")

include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")

# input
export join_data, 
# utils
        mean_and_std_cols, hampel!, zscore!, exclude_elem, split_df, window_df,
        split_sizes3, split_sizes2, create_chunks, create_idxs,
        getHoursLater, getX, getY, make_pairs, make_minibatch, remove_missing_pairs!,
# evaluation
        RSME, RSR,
# jongro01_DNN
        filter_jongro, read_jongro, train_all_DNN,
# plot
        plot_totaldata,
        get_prediction_table,
        plot_corr,
        plot_DNN_scatter,
        plot_DNN_histogram,
        plot_DNN_lineplot        
end