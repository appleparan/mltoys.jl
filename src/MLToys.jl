module MLToys

# to use Plots in headless system
# https://github.com/JuliaPlots/Plots.jl/issues/1076#issuecomment-327509819
include("utils.jl")
include("plots.jl")
include("evaluation.jl")

include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")

# utils
export mean_and_std_cols, zscore!, exclude_elem, split_df, window_df,
        split_sizes, create_chunks, create_idxs,
        getHoursLater, getX, getY, make_pairs, make_minibatch,
# evaluation
        RSME, RSR,
# jongro01_DNN
        filter_jongro, read_jongro, train_all,
# plot
        plot_DNN,
        plot_DNN_toCSV,
        plot_totaldata,
        plot_initdata
end