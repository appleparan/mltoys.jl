module MLToys

include("utils.jl")
include("evaluation.jl")
include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")

# utils
export  standardize!,
        exclude_elem,
        train_test_size_split,
        train_test_idxs_split,
        getHoursLater,
        getX,
        getY,
        make_minibatch,
        split_df,
        window_df,
        perm_idx,
        perm_df,
# evaluation
        RSME,
        RSR,
# jongro01_DNN
        filter_jongro,
        read_jongro,
        train_all


end