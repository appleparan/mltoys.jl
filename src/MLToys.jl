module MLToys

include("utils.jl")
include("jongro01_DNN/preprocess.jl")
include("jongro01_DNN/model.jl")

# utils
export  standardize!,
        exclude_elem,
        getHoursLater,
        make_minibatch,
        perm_idx,
        perm_df,
        split_df,
# jongro01_DNN
        filter_jongro,
        read_jongro,
        train

end