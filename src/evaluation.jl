using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data

function RSME(dataset, model)
    # RSME
    acc = 0.0

    for (x, y) in dataset
        ŷ = model(x)
        @assert size(ŷ) == size(y)

        acc += sqrt(sum(abs2.(ŷ .- y)) / length(y))
    end

    # acc is incomplete TrackedReal, convert it to pure Real type
    Flux.Tracker.data(acc / length(dataset))
end

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
# RSR

function RSR(dataset, model, μ::Real)
    RMSE = 0.0
    STD_obs = 0.0
    for (x, y) in dataset
        ŷ = model(x)
        @assert size(ŷ) == size(y)

        RMSE += sum(abs2.(y .- ŷ))
        STD_obs += sum(abs2.(y .- μ))
    end

    Flux.Tracker.data(RMSE) / Flux.Tracker.data(STD_obs)
end