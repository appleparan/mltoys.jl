using CuArrays

using Flux
using Flux.Tracker
using Flux.Tracker: param, back!, grad, data

function RSME(dataset, model)
    # RSME
    acc = 0.0

    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        acc += sqrt(sum(abs2.(ŷ .- y)) / length(y))
    end

    # acc is incomplete TrackedReal, convert it to pure Real type
    Flux.Tracker.data(acc / length(dataset))
end

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.532.2506&rep=rep1&type=pdf
# RSR
function RSR(setname::String, dataset, model, μσ)
    RMSE = 0.0
    STD_obs = 0.0
    μ = μσ[setname, "μ"][:value]
    for (x, y) in dataset
        ŷ = model(x |> gpu)
        @assert size(ŷ) == size(y)

        RMSE += sum(abs2.(y .- ŷ))
        STD_obs += sum(abs2.(y .- μ))
    end

    sqrt(Flux.Tracker.data(RMSE)) / sqrt(Flux.Tracker.data(STD_obs))
end

function huber_loss(ŷ, y)
    δ = 0.5
    error = abs.(ŷ .- y)
    cond = error .< δ
    squared_loss = 0.5 * error.^2
    linear_loss = δ .* (error .- δ .* 0.5)

    Int.(cond) .* squared_loss + Int.(.!cond) .* linear_loss    
end

function huber_loss_mean(ŷ, y)
    mean.(humber_loss(ŷ, y))
end
