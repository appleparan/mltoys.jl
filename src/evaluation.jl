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
    error = abs.(ŷ - y)
    cond = error .< δ
    squared_loss = 0.5 * error.^2
    linear_loss = δ .* (error .- δ^2 .* 0.5)

    is_squared = Int.(cond)
    is_linear = Int.(.!cond)

    # why this line doesn't work?
    # Int.(cond) .* squared_loss + Int.(.!cond) .* (10 .* linear_loss)
    #=
    ERROR: LoadError: MethodError: no method matching !(::ForwardDiff.Dual{Nothing,Bool,1})
    Closest candidates are:
        !(!Matched::Missing) at missing.jl:83
        !(!Matched::Bool) at bool.jl:35
        !(!Matched::Function) at operators.jl:853
    =#
    is_squared .* squared_loss + is_linear .* (10 .* linear_loss)
end

function huber_loss_mean(ŷ, y)
    mean(huber_loss(ŷ, y))
end
