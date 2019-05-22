
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

function mse_rnn(x, y)
    l = Flux.mse(m(x), y)
    Flux.reset!(m)
    return l
end
