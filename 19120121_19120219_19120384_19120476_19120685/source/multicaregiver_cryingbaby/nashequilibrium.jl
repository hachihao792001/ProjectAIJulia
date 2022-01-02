include("conditionalplan.jl")
include("utility.jl")
include("simplegame.jl")

# Cấu trúc của POMG Nash
struct POMGNashEquilibrium
    b # initial belief
    d # depth of conditional plans
end

struct NashEquilibrium end

function tensorform(𝒫::SimpleGame)
    ℐ, 𝒜, R = 𝒫.ℐ, 𝒫.𝒜, 𝒫.R
    ℐ′ = eachindex(ℐ)
    𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
    R′ = [R(a) for a in joint(𝒜)]
    return ℐ′, 𝒜′, R′
end

Pkg.add("JuMP")
using JuMP
Pkg.add("Ipopt")
using Ipopt
# Thuật toán NashEquilibrium của Simple game
function solveNE(M::NashEquilibrium, 𝒫::SimpleGame)
    ℐ, 𝒜, R = tensorform(𝒫)
    model = Model(Ipopt.Optimizer)
    @variable(model, U[ℐ])
    @variable(model, π[i=ℐ, 𝒜[i]] ≥ 0)
    @NLobjective(model, Min,
        sum(U[i] - sum(prod(π[j,a[j]] for j in ℐ) * R[y][i]
            for (y,a) in enumerate(joint(𝒜))) for i in ℐ))
    @NLconstraint(model, [i=ℐ, ai=𝒜[i]],
        U[i] ≥ sum(
            prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : π[j,a[j]] for j in ℐ)
            * R[y][i] for (y,a) in enumerate(joint(𝒜))))
    @constraint(model, [i=ℐ], sum(π[i,ai] for ai in 𝒜[i]) == 1)
    optimize!(model)
    πi′(i) = SimpleGamePolicy(𝒫.𝒜[i][ai] => value(π[i,ai]) for ai in 𝒜[i])
    return [πi′(i) for i in ℐ]
end


# Từ Conditional Plan, và Utility của POMG, chuyển về dạng Simple game và giải quyết theo NashEquilibrium của Simple Game 
function solve(M::POMGNashEquilibrium, 𝒫::POMG)
    ℐ, γ, b, d = 𝒫.ℐ, 𝒫.γ, M.b, M.d
    # Tạo conditional plan
    Π = create_conditional_plans(𝒫, d)
    # Tính hàm utility
    U = Dict(π => utility(𝒫, b, π) for π in joint(Π))
    # Chuyển về Simple Game
    𝒢 = SimpleGame(γ, ℐ, Π, π -> U[π])
    # Dùng NashEquilibrium của Simple Game 
    π = solveNE(NashEquilibrium(), 𝒢)
    # Trả về 1 tuple chứa plan của 2 agent thỏa điều kiện reward của agent là cao nhất => từ đó suy ra được chuỗi action của mỗi agent cần làm
    return Tuple(argmax(πi.p) for πi in π)
end