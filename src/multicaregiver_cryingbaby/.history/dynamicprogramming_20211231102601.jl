include("nashequilibrium.jl")

# Cấu trúc của POMG theo dynamicprogramming
struct POMGDynamicProgramming
    b # giá belief khởi tạo 
    d # độ cao của conditional plans
end

Pkg.add("JuMP")
using JuMP
    
# Hàm kiểm tra xem policy  πi có phụ thuộc vào một policy khác theo công thức (... trong report)
function is_dominated(𝒫::POMG, Π, i, πi)
    ℐ, 𝒮 = 𝒫.ℐ, 𝒫.𝒮
    jointΠnoti = joint([Π[j] for j in ℐ if j ≠ i])
    π(πi′, πnoti) = [j==i ? πi′ : πnoti[j>i ? j-1 : j] for j in ℐ]
    Ui = Dict((πi′, πnoti, s) => evaluate_plan(𝒫, π(πi′, πnoti), s)[i]
                for πi′ in Π[i], πnoti in jointΠnoti, s in 𝒮)
    model = Model(Ipopt.Optimizer)
    @variable(model, δ)
    @variable(model, b[jointΠnoti, 𝒮] ≥ 0)
    @objective(model, Max, δ)
    @constraint(model, [πi′=Π[i]],
        sum(b[πnoti, s] * (Ui[πi′, πnoti, s] - Ui[πi, πnoti, s])
        for πnoti in jointΠnoti for s in 𝒮) ≥ δ)
    @constraint(model, sum(b) == 1)
    optimize!(model)
    return value(δ) ≥ 0
end

using Random
# Hàm thực hiện việc loại bỏ các policy có phụ thuộc
function prune_dominated!(Π, 𝒫::POMG)
    done = false
    while !done
        done = true
        for i in shuffle(𝒫.ℐ)
            for πi in shuffle(Π[i])
                if length(Π[i]) > 1 && is_dominated(𝒫, Π, i, πi)
                    filter!(πi′ -> πi′ ≠ πi, Π[i])
                    done = false
                    break
                end
            end
        end
    end
end

    # Loại bỏ khác policy phụ thuộc, chuyển bài toán về SimpleGame và thực hiện NashEquilibrium
function solveDP(M::POMGDynamicProgramming, 𝒫::POMG)
    ℐ, 𝒮, 𝒜, R, γ, b, d = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.γ, M.b, M.d
    # Tạo ConditionalPlan
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    # Xét theo độ sâu d của plan
    for t in 1:d
        # Mở rộng plan theo độ sâu, nếu có sự phụ thuộc thì loại bỏ
        Π = expand_conditional_plans(𝒫, Π)
        prune_dominated!(Π, 𝒫)
    end

    # Chuyển về dạng simple game
    𝒢 = SimpleGame(γ, ℐ, Π, π -> utility(𝒫, b, π))
    π = solveNE(NashEquilibrium(), 𝒢)
    return Tuple(argmax(πi.p) for πi in π)
end
    

