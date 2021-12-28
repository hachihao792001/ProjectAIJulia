
#---------------------------------
#---------------------------------
#--Tạo conditional plan-----------
#---------------------------------
struct ConditionalPlan
    a # action to take at root
    subplans # dictionary mapping observations to subplans
end
ConditionalPlan(a) = ConditionalPlan(a, Dict())
(π::ConditionalPlan)() = π.a
(π::ConditionalPlan)(o) = π.subplans[o]



function create_conditional_plans(𝒫, d)
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
        # thực hiện với mỗi agent như sau
    # Với mỗi 1 action trong action space, tạo 1 ConditionalPlan
    # với a_i trong action space được chọn làm root
    Π = [[ConditionalPlan(ai) for ai in 𝒜[i]] for i in ℐ]
    # Theo độ sâu d của plan, từ đó mở rộng conditional plan ra d lần
    for t in 1:d
        Π = expand_conditional_plans(𝒫, Π)
    end
    return Π
end


function expand_conditional_plans(𝒫, Π)
    # Lấy agent, actions, observations của pomg
    ℐ, 𝒜, 𝒪 = 𝒫.ℐ, 𝒫.𝒜, 𝒫.𝒪
    return [[ConditionalPlan(ai, Dict(oi => πi for oi in 𝒪[i])) for πi in Π[i] for ai in 𝒜[i]] for i in ℐ]
end


joint(X) = vec(collect(Base.Iterators.product(X...)))
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]
