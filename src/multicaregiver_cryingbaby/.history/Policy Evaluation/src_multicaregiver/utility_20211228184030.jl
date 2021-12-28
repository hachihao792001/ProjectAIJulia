

function lookahead(𝒫::POMG, U, s, a)
    𝒮, 𝒪, T, O, R, γ = 𝒫.𝒮, joint(𝒫.𝒪), 𝒫.T, 𝒫.O, 𝒫.R, 𝒫.γ
    println("===abc========================================================")
    # # tmp = sum(    O(a,s′,o)*U(o,s′) for o in 𝒪   )for s′ in 𝒮
    # result = 0
    # print("My O: ")
    # println(𝒪)
    # for s′ in 𝒮
    #     for o in 𝒪
    #         my_o = O(a,s′,o)
    #         println("my o:---------------------------")
    #         println(my_o)
    #         println("my u:---------------------------")
    #         my_u = U(o,s′) 
    #         println(my_u)

    #         t=my_o*my_u
    #         result+=t
    #     end
    # end
    # println("===+++++++++++++++++xxxxxxxxxxxxxxxx")

    # u′ = sum(  T(s,a,s′) * result)
                 
    u′ = sum(T(s,a,s′)*sum(O(a,s′,o)*U(o,s′) for o in 𝒪) for s′ in 𝒮)
    return R(s,a) + γ*u′
end

function evaluate_plan(𝒫::POMG, π, s)
    a = Tuple(πi() for πi in π)
    U(o,s′) = evaluate_plan(𝒫, [πi(oi) for (πi, oi) in zip(π,o)], s′)

    return isempty(first(π).subplans) ? 𝒫.R(s,a) : lookahead(𝒫, U, s, a)
end


# Tính Utility theo công thức (26.2 Algorithms for Decision Making)
function utility(𝒫::POMG, b, π)
    u = [evaluate_plan(𝒫, π, s) for s in 𝒫.𝒮]
    return sum(bs * us for (bs, us) in zip(b, u))
end 