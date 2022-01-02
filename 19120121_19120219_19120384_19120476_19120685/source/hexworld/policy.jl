#struct of the policy
include("visual.jl")
struct ValueFunctionPolicy
    mdp::MDP
    U::Vector{Float64}
end

# lookahead return value for each action (a) at state (s)
function lookahead(mdp::MDP, U::Vector{Float64}, s::Int64, a::Int64)
    𝒮, T, R, γ = mdp.𝒮, mdp.T, mdp.R, mdp.γ
    return R[s, a] + γ * sum(T[s, a, s′] * U[s′] for s′ in 𝒮)
end

#search for the policy (action should go)
function greedy(mdp::MDP, U::Vector{Float64}, s)
    u, a = findmax(a -> lookahead(mdp, U, s, a), mdp.𝒜)
    return (a = a, u = u)
end

#lamdba expression with type of function is ValueFunctionPolicy
#the argument is a state (it is a number)
(policy::ValueFunctionPolicy)(s) = greedy(policy.mdp, policy.U, s).a # tra ve so nguyen


#duyệt qua nhiều lần và trả về kết quả mảng U
function iterative_policy_evaluation(mdp::MDP, policy, k_max)
    𝒮 = mdp.𝒮
    U = [0.0 for s in 𝒮]
    for k in 1:k_max
        U = [lookahead(mdp, U, s, policy(s)) for s in 𝒮]
    end
    return U
end

function policy_evalation(mdp::MDP,policy)
    𝒮,R,T,γ=mdp.𝒮,mdp.R,mdp.T,mdp.γ
    R′=[R[s,policy(s)] for s in 𝒮]
    T′=[T(s,policy(s),s′) for s in 𝒮,s′ in 𝒮]
    I=Matrix{Int}(I,length(R′),length(R′))
    return (I-γ*T′)\R′
end

#call this function in main to solve this problem
function Solve(mdp::MDP, policy, k_max)
    𝒮 = mdp.𝒮
    hexworld=HexWorld()
    U=[0.0 for s in 𝒮]
    for k = 1:k_max
        # println(k)
        U = iterative_policy_evaluation(mdp, policy, k_max)
        policy′ = ValueFunctionPolicy(mdp, U)
        # println("U: $U")
        actions=[policy′(s) for s in 𝒮]
        visualiseHexworld("src/hexworld/visualization/iteration_$k.png",hexworld.hexes,actions,U)
        if all(policy(s) == policy′(s) for s in 𝒮)
            break
        end
        policy = policy′
    end
    return [policy(s) for s in 𝒮]
end



