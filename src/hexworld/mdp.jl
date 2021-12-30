struct MDP
    γ  # discount factor
    𝒮  # state space
    𝒜  # action space
    T  # transition function
    R  # reward function
end

#constructor with 3 argument is T,R,γ
function MDP(T::Array{Float64, 3}, R::Array{Float64, 2}, γ::Float64)
    return MDP(γ, 1:size(R,1), 1:size(R,2), T, R)
end

