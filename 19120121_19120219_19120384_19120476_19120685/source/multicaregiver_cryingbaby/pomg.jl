include("multicaregiver.jl")

struct POMG
    γ  # discount factor
    ℐ  # agents
    𝒮  # state space
    𝒜  # joint action space
    𝒪  # joint observation space
    T  # transition function
    O  # joint observation function
    R  # joint reward function
end

#-----------POMG------------------
# Hàm tạo 1 POMG từ các thuộc tính của MultiCaregiverCryingBaby
function POMG(pomg::BabyPOMG)
    return POMG(
        pomg.babyPOMDP.γ,
        vec(collect(1:n_agents(pomg))),
        ordered_states(pomg),
        [ordered_actions(pomg, i) for i in 1:n_agents(pomg)],
        [ordered_observations(pomg, i) for i in 1:n_agents(pomg)],
        (s, a, s′) -> transition(pomg, s, a, s′),
        (a, s′, o) -> joint_observation(pomg, a, s′, o),
        (s, a) -> joint_reward(pomg, s, a)
    )
end

