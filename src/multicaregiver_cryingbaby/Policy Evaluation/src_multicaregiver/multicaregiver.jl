using Pkg
Pkg.add("Parameters")
using Parameters: @with_kw

include("pomg.jl")

# Khai báo cấu trúc của Crying Baby với các thuộc tính có giá trị mặc định
@with_kw struct CryingBaby
    # hungry reward khi HUNGRY độc lập với các action khác 
    r_hungry::Float64 = -10.0
    # feed reward
    r_feed::Float64 = -5.0 
    # sing reward
    r_sing::Float64 = -0.5
    # Tỷ lệ sẽ trở nên đói dần = 10%
    p_become_hungry::Float64 = 0.1
    # Tỷ lệ CRYING khi HUNGRY = 80%
    p_cry_when_hungry::Float64 = 0.8
    # Tỷ lệ CRYING khi SATED = 10%
    p_cry_when_not_hungry::Float64 = 0.1
    # Tỷ lệ CRYING khi HUNGRY + SING = 90%
    p_cry_when_hungry_in_sing::Float64 = 0.9
    γ::Float64 = 0.9
end

# Khai báo giá trị mặc định cho các thuộc tính
SATED = 1
HUNGRY = 2
FEED = 1
IGNORE = 2
SING = 3
CRYING = true
QUIET = false

# Cấu trúc của Baby POMG là 1 CryingBaby
struct BabyPOMG
    babyPOMDP::CryingBaby
end

# Cấu trúc của MultiCaregiverCryingBaby cũng bao gồm 1 Crying Baby
function MultiCaregiverCryingBaby()
    BabyPOMDP = CryingBaby()
    return BabyPOMG(BabyPOMDP)
end

# Số agent mặc định là 2
n_agents(pomg::BabyPOMG) = 2
# State space 
ordered_states(pomg::BabyPOMG) = [SATED, HUNGRY]
# Action space
ordered_actions(pomg::BabyPOMG, i::Int) = [FEED, IGNORE, SING]
# Joint action
ordered_joint_actions(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_actions(pomg, i) for i in 1:n_agents(pomg)]...)))
n_actions(pomg::BabyPOMG, i::Int) = length(ordered_actions(pomg, i))
n_joint_actions(pomg::BabyPOMG) = length(ordered_joint_actions(pomg))
# Observation space
ordered_observations(pomg::BabyPOMG, i::Int) = [CRYING, QUIET]
# Joint observation
ordered_joint_observations(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_observations(pomg, i) for i in 1:n_agents(pomg)]...)))
n_observations(pomg::BabyPOMG, i::Int) = length(ordered_observations(pomg, i))
n_joint_observations(pomg::BabyPOMG) = length(ordered_joint_observations(pomg))

# TRANSITION FUNCTION T (s' | s,a)
#Hàm transition, tức từ state s, thông qua action a, chuyển sang state s'
function transition(pomg::BabyPOMG, s, a, s′)

    # Nếu 1 trong 2 agent có action là FEED:
    #      ->  khả năng baby STATED = 100%
    #      ->  khả năng baby HUNGRY = 0%
    if a[1] == FEED || a[2] == FEED
        if s′ == SATED
            return 1.0
        else
            return 0.0
        end
    else

    # Nếu cả 2 agent có hành động là SING, hoặc IGNORE thì:
    #       Nếu ban đầu state s là HUNGRY -> state s' vẫn là HUNGRY = 100%
    #                                     -> state s' SATED = 0%
        if s == HUNGRY
            if s′ == HUNGRY
                return 1.0
            else
                return 0.0
            end
    #       Nếu ban đầu state s là STATED -> khả năng trở nên đói = 50%
    #                                     -> state s' là STATED = 100% - khả năng đói
    #                                     -> state s' là HUNGRY = khả năng đói
        else
            probBecomeHungry = 0.5 #pomg.babyPOMDP.p_become_hungry
            if s′ == SATED
                return 1.0 - probBecomeHungry
            else
                return probBecomeHungry
            end
        end
    end
end


# OBSERVATION FUNCTION O (o |  a, s')
#Hàm Joint Observation, tức từ action a, và state s', quan sát được observation o => tỷ lệ của observation của baby trong trường hợp này
function joint_observation(pomg::BabyPOMG, a, s′, o)
    # Nếu 1 trong 2 agent thực hiện action SING:
    if a[1] == SING || a[2] == SING
        # Nếu state s' là HUNGRY:
        #   +Observation của cả 2 đều là CRYING -> tỷ lệ CRYING = 90%
        #   +Observation của cả 2 đều là QUIET  -> tỷ lệ QUITED = 10%

        if s′ == HUNGRY
            if o[1] == CRYING && o[2] == CRYING
                return pomg.babyPOMDP.p_cry_when_hungry_in_sing
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.babyPOMDP.p_cry_when_hungry_in_sing
            else
                return 0.0
            end
        # Nếu state s' là STATED
        #   +Observation của cả 2 đều là QUITED  -> tỷ lệ QUITED = 100%
        #   +Observation 1 trong 2 là CRYING     -> tỷ lệ CRYING  = 0%
        else
            if o[1] == QUIET && o[2] == QUIET
                return 1.0
            else
                return 0.0
            end
        end
    # Nếu 1 trong 2 agent thực hiện action FEED hoặc IGNORE:
    else
   # Nếu state s' là HUNGRY:
        #   +Observation của cả 2 đều là CRYING -> tỷ lệ CRYING = 90%
        #   +Observation của cả 2 đều là QUIET  -> tỷ lệ QUITED = 10%
        if s′ == HUNGRY
            if o[1] == CRYING && o[2] == CRYING
                return pomg.babyPOMDP.p_cry_when_hungry
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.babyPOMDP.p_cry_when_hungry
            else
                return 0.0
            end
    # Nếu state s' là SATED:
        #   +Observation của cả 2 đều là CRYING -> tỷ lệ CRYING = 0%
        #   +Observation của cả 2 đều là QUIET  -> tỷ lệ QUITED = 100%        
            if o[1] == CRYING && o[2] == CRYING
                return pomg.babyPOMDP.p_cry_when_not_hungry
            elseif o[1] == QUIET && o[2] == QUIET
                return 1.0 - pomg.babyPOMDP.p_cry_when_not_hungry
            else
                return 0.0
            end
        end
    end
    return 0.0
end

# REWARD FUNCTION R (a)
#Hàm joint Reward, ứng với state s, thực hiện action a sẽ được thưởng thế nào
function joint_reward(pomg::BabyPOMG, s, a)
    # Thưởng cho 2 agent (bảo mẫu)
    r = [0.0, 0.0]

    # Nếu state là HUNGRY -> cộng là reward của cả 2 giá trị hungry reward = 
    if s == HUNGRY
        r += [pomg.babyPOMDP.r_hungry, pomg.babyPOMDP.r_hungry]
    end

    # Agent thứ nhất có sở trường FEED -> khi FEED, reward + 2.5 (1/2 của feed reward)
    #                                  -> khi SING, reward + 0.5 (sing reward)
    if a[1] == FEED
        r[1] += pomg.babyPOMDP.r_feed / 2.0
    elseif a[1] == SING
        r[1] += pomg.babyPOMDP.r_sing
    end

    # Agent thứ nhất có sở trường SING -> khi FEED, reward + 5.0 (feed reward)
    #                                  -> khi SING, reward + 0.25 (1/2 của sing reward)    
    if a[2] == FEED
        r[2] += pomg.babyPOMDP.r_feed
    elseif a[2] == SING
        r[2] += pomg.babyPOMDP.r_sing / 2.0
    end

    return r
end
#joint reward
joint_reward(pomg::BabyPOMG, b::Vector{Float64}, a) = sum(joint_reward(pomg, s, a) * b[s] for s in ordered_states(pomg))

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

