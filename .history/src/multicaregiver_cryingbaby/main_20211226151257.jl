struct POMDP
    γ   # discount factor
    𝒮   # state space
    𝒜   # action space
    𝒪   # observation space
    T   # transition function
    R   # reward function
    O   # observation function
    TRO # sample transition, reward, and observation
end

struct BoolDistribution
    p::Float64 # probability of true
end

pdf(d::BoolDistribution, s::Bool) = s ? d.p : 1.0-d.p
rand(rng::AbstractRNG, d::BoolDistribution) = rand(rng) <= d.p
iterator(d::BoolDistribution) = [true, false]
Base.:(==)(d1::BoolDistribution, d2::BoolDistribution) = d1.p == d2.p
Base.hash(d::BoolDistribution, u::UInt64=UInt64(0)) = hash(d.p, u)
Base.length(d::BoolDistribution) = 2

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

# CryingBaby = CryingBaby(-10.0, -5.0, -0.5, 0.1, 0.8, 0.1, 0.9, 0.9)

SATED = 1
HUNGRY = 2
FEED = 1
IGNORE = 2
SING = 3
CRYING = true
QUIET = false

CRYING_BABY_ACTION_COLORS = Dict(
    FEED => "pastelBlue",
    IGNORE => "pastelGreen",
    SING => "pastelRed"
)
CRYING_BABY_ACTION_NAMES = Dict(
    FEED => "feed",
    IGNORE => "ignore",
    SING => "sing",
)

n_states(::CryingBaby) = 2
n_actions(::CryingBaby) = 3
n_observations(::CryingBaby) = 2
discount(pomdp::CryingBaby) = pomdp.γ

ordered_states(::CryingBaby) = [SATED, HUNGRY]
ordered_actions(::CryingBaby) = [FEED, IGNORE, SING]
ordered_observations(::CryingBaby) = [CRYING,QUIET]

# 2 tỷ lệ cộng lại là 100%
two_state_categorical(p1::Float64) = Categorical([p1,1.0 - p1])

# Hàm transition, tại state s, thực hiện action a => tỷ lệ (STATED, HUNGRY)
function transition(pomdp::CryingBaby, s::Int, a::Int)
    # Nếu action a là FEED -> dù state s là HUNGRY, hay STATED -> tỷ lệ (STATED, HUNGRY) = (100%, 0%)
    if a == FEED
        return two_state_categorical(1.0)
    else
        # Nếu state s là HUNGRY -> dù action là FEED,SING,IGNORE -> tỷ lệ (STATED, HUNGRY) = (0%, 100%)
        if s == HUNGRY
            return two_state_categorical(0.0)
        else
        # Trường hợp còn lại là state s là SATED, action SING/ IGNORE -> tỷ lệ (STATED, HUNGRY) = (90%, 10%)
            return two_state_categorical(1.0-pomdp.p_become_hungry)
        end
    end
end

#Hàm observation, tại state s', thực hiện action a => tỷ lệ (CRYING)
function observation(pomdp::CryingBaby, a::Int, s′::Int)
    if a == SING
        # Nếu state HUNGRY, thực hiện SING -> tỷ lệ (CRYING) = (90%)
        if s′ == HUNGRY
            return BoolDistribution(pomdp.p_cry_when_hungry_in_sing)
        # Nếu state STAED, thực hiện SING -> tỷ lệ CRYING (0%)
        else
            return BoolDistribution(0.0)
        end
    else 
        # Nếu state HUNGRY, thực hiện FEED/IGNORE -> tỷ lệ (CRYING) = (80%)/90%???
       
       #BUGGGGGGGGGGGGGGGGGGGGGGGGGG
        if s′ == HUNGRY
            return BoolDistribution(pomdp.p_cry_when_hungry)
        else
        # Nếu state STATED, thực hiện FEED/IGNORE -> tỷ lệ (CRYING) = 10%
            return BoolDistribution(pomdp.p_cry_when_not_hungry)
        end
    end
end

# Hàm reward, giá trị phần thưởng ứng với state s và thực hiện action a
function reward(pomdp::CryingBaby, s::Int, a::Int)
    r = 0.0
    # Nếu state HUNGRY -> reward +=-10
    if s == HUNGRY
        r += pomdp.r_hungry
    end
    # Nếu action FEED -> reward += -5
    #            SING -> reward +=-0.5
    if a == FEED
        r += pomdp.r_feed
    elseif a == SING
        r += pomdp.r_sing
    end
    return r
end

# Reward chính là tổng của tất cả các reward của 1 baby, ứng với state s và action a trong tất cả state >?????????
#?????????????????????

reward(pomdp::CryingBaby, b::Vector{Float64}, a::Int) = sum(reward(pomdp,s,a)*b[s] for s in ordered_states(pomdp))

# γ Tỷ lệ chiết khấu , ở bài này lấy γ= 0.9
function DiscretePOMDP(pomdp::CryingBaby; γ::Float64=pomdp.γ)
    nState = n_states(pomdp)
    nAction = n_actions(pomdp)
    nObservation = n_observations(pomdp)

    T = zeros(nS, nA, nS)
    R = Array{Float64}(undef, nS, nA)
    O = Array{Float64}(undef, nA, nS, nO)

    s_s = 1
    s_h = 2

    a_f = 1
    a_i = 2
    a_s = 3

    o_c = 1
    o_q = 2

    T[s_s, a_f, :] = [1.0, 0.0]
    T[s_s, a_i, :] = [1.0-pomdp.p_become_hungry, pomdp.p_become_hungry]
    T[s_s, a_s, :] = [1.0-pomdp.p_become_hungry, pomdp.p_become_hungry]
    T[s_h, a_f, :] = [1.0, 0.0]
    T[s_h, a_i, :] = [0.0, 1.0]
    T[s_h, a_s, :] = [0.0, 1.0]

    R[s_s, a_f] = reward(pomdp, s_s, a_f)
    R[s_s, a_i] = reward(pomdp, s_s, a_i)
    R[s_s, a_s] = reward(pomdp, s_s, a_s)
    R[s_h, a_f] = reward(pomdp, s_h, a_f)
    R[s_h, a_i] = reward(pomdp, s_h, a_i)
    R[s_h, a_s] = reward(pomdp, s_h, a_s)

    O[a_f, s_s, :] = [observation(pomdp, a_f, s_s).p, 1 - observation(pomdp, a_f, s_s).p]
    O[a_f, s_h, :] = [observation(pomdp, a_f, s_h).p, 1 - observation(pomdp, a_f, s_h).p]
    O[a_i, s_s, :] = [observation(pomdp, a_i, s_s).p, 1 - observation(pomdp, a_i, s_s).p]
    O[a_i, s_h, :] = [observation(pomdp, a_i, s_h).p, 1 - observation(pomdp, a_i, s_h).p]
    O[a_s, s_s, :] = [observation(pomdp, a_s, s_s).p, 1 - observation(pomdp, a_s, s_s).p]
    O[a_s, s_h, :] = [observation(pomdp, a_s, s_h).p, 1 - observation(pomdp, a_s, s_h).p]

    return DiscretePOMDP(T, R, O, γ)
end

function POMDP(pomdp::CryingBaby; γ::Float64=pomdp.γ)
    disc_pomdp = DiscretePOMDP(pomdp)
    return POMDP(disc_pomdp)
end


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


struct BabyPOMG
    babyPOMDP::CryingBaby
end

function MultiCaregiverCryingBaby()
    BabyPOMDP = CryingBaby()
    return BabyPOMG(BabyPOMDP)
end

n_agents(pomg::BabyPOMG) = 2

ordered_states(pomg::BabyPOMG) = [SATED, HUNGRY]
ordered_actions(pomg::BabyPOMG, i::Int) = [FEED, IGNORE, SING]
ordered_joint_actions(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_actions(pomg, i) for i in 1:n_agents(pomg)]...)))

n_actions(pomg::BabyPOMG, i::Int) = length(ordered_actions(pomg, i))
n_joint_actions(pomg::BabyPOMG) = length(ordered_joint_actions(pomg))

ordered_observations(pomg::BabyPOMG, i::Int) = [CRYING, QUIET]
ordered_joint_observations(pomg::BabyPOMG) = vec(collect(Iterators.product([ordered_observations(pomg, i) for i in 1:n_agents(pomg)]...)))

n_observations(pomg::BabyPOMG, i::Int) = length(ordered_observations(pomg, i))
n_joint_observations(pomg::BabyPOMG) = length(ordered_joint_observations(pomg))


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
end

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

joint_reward(pomg::BabyPOMG, b::Vector{Float64}, a) = sum(joint_reward(pomg, s, a) * b[s] for s in ordered_states(pomg))

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


