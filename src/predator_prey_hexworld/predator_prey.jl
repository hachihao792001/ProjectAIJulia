struct MGPolicy
	p # dictionary mapping states to simple game policies
	MGPolicy(p::Base.Generator) = new(Dict(p))
end

(πi::MGPolicy)(s, ai) = πi.p[s](ai)
(πi::SimpleGamePolicy)(s, ai) = πi(ai)

probability(𝒫::MG, s, π, a) = prod(πj(s, aj) for (πj, aj) in zip(π, a))
reward(𝒫::MG, s, π, i) =
	sum(𝒫.R(s,a)[i]*probability(𝒫,s,π,a) for a in joint(𝒫.𝒜))
transition(𝒫::MG, s, π, s′) =
	sum(𝒫.T(s,a,s′)*probability(𝒫,s,π,a) for a in joint(𝒫.𝒜))

function policy_evaluation(𝒫::MG, π, i)
	𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
	p(s,a) = prod(πj(s, aj) for (πj, aj) in zip(π, a))
	R′ = [sum(R(s,a)[i]*p(s,a) for a in joint(𝒜)) for s in 𝒮]
	T′ = [sum(T(s,a,s′)*p(s,a) for a in joint(𝒜)) for s in 𝒮, s′ in 𝒮]
	return (I - γ*T′)\R′
end

function best_response(𝒫::MG, π, i)
	𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
	T′(s,ai,s′) = transition(𝒫, s, joint(π, SimpleGamePolicy(ai), i), s′)
	R′(s,ai) = reward(𝒫, s, joint(π, SimpleGamePolicy(ai), i), i)
	πi = solve(MDP(γ, 𝒮, 𝒜[i], T′, R′))
	return MGPolicy(s => SimpleGamePolicy(πi(s)) for s in 𝒮)
end

function softmax_response(𝒫::MG, π, i, λ)
	𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
	T′(s,ai,s′) = transition(𝒫, s, joint(π, SimpleGamePolicy(ai), i), s′)
	R′(s,ai) = reward(𝒫, s, joint(π, SimpleGamePolicy(ai), i), i)
	mdp = MDP(γ, 𝒮, joint(𝒜), T′, R′)
	πi = solve(mdp)
	Q(s,a) = lookahead(mdp, πi.U, s, a)
	p(s) = SimpleGamePolicy(a => exp(λ*Q(s,a)) for a in 𝒜[i])
	return MGPolicy(s => p(s) for s in 𝒮)
end

struct PredatorPreyHexWorldMG
    hexes::Vector{Tuple{Int, Int}}
    hexWorldDiscreteMDP::DiscreteMDP
end

n_agents(mg::PredatorPreyHexWorldMG) = 2

ordered_states(mg::PredatorPreyHexWorldMG, i::Int) = vec(collect(1:length(mg.hexes)))
ordered_states(mg::PredatorPreyHexWorldMG) = vec(collect(Iterators.product([ordered_states(mg, i) for i in 1:n_agents(mg)]...)))

ordered_actions(mg::PredatorPreyHexWorldMG, i::Int) = vec(collect(1:n_actions(mg.hexWorldDiscreteMDP)))
ordered_joint_actions(mg::PredatorPreyHexWorldMG) = vec(collect(Iterators.product([ordered_actions(mg, i) for i in 1:n_agents(mg)]...)))

n_actions(mg::PredatorPreyHexWorldMG, i::Int) = length(ordered_actions(mg, i))
n_joint_actions(mg::PredatorPreyHexWorldMG) = length(ordered_joint_actions(mg))

function transition(mg::PredatorPreyHexWorldMG, s, a, s′)
    # When a prey is caught (new prey born), it teleports to a random location and the predator remains (eating).
    # Otherwise, both transition following HexWorldMDP.
    if s[1] == s[2]
        prob = Float64(s′[1] == s[1]) / length(mg.hexes)
    else
        prob = mg.hexWorldDiscreteMDP.T[s[1], a[1], s′[1]] * mg.hexWorldDiscreteMDP.T[s[2], a[2], s′[2]]
    end
    return prob
end

function reward(mg::PredatorPreyHexWorldMG, i::Int, s, a)
    r = 0.0

    if i == 1
        # Predators get -1 for moving and 10 for catching the prey.
        if s[1] == s[2]
            return 10.0
        else
            return -1.0
        end
    elseif i == 2
        # Prey get -1 for moving and -100 for being caught.
        if s[1] == s[2]
            r = -100.0
        else
            r = -1.0
        end
    end

    return r
end

function joint_reward(mg::PredatorPreyHexWorldMG, s, a)
    return [reward(mg, i, s, a) for i in 1:n_agents(mg)]
end

function MG(mg::PredatorPreyHexWorldMG)
    return MG(
        mg.hexWorldDiscreteMDP.γ,
        vec(collect(1:n_agents(mg))),
        ordered_states(mg),
        [ordered_actions(mg, i) for i in 1:n_agents(mg)],
        (s, a, s′) -> transition(mg, s, a, s′),
        (s, a) -> joint_reward(mg, s, a)
    )
end

function PredatorPreyHexWorldMG(hexes::Vector{Tuple{Int,Int}},
                                r_bump_border::Float64,
                                p_intended::Float64,
                                γ::Float64)
    hexWorld = HexWorldMDP(hexes,
                           r_bump_border,
                           p_intended,
                           Dict{Tuple{Int64,Int64},Float64}(),
                           γ)
    mdp = hexWorld.mdp
    return PredatorPreyHexWorldMG(hexes, mdp)
end

# const CirclePredatorPreyHexWorld = PredatorPreyHexWorldMG(
#     [
#      (-1, 2), (0, 2),
#      (-1, 1), (1, 1),
#      (0, 0), (1, 0),
#      ],
#     HexWorldRBumpBorder,
#     HexWorldPIntended,
#     0.95
# )

function CirclePredatorPreyHexWorld()
    CirclePredatorPreyHexWorld = PredatorPreyHexWorldMG(
        [
         (-1, 2), (0, 2), (1, 2),
         (-1, 1), (1, 1), (3, 1), (4, 1),
         (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         ],
        HexWorldRBumpBorder,
        HexWorldPIntended,
        0.95
    )
    return CirclePredatorPreyHexWorld
end

# const PredatorPreyHexWorld = PredatorPreyHexWorldMG(
#     [
#      (-1, 2), (0, 2), (1, 2),
#      (-1, 1), (1, 1), (3, 1), (4, 1),
#      (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
#      ],
#     HexWorldRBumpBorder,
#     HexWorldPIntended,
#     0.95
# )

function PredatorPreyHexWorld()
    PredatorPreyHexWorld = PredatorPreyHexWorldMG(
        [
         (-1, 2), (0, 2), (1, 2),
         (-1, 1), (1, 1), (3, 1), (4, 1),
         (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         ],
        HexWorldRBumpBorder,
        HexWorldPIntended,
        0.95
    )
    return PredatorPreyHexWorld
end
