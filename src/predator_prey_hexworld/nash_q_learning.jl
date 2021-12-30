mutable struct NashQLearning
	𝒫 # Markov game
	i # agent index
	Q # state-action value estimates
	N # history of actions performed
end

function NashQLearning(𝒫::MG, i)
	ℐ, 𝒮, 𝒜 = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜
	Q = Dict((j, s, a) => 0.0 for j in ℐ, s in 𝒮, a in joint(𝒜))
	N = Dict((s, a) => 1.0 for s in 𝒮, a in joint(𝒜))
	return NashQLearning(𝒫, i, Q, N)
end

function (πi::NashQLearning)(s)
	𝒫, i, Q, N = πi.𝒫, πi.i, πi.Q, πi.N
	ℐ, 𝒮, 𝒜, 𝒜i, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.𝒜[πi.i], 𝒫.γ
	M = NashEquilibrium()
	𝒢 = SimpleGame(γ, ℐ, 𝒜, a -> [Q[j, s, a] for j in ℐ])
	π = solve(M, 𝒢)
	ϵ = 1 / sum(N[s, a] for a in joint(𝒜))
	πi′(ai) = ϵ/length(𝒜i) + (1-ϵ)*π[i](ai)
	return SimpleGamePolicy(ai => πi′(ai) for ai in 𝒜i)
end

function update!(πi::NashQLearning, s, a, s′)
	𝒫, ℐ, 𝒮, 𝒜, R, γ = πi.𝒫, πi.𝒫.ℐ, πi.𝒫.𝒮, πi.𝒫.𝒜, πi.𝒫.R, πi.𝒫.γ
	i, Q, N = πi.i, πi.Q, πi.N
	M = NashEquilibrium()
	𝒢 = SimpleGame(γ, ℐ, 𝒜, a′ -> [Q[j, s′, a′] for j in ℐ])
	π = solve(M, 𝒢)
	πi.N[s, a] += 1
	α = 1 / sqrt(N[s, a])
	for j in ℐ
		πi.Q[j,s,a] += α*(R(s,a)[j] + γ*utility(𝒢,π,j) - Q[j,s,a])
	end
end