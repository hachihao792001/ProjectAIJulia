mutable struct MGGradientAscent
	𝒫 # Markov game
	i # agent index
	t # time step
	Qi # state-action value estimates
	πi # current policy
end

function MGGradientAscent(𝒫::MG, i)
	ℐ, 𝒮, 𝒜 = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜
	Qi = Dict((s, a) => 0.0 for s in 𝒮, a in joint(𝒜))
	uniform() = Dict(s => SimpleGamePolicy(ai => 1.0 for ai in 𝒫.𝒜[i]) for s in 𝒮)
	return MGGradientAscent(𝒫, i, 1, Qi, uniform())
end

function (πi::MGGradientAscent)(s)
	𝒜i, t = πi.𝒫.𝒜[πi.i], πi.t
	ϵ = 1 / sqrt(t)
	πi′(ai) = ϵ/length(𝒜i) + (1-ϵ)*πi.πi[s](ai)
	return SimpleGamePolicy(ai => πi′(ai) for ai in 𝒜i)
end

function update!(πi::MGGradientAscent, s, a, s′)
	𝒫, i, t, Qi = πi.𝒫, πi.i, πi.t, πi.Qi
	ℐ, 𝒮, 𝒜i, R, γ = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜[πi.i], 𝒫.R, 𝒫.γ
	jointπ(ai) = Tuple(j == i ? ai : a[j] for j in ℐ)
	α = 1 / sqrt(t)
	Qmax = maximum(Qi[s′, jointπ(ai)] for ai in 𝒜i)
	πi.Qi[s, a] += α * (R(s, a)[i] + γ * Qmax - Qi[s, a])
	u = [Qi[s, jointπ(ai)] for ai in 𝒜i]
	π′ = [πi.πi[s](ai) for ai in 𝒜i]
	π = project_to_simplex(π′ + u / sqrt(t))
	πi.t = t + 1
	πi.πi[s] = SimpleGamePolicy(ai => p for (ai, p) in zip(𝒜i, π))
end