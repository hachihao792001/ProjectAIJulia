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