include("./simplegame.jl")

function best_response(𝒫::MG, π, i)
	𝒮, 𝒜, R, T, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T, 𝒫.γ
	T′(s,ai,s′) = transition(𝒫, s, joint(π, SimpleGamePolicy(ai), i), s′)
	R′(s,ai) = reward(𝒫, s, joint(π, SimpleGamePolicy(ai), i), i)
	πi = solve(MDP(γ, 𝒮, 𝒜[i], T′, R′))
	return MGPolicy(s => SimpleGamePolicy(πi(s)) for s in 𝒮)
end