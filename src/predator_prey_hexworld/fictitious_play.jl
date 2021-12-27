function tensorform(𝒫::MG)
	ℐ, 𝒮, 𝒜, R, T = 𝒫.ℐ, 𝒫.𝒮, 𝒫.𝒜, 𝒫.R, 𝒫.T
	ℐ′ = eachindex(ℐ)
	𝒮′ = eachindex(𝒮)
	𝒜′ = [eachindex(𝒜[i]) for i in ℐ]
	R′ = [R(s,a) for s in 𝒮, a in joint(𝒜)]
	T′ = [T(s,a,s′) for s in 𝒮, a in joint(𝒜), s′ in 𝒮]
	return ℐ′, 𝒮′, 𝒜′, R′, T′
end

function solve(M::NashEquilibrium, 𝒫::MG)
	ℐ, 𝒮, 𝒜, R, T = tensorform(𝒫)

	𝒮′, 𝒜′, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.γ

	model = Model(Ipopt.Optimizer)

	@variable(model, U[ℐ, 𝒮])

	@variable(model, π[i=ℐ, 𝒮, ai=𝒜[i]] ≥ 0)

	@NLobjective(model, Min,
		sum(U[i,s] - sum(prod(π[j,s,a[j]] for j in ℐ)
			* (R[s,y][i] + γ*sum(T[s,y,s′]*U[i,s′] for s′ in 𝒮))
			for (y,a) in enumerate(joint(𝒜))) for i in ℐ, s in 𝒮))

	@NLconstraint(model, [i=ℐ, s=𝒮, ai=𝒜[i]],
		U[i,s] ≥ sum(
			prod(j==i ? (a[j]==ai ? 1.0 : 0.0) : π[j,s,a[j]] for j in ℐ)
			* (R[s,y][i] + γ*sum(T[s,y,s′]*U[i,s′] for s′ in 𝒮))
			for (y,a) in enumerate(joint(𝒜))))

	@constraint(model, [i=ℐ, s=𝒮], sum(π[i,s,ai] for ai in 𝒜[i]) == 1)

	optimize!(model)
	π′ = value.(π)
	πi′(i,s) = SimpleGamePolicy(𝒜′[i][ai] => π′[i,s,ai] for ai in 𝒜[i])
	πi′(i) = MGPolicy(𝒮′[s] => πi′(i,s) for s in 𝒮)

	return [πi′(i) for i in ℐ]
end

function randstep(𝒫::MG, s, a)
	s′ = rand(SetCategorical(𝒫.𝒮, [𝒫.T(s, a, s′) for s′ in 𝒫.𝒮]))
	r = 𝒫.R(s,a)
	return s′, r
end

function simulate(𝒫::MG, π, k_max, b)
	s = rand(b)
	for k = 1:k_max
		a = Tuple(πi(s)() for πi in π)
		s′, r = randstep(𝒫, s, a)

		for πi in π
			update!(πi, s, a, s′)
		end

		s = s′
	end
	
	return π
end