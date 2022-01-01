# ----------------------------- Fictitious Play --------------------------------
function simulate(𝒫::SimpleGame, π, k_max)
    for k = 1:k_max
        a = [πi() for πi in π]
        for πi in π
            update!(πi, a)
            print("Agent ")
            print(πi.i)
            print(πi.N)
            println(πi.πi)
        end
    end
    return π
end

mutable struct FictitiousPlay
    𝒫 # simple game
    i # agent index
    N # array of action count dictionaries
    πi # current policy
    end
        
function FictitiousPlay(𝒫::SimpleGame, i)
    N = [Dict(aj => 1 for aj in 𝒫.𝒜[j]) for j in 𝒫.ℐ]
    πi = SimpleGamePolicy(ai => 1.0 for ai in 𝒫.𝒜[i])
    return FictitiousPlay(𝒫, i, N, πi)
    end

(πi::FictitiousPlay)() = πi.πi()
    
(πi::FictitiousPlay)(ai) = πi.πi(ai)
    
function update!(πi::FictitiousPlay, a)
    N, 𝒫, ℐ, i = πi.N, πi.𝒫, πi.𝒫.ℐ, πi.i
    for (j, aj) in enumerate(a)
    N[j][aj] += 1
    end
    p(j) = SimpleGamePolicy(aj => u/sum(values(N[j])) for (aj, u) in N[j])
    π = [p(j) for j in ℐ]
    πi.πi = best_response(𝒫, π, i)
    end
# -----------------------------------------------