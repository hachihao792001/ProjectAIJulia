# ----------------------------- IteratedBestResponse --------------------------------
function best_response(𝒫::SimpleGame, π, i)
      U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
      ai = argmax(U, 𝒫.𝒜[i])
      return SimpleGamePolicy(ai)
end

struct IteratedBestResponse
      k_max # number of iterations
      π # initial policy
end
function IteratedBestResponse(𝒫::SimpleGame, k_max)
      π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
      return IteratedBestResponse(k_max, π)
end
function solve(M::IteratedBestResponse, 𝒫)
      π = M.π
      for k in 1:M.k_max
            π = [best_response(𝒫, π, i) for i in 𝒫.ℐ]
      end
      return π
end