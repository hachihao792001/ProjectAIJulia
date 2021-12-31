# ----------------------------- HierarchicalSoftmax --------------------------------
function softmax_response(𝒫::SimpleGame, π, i, λ)
      𝒜i = 𝒫.𝒜[i]
      U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
      return SimpleGamePolicy(ai => exp(λ * U(ai)) for ai in 𝒜i)
end

struct HierarchicalSoftmax
      λ # precision parameter
      k # level
      π # initial policy
end
function HierarchicalSoftmax(𝒫::SimpleGame, λ, k)
      # π một danh sách các danh sách SimpleGamePolicy của mỗi joint action trong joint action space của 𝒫
      π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
      return HierarchicalSoftmax(λ, k, π)
end
function solve(M::HierarchicalSoftmax, 𝒫)
      π = M.π
      for k in 1:M.k
            π = [softmax_response(𝒫, π, i, M.λ) for i in 𝒫.ℐ]
      end
      return π
end