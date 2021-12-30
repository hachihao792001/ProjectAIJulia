
using LinearAlgebra
using GridInterpolations
using Distributions

struct SetCategorical{S}
      elements::Vector{S} # Set elements (could be repeated)
      distr::Categorical # Categorical distribution over set elements

      function SetCategorical(elements::AbstractVector{S}) where {S}
            weights = ones(length(elements))
            return new{S}(elements, Categorical(normalize(weights, 1)))
      end

      function SetCategorical(elements::AbstractVector{S}, weights::AbstractVector{Float64}) where {S}
            ℓ₁ = norm(weights, 1)
            if ℓ₁ < 1e-6 || isinf(ℓ₁)
                  return SetCategorical(elements)
            end
            distr = Categorical(normalize(weights, 1))
            return new{S}(elements, distr)
      end
end