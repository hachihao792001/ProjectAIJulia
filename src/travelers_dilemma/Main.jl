import Pkg
using LinearAlgebra
using Distributions
using Plots

struct SimpleGame
      γ  # discount factor
      ℐ  # agents
      𝒜  # joint action space
      R  # joint reward function
end

struct Travelers end

n_agents(simpleGame::Travelers) = 2

ordered_actions(simpleGame::Travelers, i::Int) = 2:100
ordered_joint_actions(simpleGame::Travelers) = vec(collect(Iterators.product([ordered_actions(simpleGame, i) for i in 1:n_agents(simpleGame)]...)))

n_joint_actions(simpleGame::Travelers) = length(ordered_joint_actions(simpleGame))
n_actions(simpleGame::Travelers, i::Int) = length(ordered_actions(simpleGame, i))

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


function reward(simpleGame::Travelers, i::Int, a)
      if i == 1
            noti = 2
      else
            noti = 1
      end
      if a[i] == a[noti]
            r = a[i]
      elseif a[i] < a[noti]
            r = a[i] + 2
      else
            r = a[noti] - 1
      end
      return r
end

function joint_reward(simpleGame::Travelers, a)
      return [reward(simpleGame, i, a) for i in 1:n_agents(simpleGame)]
end

function SimpleGame(simpleGame::Travelers)
      return SimpleGame(
            0.9,
            vec(collect(1:n_agents(simpleGame))),
            [ordered_actions(simpleGame, i) for i in 1:n_agents(simpleGame)],
            (a) -> joint_reward(simpleGame, a)
      )
end

struct SimpleGamePolicy
      p # dictionary mapping actions to probabilities
      function SimpleGamePolicy(p::Base.Generator)
            return SimpleGamePolicy(Dict(p))
      end
      function SimpleGamePolicy(p::Dict)
            vs = collect(values(p))
            vs ./= sum(vs)
            return new(Dict(k => v for (k, v) in zip(keys(p), vs)))
      end
      SimpleGamePolicy(ai) = new(Dict(ai => 1.0))
end

(πi::SimpleGamePolicy)(ai) = get(πi.p, ai, 0.0)
function (πi::SimpleGamePolicy)()
      D = SetCategorical(collect(keys(πi.p)), collect(values(πi.p)))
      return rand(D)
end

joint(X) = vec(collect(Iterators.product(X...)))
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

function utility(𝒫::SimpleGame, π, i)
      𝒜, R = 𝒫.𝒜, 𝒫.R
      p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
      return sum(R(a)[i] * p(a) for a in joint(𝒜))
end



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

simpleGame = Travelers()
p = SimpleGame(simpleGame)

HS = HierarchicalSoftmax(p, 0.5, 10)
IBR = IteratedBestResponse(p, 100)

print("Begin solving IBR...")
π = solve(IBR, p)
print("\nNash equilibrium: ")
print(keys(π[1].p), keys(π[2].p))

print("Begin solving HS")
D = solve(HS, p)
bar(collect(keys(D[1].p)), collect(values(D[1].p)), orientation=:vertical)