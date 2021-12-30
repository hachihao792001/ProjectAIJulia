include("./SimpleGamePolicy.jl")

struct Travelers end

function SimpleGame(simpleGame::Travelers)
      return SimpleGame(
            0.9,
            vec(collect(1:n_agents(simpleGame))),
            [ordered_actions(simpleGame, i) for i in 1:n_agents(simpleGame)],
            (a) -> joint_reward(simpleGame, a)
      )
end

n_agents(simpleGame::Travelers) = 2
ordered_actions(simpleGame::Travelers, i::Int) = 2:100

function reward(simpleGame::Travelers, i::Int, a)
      if i == 1
            notI = 2
      else
            notI = 1
      end
      if a[i] == a[notI]
            r = a[i]
      elseif a[i] < a[notI]
            r = a[i] + 2
      else
            r = a[notI] - 1
      end
      return r
end

function joint_reward(simpleGame::Travelers, a)
      return [reward(simpleGame, i, a) for i in 1:n_agents(simpleGame)]
end

joint(X) = vec(collect(Iterators.product(X...)))
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

function utility(𝒫::SimpleGame, π, i)
      𝒜, R = 𝒫.𝒜, 𝒫.R
      p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
      return sum(R(a)[i] * p(a) for a in joint(𝒜))
end