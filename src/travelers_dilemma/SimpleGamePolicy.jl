include("./SimpleGame.jl")
include("./SetCategorical.jl")

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