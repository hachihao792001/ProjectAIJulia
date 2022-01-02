include("./SimpleGame.jl")
include("./SetCategorical.jl")

struct SimpleGamePolicy
      p # dictionary bắt cặp actions và probabilities
      function SimpleGamePolicy(p::Base.Generator)
            return SimpleGamePolicy(Dict(p))
      end
      # hàm lấy vào 1 dictionary các cặp (action, prob), tính phần trăm các prob trong tổng số prob,
      # và gán lại phần trăm đó vào trong từ cặp, rồi trả về lại dictionary mới
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