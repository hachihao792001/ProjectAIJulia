# ----------------------------- IteratedBestResponse --------------------------------
# hàm best_response trả về một SimpleGamePolicy với những cặp (action, prob) được chọn theo công thức của best response
function best_response(𝒫::SimpleGame, π, i)
      U(ai) = utility(𝒫, joint(π, SimpleGamePolicy(ai), i), i)
      ai = argmax(U, 𝒫.𝒜[i])
      return SimpleGamePolicy(ai)
end

struct IteratedBestResponse
      k_max # số lần lặp tuần tự
      π # chính sách ban đầu
end
function IteratedBestResponse(𝒫::SimpleGame, k_max)
      # π một danh sách các danh sách SimpleGamePolicy của mỗi joint action trong joint action space của 𝒫
      π = [SimpleGamePolicy(ai => 1.0 for ai in 𝒜i) for 𝒜i in 𝒫.𝒜]
      return IteratedBestResponse(k_max, π)
end
function solve(M::IteratedBestResponse, 𝒫)
      π = M.π
      # lặp k_max lần, mỗi lần lặp dùng lại π cũ để tính danh sách các best_response để gán vào π mới
      for k in 1:M.k_max
            π = [best_response(𝒫, π, i) for i in 𝒫.ℐ]
      end
      return π
end