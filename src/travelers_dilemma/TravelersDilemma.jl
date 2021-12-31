include("./SimpleGamePolicy.jl")

struct Travelers end

function SimpleGame(simpleGame::Travelers)
      # tạo 1 SimpleGame có discount factor là 0.9 (xác suất để vòng lặp chạy tiếp)
      # agent là 1, 2
      # joint action space là 1 list gồm 2 mảng [2:100] (tất cả mọi action mà 2 agent có thể thực hiện)
      # joint reward function là 1 lamda expression trả về kết quả của hàm joint_reward
      return SimpleGame(
            0.9,
            vec(collect(1:n_agents(simpleGame))),
            [ordered_actions(simpleGame, i) for i in 1:n_agents(simpleGame)],
            (a) -> joint_reward(simpleGame, a)
      )
end

# khai báo số agent trong bài toán
n_agents(simpleGame::Travelers) = 2
# khai báo các action mà agent có thể thực hiện
ordered_actions(simpleGame::Travelers, i::Int) = 2:100

# hàm tính phần thưởng khi 1 agent có số thứ tự i thực hiện action a
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

# hàm tính và trả về 1 list các phần thưởng tương ứng Với từng agent với action a
function joint_reward(simpleGame::Travelers, a)
      return [reward(simpleGame, i, a) for i in 1:n_agents(simpleGame)]
end

# phân phối từng mảng trong X với nhau
joint(X) = vec(collect(Iterators.product(X...)))
# thay thế phần tử vị trí i trong π thành πi 
joint(π, πi, i) = [i == j ? πi : πj for (j, πj) in enumerate(π)]

# hàm tính sự "có lợi" của một joint policy π
function utility(𝒫::SimpleGame, π, i)
      𝒜, R = 𝒫.𝒜, 𝒫.R
      # p(a) tính tích của tất cả các xác suất của các action trong joint action a
      p(a) = prod(πj(aj) for (πj, aj) in zip(π, a))
      # tính tổng mọi (phần thưởng của agent số thứ tự i * tích xác suất) của mọi joint action a trong joint action space 𝒜
      return sum(R(a)[i] * p(a) for a in joint(𝒜))
end