include("hexWorld.jl")
include("mdp.jl")
include("Policy.jl")
include("visual.jl")


k_max=100 #số vòng lặp
hexworld=HexWorld() #khởi tạo HexworldMDP
mdp=createMDP(hexworld) #ép kiểu về 1 MDP

U = [0.0 for s in 1:25] #khởi tạo mảng U ban đầu là 0 hết để khởi tạo policy

policy=ValueFunctionPolicy(mdp,U)  #khởi tạo biến policy
actions=Solve(mdp,policy,k_max) #kết quả trả về 1 mảng actions 

for i in 1:length(hexworld.hexes)
    state=hexworld.hexes[i]
    action=actions[i]
    println("state : $state <==> action :$action") #in ra kết quả
end


