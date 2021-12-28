include("hexWorld.jl")
include("mdp.jl")
include("Policy.jl")


k_max=100
m=HexWorld()
display(m.mdp.T)
# println("==============Reward===========")
# display(m.mdp.R)
mdp=createMDP(m)

U = [0.0 for s in 1:25]
policy=ValueFunctionPolicy(mdp,U)
actions=Solve(mdp,policy,k_max)

for i in 1:length(m.hexes)
    state=m.hexes[i]
    action=actions[i]
    println("state : $state <==> action :$action")
end


