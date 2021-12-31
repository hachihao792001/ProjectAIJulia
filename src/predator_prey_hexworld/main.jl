include("./predator_prey.jl")

using DecisionMakingProblems

const p = DecisionMakingProblems

m = PredatorPreyHexWorld() 

hexes = m.hexes
println("hexes: ", hexes)

p.n_agents(m)
println("n_agents: ", p.n_agents(m))

println("Ordered states: ", p.ordered_states(m, rand(1:2)))

println("Ordered states (only m): ", p.ordered_states(m))

println("ordered_actions: ", p.ordered_actions(m, rand(1:2)))

println("ordered_joint_actions: ", p.ordered_joint_actions(m))

p.n_actions(m, rand(1:2))
println("n_actions: ", p.n_actions(m, rand(1:2)))

p.n_joint_actions(m)
println("n_joint_actions: ", p.n_joint_actions(m))


println("transition: ", p.transition(m, rand(p.ordered_states(m)), rand(p.ordered_joint_actions(m)), rand(p.ordered_states(m))))

println("reward: ", p.reward(m, rand(1:2), rand(p.ordered_states(m)), rand(p.ordered_joint_actions(m))))

println("joint_reward: ", p.joint_reward(m, rand(p.ordered_states(m)), rand(p.ordered_joint_actions(m))))

mg = p.MG(m)
println("MG: ", mg)