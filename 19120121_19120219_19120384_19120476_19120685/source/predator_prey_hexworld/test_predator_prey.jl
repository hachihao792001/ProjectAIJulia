include("./predator_prey.jl")

using DecisionMakingProblems
using Test

const p = DecisionMakingProblems

@testset "predator_prey.jl" begin
    m = PredatorPreyHexWorld() 
    hexes = m.hexes
    @test p.n_agents(m) == 2
    @test length(p.ordered_states(m, rand(1:2))) == length(hexes) && length(p.ordered_states(m)) == length(hexes)^2
    @test length(p.ordered_actions(m, rand(1:2))) == 6 && length(p.ordered_joint_actions(m)) == 36
    @test p.n_actions(m, rand(1:2)) == 6 && p.n_joint_actions(m) == 36

    @test 0.0 <= p.transition(m, rand(p.ordered_states(m)), rand(p.ordered_joint_actions(m)), rand(p.ordered_states(m))) <= 1.0
    @test -1.0 <= p.reward(m, rand(1:2), rand(p.ordered_states(m)), rand(p.ordered_joint_actions(m))) <= 10.0
    @test [-1.0, -1.0] <= p.joint_reward(m, rand(p.ordered_states(m)), rand(p.ordered_joint_actions(m))) <= [10.0, 10.0]
    mg = p.MG(m)
end