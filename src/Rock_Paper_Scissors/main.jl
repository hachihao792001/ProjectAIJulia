
include("./RockPaperScissors.jl")
include("./NashEquilibrium.jl")
include("./HierarchicalSoftmax.jl")
include("./IteratedBestResponse.jl")
include("./FictitiousPlay.jl")


simpleGame = RockPaperScissors()
p = SimpleGame(simpleGame)

HS = HierarchicalSoftmax(p, 0.5, 10)
IBR = IteratedBestResponse(p, 100)
NQ = NashEquilibrium()

println("Begin solving Nash Equilibrium...")
solve(NQ,p)
println("Done")


println("Begin solving Iterated Best Response...")
solve(IBR, p)
println("Done")


# # println("\nBegin solving Hierarchical Softmax...")
# # D = solve(HS, p)
# # println("Done")

println("\nBegin solving Fictitious Play...")
pi = [(FictitiousPlay(p,i)) for i in 1:2]
k_max=100
v = simulate(p,pi,k_max)
println("Done")


