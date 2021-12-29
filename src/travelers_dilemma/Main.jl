import Pkg
using Plots
include("./TravelersDilemma.jl")
include("./HierarchicalSoftmax.jl")
include("./IteratedBestResponse.jl")


simpleGame = Travelers()
p = SimpleGame(simpleGame)

HS = HierarchicalSoftmax(p, 0.5, 10)
IBR = IteratedBestResponse(p, 100)

println("Begin solving Iterated Best Response...")
π = solve(IBR, p)
println("Done")
print("Nash equilibrium: ")
println(keys(π[1].p), keys(π[2].p))

println("\nBegin solving Hierarchical Softmax...")
D = solve(HS, p)
println("Done")
for i = 2:100
      println("$i : $(D[1].p[i])")
end
bar(collect(keys(D[1].p)), collect(values(D[1].p)), orientation = :vertical, legend = false)
