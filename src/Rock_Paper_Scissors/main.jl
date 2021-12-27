include("./RockPaperScissors.jl")
include("./IteratedBestResponse.jl")
include("./HierarchicalSoftmax.jl")
include("./FictitiousPlay.jl")

using Plots

function main()
    rps = RockPaperScissors()
    simpleGameRPS = SimpleGame(rps)


    IBR = IteratedBestResponse(simpleGameRPS, 100) 
    HS = HierarchicalSoftmax(simpleGameRPS,0.5,10)

    println("Begin solving Iterated Best Response...")
    println(solve(IBR, simpleGameRPS))
    println("Done")

    println("\nBegin solving Hierarchical Softmax...")
    D = solve(HS, simpleGameRPS)
    println("Done")

    println("\nBegin solving Fictitious Play...")
    pi = [(FictitiousPlay(simpleGameRPS,i)) for i in 1:2]
    k_max=100
    v = simulate(simpleGameRPS,pi,k_max)
    println("Done")
   


end

main()





