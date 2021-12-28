# include("conditionalplan.jl")
# include("multicaregiver.jl")
# include("nashequilibrium.jl")
# include("dynamicprogramming.jl")
# include("pomg.jl")

# baby=MultiCaregiverCryingBaby()
# pomg=POMG(baby)

# #NashEquilibrium
# nash = POMGNashEquilibrium(rand(Float64, 3),2)
# solve(nash,pomg)

# #dynamicprogramming
# dyn = POMGDynamicProgramming([0.5,0.5,0.5],2)
# solveDP(dyn,pomg)