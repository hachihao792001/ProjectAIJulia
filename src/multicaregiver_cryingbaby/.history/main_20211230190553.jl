include("multicaregiver.jl")
include("nashequilibrium.jl")
include("dynamicprogramming.jl")

baby=MultiCaregiverCryingBaby()
pomg=POMG(baby)

# NashEquilibrium
nash = POMGNashEquilibrium(rand(Float64,2),2)
result = solve(nash,pomg)
print("Result: ----------------------------------------")
print(result)
# #dynamicprogramming
dyn = POMGDynamicProgramming(rand(Float64, 2),2)
resultOptimize = solveDP(dyn,pomg)
print("Result Optimize: ----------------------------------------")
print(resultOptimize)