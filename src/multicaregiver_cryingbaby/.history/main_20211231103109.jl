include("multicaregiver.jl")
include("nashequilibrium.jl")
include("dynamicprogramming.jl")

baby=MultiCaregiverCryingBaby()
pomg=POMG(baby)

# 1: Giai quyết theo NashEquilibrium của POMG
nash = POMGNashEquilibrium(rand(Float64,2),2)
result = solve(nash,pomg)
print("-----------------------Result of POMG NashEquilibrium: -----------------------")
print(result)

# 2: Giai quyết theo cách sử dụng Dynamic programming
dyn = POMGDynamicProgramming(rand(Float64, 2),2)
resultOptimize = solveDP(dyn,pomg)
print("-----------------------Result of POMG DynamicProgramming: -----------------------")
print(resultOptimize)