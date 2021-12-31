include("pomg.jl")
include("nashequilibrium.jl")
include("dynamicprogramming.jl")

# Tạo 1 baby với các thuộc tính đã có sẵn và chuyển baby về dạng POMG
baby=MultiCaregiverCryingBaby()
pomg=POMG(baby)

# 1: Giải quyết theo NashEquilibrium của POMG
nash = POMGNashEquilibrium(rand(Float64,2),2)
result = solve(nash,pomg)
print("-----------------------Result of POMG NashEquilibrium: -----------------------")
print(result)

# 2: Giải quyết theo cách sử dụng Dynamic programming
dyn = POMGDynamicProgramming(rand(Float64, 2),2)
resultOptimize = solveDP(dyn,pomg)
print("-----------------------Result of POMG DynamicProgramming: -----------------------")
print(resultOptimize)