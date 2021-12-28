
using Pkg
Pkg.add("Parameters")
using Parameters: @with_kw

@with_kw struct CryingBaby
    # hungry reward khi HUNGRY độc lập với các action khác 
    r_hungry::Float64 = -10.0
    # feed reward
    r_feed::Float64 = -5.0 
    # sing reward
    r_sing::Float64 = -0.5

    # Tỷ lệ sẽ trở nên đói dần = 10%
    p_become_hungry::Float64 = 0.1
    # Tỷ lệ CRYING khi HUNGRY = 80%
    p_cry_when_hungry::Float64 = 0.8
    # Tỷ lệ CRYING khi SATED = 10%
    p_cry_when_not_hungry::Float64 = 0.1
    # Tỷ lệ CRYING khi HUNGRY + SING = 90%
    p_cry_when_hungry_in_sing::Float64 = 0.9
    γ::Float64 = 0.9
end