include("mdp.jl")
include("discreteMDP.jl")

function hex_neighbors(hex::Tuple{Int,Int})  #hex có kiểu tuple như hex=(2,3)  i,j=hex => i =2,j=3 
    i, j = hex                               # destructor các giá trị trong value ra
    [(i + 1, j), (i, j + 1), (i - 1, j + 1), (i - 1, j), (i, j - 1), (i + 1, j - 1)]  # trả về (return) các giá trị có thể đi tiếp
    
end
 
# i=1 ,j=1
# (2,1) (1,2) (0,2) (0,1) (1,0) (2,0)
struct HexWorldMDP
    # Problem has |hexes| + 1 states, where last state is consuming.
    hexes::Vector{Tuple{Int,Int}}
    #discreateMDP (each cell)
    mdp::DiscreteMDP
    # The special hex rewards used to construct the MDP
    special_hex_rewards::Dict{Tuple{Int,Int},Float64}

    function HexWorldMDP(
        hexes::Vector{Tuple{Int,Int}}, #set of states [(a_1,b_2),(a_2,b_2)]
        r_bump_border::Float64,         #actions rewards
        p_intended::Float64,            # probability [0,1]
        special_hex_rewards::Dict{Tuple{Int,Int},Float64},  #reward
        γ::Float64,     #discunted rate [0,1]
    )

        nS = length(hexes) + 1 #number of state (last state is consuming state)
        nA = 6   #number of actions, In this problems we have six directions
        s_absorbing = nS # terminal state

        T = zeros(Float64, nS, nA, nS)  #init transition (probability) [0,1]
        R = zeros(Float64, nS, nA)      #action [1,6] because we have 6 choice to go

        p_veer = (1.0 - p_intended) / 2 # probability of veering left or right

        for s in 1 : length(hexes)
            hex = hexes[s]
            if !haskey(special_hex_rewards, hex)  #kiểm tra xem trong các states đã đi có key hex hay chưa
                # Action taken from a normal tile
                neighbors = hex_neighbors(hex)  #trả về 1 mảng 6 vị trí lân cận
                for (a,neigh) in enumerate(neighbors)  # a là index được đánh số từ 1, neigh là cặp các tupel tương ứng (i,j+1),...
                    # Indended transition.
                    s′ = findfirst(h -> h == neigh, hexes) #s' là state ở (t+1) #findFirst trả về index đầu tiên == neigh trong tập hexes
                    if s′ == nothing  # TH đụng tường ko đi được nữa
                        # Off the map!
                        s′ = s
                        R[s,a] += r_bump_border*p_intended
                    end
                    T[s,a,s′] += p_intended  #tính tỷ lệ đi đến state 

                    # Unintended veer left.
                    a_left = mod1(a+1, nA)
                    neigh_left = neighbors[a_left]
                    s′ = findfirst(h -> h == neigh_left, hexes)
                    if s′ == nothing
                        # Off the map!
                        s′ = s
                        R[s,a] += r_bump_border*p_veer
                    end
                    T[s,a,s′] += p_veer
                   
                    # Unintended veer right.
                    a_right = mod1(a-1, nA) # 0%6=6
                    neigh_right = neighbors[a_right]
                    s′ = findfirst(h -> h == neigh_right, hexes)
                    if s′ == nothing
                        # Off the map!
                        s′ = s
                        R[s,a] += r_bump_border*p_veer
                    end
                    T[s,a,s′] += p_veer
                   
                end
            else # nếu mảng Reward đã chứa key hex
                # Action taken from an absorbing hex
                # In absorbing hex, your action automatically takes you to the absorbing state and you get the reward.
                for a in 1 : nA
                    T[s,a,s_absorbing] = 1.0  #get the reward and can return to the terminal state
                    R[s,a] += special_hex_rewards[hex]
                end
            end
        end

        # Absorbing state stays where it is and gets no reward.
        for a in 1:nA
            T[s_absorbing, a, s_absorbing] = 1.0  #terminal state
        end

        mdp = DiscreteMDP(T, R, γ)  #init the MDPdiscrete which is defined in discreteMDP.jl file

        return new(hexes, mdp, special_hex_rewards)
    end
end

const HexWorldRBumpBorder = -1.0 # Reward for falling off hex map -> (bounded by the border)
const HexWorldPIntended = 0.7 # Probability of going intended direction (go the right direction)
const HexWorldDiscountFactor = 0.9  #--> discounted rate (γ)

# this function is used to init a hexworld with rewards
function HexWorld()
    HexWorld = HexWorldMDP(
        [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (-1, 2),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
            (8, 2), (4, 1), (5, 0), (6, 0), (7, 0), (7, 1), (8, 1), (9, 0)],
        HexWorldRBumpBorder,
        HexWorldPIntended,
        Dict{Tuple{Int,Int},Float64}(
            (0, 1) => 5.0,
            (2, 0) => -10.0,
            (9, 0) => 10.0,
        ),
        HexWorldDiscountFactor
    )
    return HexWorld
end

function createMDP(mdp::HexWorldMDP) #hàm khởi tạo 1 MDP
    return MDP(mdp.mdp.T, mdp.mdp.R, mdp.mdp.γ)
end



