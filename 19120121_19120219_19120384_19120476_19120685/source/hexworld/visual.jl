include("hexWorld.jl")

#uncomment to use
# using Pkg
# Pkg.add("Luxor")
using Luxor

#hàm tô màu cho từng cell
function drawColor(r::Int, g::Int, b::Int)
    res = "#"
    hexchar = ["A", "B", "C", "D", "E", "F"]
    for cl in (r,g,b)
        for it in (div(cl,16) , cl%16)
            if it > 9
                res *= hexchar[it-9]
            else
                res *= string(it)
            end
        end
    end
    return res
end


#hàm visualise
function visualiseHexworld(filename::String, hexes::Vector{Tuple{Int, Int}}, actions::Vector{Int}, U::Vector{Float64})
    Drawing(600,600, filename)

    radius = 30
    num_cell_a_line = 15

    grid_point = [[Point(0,0) for j in 1:100 + num_cell_a_line] for i in 1:100 + 2*num_cell_a_line]
    grid = GridHex(O, radius, 770)
    for idx in 1:300
        p = nextgridpoint(grid)

        j = mod1(idx, num_cell_a_line)
        i = Int((idx - j) / num_cell_a_line)
        j -= Int((i-i%2)/2)
        j -= 3
        i -= 3

        grid_point[i+100][j+100] = p
    end

    
    for idx in 1:length(hexes)
        (i,j) = hexes[idx]
        p = grid_point[i+100][j+100]
        r=255
        g=255
        b=255
        deviant = 25
        if U[idx] < 0
            g -= round(Int, deviant*(-U[idx]))
            b -= round(Int, deviant*(-U[idx]))
        end
        if U[idx] > 0
            r -= round(Int, deviant*U[idx])
            g -= round(Int, deviant*U[idx])
        end
        sethue(drawColor(r,g,b))
        ngon(p, radius-5, 6, pi/2, :fillstroke)  #vẽ grid cell

        if actions[idx] >= 1 && actions[idx] <= 6
            (ni,nj) = hex_neighbors(hexes[idx])[actions[idx]]
            next = grid_point[ni+100][nj+100]
        end

        #vẽ mũi tên
        sethue("#123456")
        arrow(p, between(p, next, 0.4), arrowheadlength=10, arrowheadangle=pi/4, linewidth=0)

        #vẽ tọa độ của state
        sethue("black")
        fontsize(10)
        text("("*string(i) * "," * string(j)*")", p, halign=:center, valign=:baseline)
    end

    finish()
    preview()
end