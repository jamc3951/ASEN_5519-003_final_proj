using DelimitedFiles
using StaticArrays
using Parameters: @with_kw
using POMDPs
using Random

const GWPos = SVector{2,Int}

"""
    SimpleGridWorld(;kwargs...)
Create a simple grid world MDP. Options are specified with keyword arguments.
# States and Actions
The states are represented by 2-element static vectors of integers. Typically any Julia `AbstractVector` e.g. `[x,y]` can also be used for arguments. Actions are the symbols `:up`, `:left`, `:down`, and `:right`.
# Keyword Arguments
- `size::Tuple{Int, Int}`: Number of cells in the x and y direction [default: `(10,10)`]
- `rewards::Dict`: Dictionary mapping cells to the reward in that cell, e.g. `Dict([1,2]=>10.0)`. Default reward for unlisted cells is 0.0
- `terminate_from::Set`: Set of cells from which the problem will terminate. Note that these states are not themselves terminal, but from these states, the next transition will be to a terminal state. [default: `Set(keys(rewards))`]
- `tprob::Float64`: Probability of a successful transition in the direction specified by the action. The remaining probability is divided between the other neighbors. [default: `0.7`]
- `discount::Float64`: Discount factor [default: `0.95`]
"""

map_ = readdlm("hazards/haz1_new.txt", ',', Float64)
d = Dict{GWPos, Float64}()
g = Dict{GWPos, Float64}()
for i = 1:(20*20)
    row = div(i, 20)
    col = i % 20
    d[GWPos(row,col)] = -98 * map_[i][1] -2
end
d[GWPos(2,9)] = 200
g[GWPos(2,9)] = 200
@with_kw struct SimpleGridWorld <: MDP{GWPos, Symbol}
    size::Tuple{Int, Int}           = (20,20)
    rewards::Dict{GWPos, Float64}   = d
    terminate_from::Set{GWPos}      = Set(keys(g))
    tprob::Float64                  = 0.92
    discount::Float64               = 0.95
end


# States

function POMDPs.states(mdp::SimpleGridWorld)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]])
    push!(ss, GWPos(-1,-1))
    return ss
end

function POMDPs.stateindex(mdp::SimpleGridWorld, s::AbstractVector{Int})
    if all(s.>0)
        return LinearIndices(mdp.size)[s...]
    else
        return prod(mdp.size) + 1 # TODO: Change
    end
end

struct GWUniform
    size::Tuple{Int, Int}
end
Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2]))
function POMDPs.pdf(d::GWUniform, s::GWPos)
    if all(1 .<= s[1] .<= d.size)
        return 1/prod(d.size)
    else
        return 0.0
    end
end
POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2])

POMDPs.initialstate(mdp::SimpleGridWorld) = GWUniform(mdp.size)

# Actions

POMDPs.actions(mdp::SimpleGridWorld) = (:up, :upRight, :upLeft, :down, :downLeft, :downRight,:left, :right)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box


const dir = Dict(:up=>GWPos(0,1), :upRight => GWPos(1,1), :upLeft => GWPos(-1,1),:down=>GWPos(0,-1), :downLeft=>GWPos(-1,-1),:downRight=>GWPos(1,-1),:left=>GWPos(-1,0), :right=>GWPos(1,0))
const aind = Dict(:up => 1, :upRight => 2, :upLeft => 8, :down => 5, :downLeft => 6, :downRight => 4,:left => 7, :right => 3)

POMDPs.actionindex(mdp::SimpleGridWorld, a::Symbol) = aind[a]


# Transitions

POMDPs.isterminal(m::SimpleGridWorld, s::AbstractVector{Int}) = any(s.<0)

function POMDPs.transition(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s)
        return Deterministic(GWPos(-1,-1))
    end

    destinations = MVector{length(actions(mdp))+1, GWPos}(undef)
    destinations[1] = s

    probs = @MVector(zeros(length(actions(mdp))+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(length(actions(mdp)) - 1) # probability of transitioning to another cell
        end

        dest = s + dir[act]
        destinations[i+1] = dest

        if !inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
            destinations[i+1] = GWPos(-1, -1) # dest was out of bounds - this will have probability zero, but it should be a valid state
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs)
end

function inbounds(m::SimpleGridWorld, s::AbstractVector{Int})
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2]
end

# Rewards

POMDPs.reward(mdp::SimpleGridWorld, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::SimpleGridWorld, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)


# discount

POMDPs.discount(mdp::SimpleGridWorld) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end

# deprecated in POMDPs v0.9
POMDPs.initialstate_distribution(mdp::SimpleGridWorld) = GWUniform(mdp.size)
