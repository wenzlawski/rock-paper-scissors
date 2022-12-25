# jps.jl

# # Rock paper scissors simulation in Agents and Makie

using Agents
using Random
using Distributions
using InteractiveDynamics
using GLMakie
using DrWatson: @dict
using LinearAlgebra

# RPS types
abstract type AbstractHand end
struct Rock <: AbstractHand end
struct Paper <: AbstractHand end
struct Scissors <: AbstractHand end

# Make the agent
@agent Hand ContinuousAgent{2} begin
  hand::AbstractHand
  speed::Float64
end

hand(agent::Hand) = agent.hand

# Overpower functions for RPS
overpower(a::Hand, b::Hand) = overpower(a.hand, b.hand)
overpower(_, _) = nothing
overpower(::Rock, ::Scissors) = 2
overpower(::Rock, ::Paper) = 1
overpower(::Paper, ::Scissors) = 1
overpower(::Paper, ::Rock) = 2
overpower(::Scissors, ::Rock) = 1
overpower(::Scissors, ::Paper) = 2

# Attraction values
attract(::Rock) = Scissors
attract(::Paper) = Rock
attract(::Scissors) = Paper

# Visualisation shapes and colors
am(agent::Hand) = am(agent.hand)
am(::Rock) = '⚫'
am(::Paper) = '✉'
am(::Scissors) = '✂'

# Colors
ac(agent::Hand) = ac(agent.hand)
ac(::Rock) = "#32a852"
ac(::Paper) = "#a84432"
ac(::Scissors) = "#323ca8"

# Reflect an agent at the border
function atborder(agent, dims)
  x, y = round.(agent.pos)
  if (x == 0 || x == dims[1])
    return (-1, 1)
  elseif (y == 0 || y == dims[2])
    return (1, -1)
  end
  return nothing
end

rand((1, -1))

# Model initialisation function
function initialise(;
  rps=(10, 10, 10),
  speed=1.0,
  base_speed=0.5,
  cohere_factor=0.1,
  separation=4.0,
  separate_factor=0.25,
  match_factor=0.01,
  flee_factor=0.1,
  agility=0.7,
  visual_distance=5.0,
  cluster_size=30,
  extent=(100, 100),
  periodic=true
)
  properties = @dict(
    speed,
    base_speed,
    cohere_factor,
    separation,
    separate_factor,
    match_factor,
    flee_factor,
    agility,
    visual_distance,
    periodic
  )

  # Initialise model 
  space = ContinuousSpace(extent; periodic=periodic)
  model = ABM(Hand, space; properties)

  # Add agents
  for (qty, typ) in zip(rps, (Rock, Paper, Scissors))
    center = cluster_size .+ (extent[1] - cluster_size) .* rand.((Float64, Float64))
    for _ in 1:qty
      v1 = rand()
      v2 = rand((1, -1)) * 1 - v1
      v1 *= rand((1, -1))
      vel = (v1, v2)
      add_agent!(
        center .- (rand.((Float64, Float64)) .* cluster_size),
        Hand,
        model,
        vel,
        typ(),
        speed,
      )
    end
  end

  return model
end


# Model step function
function model_step!(model)
  close = interacting_pairs(model, 2, :all)
  for (a, b) in close
    op = overpower(a, b)
    if isnothing(op)
      continue
    end
    change, other = op == 1 ? (a, b) : (b, a)
    change.hand = deepcopy(other.hand)
  end

  # calculate speeds
  r = p = s = 1
  ags = collect(allagents(model))
  Threads.@threads for i in eachindex(ags)
    a = ags[i]
    if hand(a) isa Rock
      r += 1
    elseif hand(a) isa Paper
      p += 1
    else
      s += 1
    end
  end
  speeds = normalize(sqrt.(1 ./ [r, p, s]))
  Threads.@threads for i in eachindex(ags)
    a = ags[i]
    if hand(a) isa Rock
      a.speed = speeds[1] * model.speed + model.base_speed
    elseif hand(a) isa Paper
      a.speed = speeds[2] * model.speed + model.base_speed
    else
      a.speed = speeds[3] * model.speed + model.base_speed
    end
  end
end



# Agent step function
function agent_step!(agent, model)
  # check for agent border collision
  # chasing logic
  agent.vel = agent.vel .+ tuple(rand(Normal(0.0, 0.1), 2)...)
  close = nearby_ids_exact(agent, model, model.visual_distance)
  target = nothing
  cohere = flee = separate = match = (0.0, 0.0)
  Ns = Nf = 0
  for id in close
    if hand(model[id]) isa typeof(hand(agent))
      # Separation
      Ns += 1
      heading = model[id].pos .- agent.pos
      cohere = cohere .+ heading
      if euclidean_distance(agent, model[id], model) < model.separation
        separate = separate .- heading
      end
      match = match .+ model[id].vel
    elseif overpower(hand(agent), hand(model[id])) == 1
      # Fleeing
      Nf += 1
      heading = model[id].pos .- agent.pos
      flee = flee .+ (heading .* -1)
    end

    # Following
    if isnothing(target) && hand(model[id]) isa attract(agent.hand)
      target = model[id]
    end
  end
  Ns = max(Ns, 1)
  Nf = max(Nf, 1)
  cohere = cohere ./ Ns .* model.cohere_factor
  separate = separate ./ Ns .* model.separate_factor
  match = match ./ Ns .* model.match_factor
  flee = flee ./ Nf .* model.flee_factor

  if !isnothing(target)
    # lead the agent in the target direction
    head = (target.pos .- agent.pos) .* model.agility
    mvel = target.vel .* 0.1
    agent.vel = (agent.vel .+ head .+ mvel .+ separate .+ match .+ cohere .+ flee) ./ 2
    agent.vel = agent.vel ./ norm(agent.vel)
  else
    agent.vel = (agent.vel .+ separate .+ match .+ cohere .+ flee) ./ 2
    agent.vel = agent.vel ./ norm(agent.vel)
  end

  # check border
  if !model.periodic
    border = atborder(agent, spacesize(model))
    if !isnothing(border)
      agent.vel = agent.vel .* border
    end
  end
  move_agent!(agent, model, agent.speed)
end
