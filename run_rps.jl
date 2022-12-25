includet("rps.jl")

# Initialise the model
imodel() = initialise(;
  rps=(10, 10, 10),
  speed=0.7,
  base_speed=0.5,
  cohere_factor=0.01,
  separation=7.0,
  separate_factor=0.25,
  match_factor=0.05,
  flee_factor=0.1,
  agility=0.3,
  visual_distance=10.0,
  cluster_size=20,
  extent=(100, 100),
  periodic=false
)

model = imodel()
# Static step
step!(model, agent_step!, model_step!)

# Plot static
figure, = abmplot(model; am=am, ac=ac, as=30)
figure

# Video
abmvideo(
  "rps.mp4", model, agent_step!;
  am=am, ac=ac,
  framerate=20, frames=100,
  title="RPS"
)

# Interactive
function run_sim(model)
  rock(a) = hand(a) isa Rock
  paper(a) = hand(a) isa Paper
  scissors(a) = hand(a) isa Scissors
  adata = [(rock, count), (paper, count), (scissors, count)]

  properties = Dict(
    :speed => 0:0.01:1,
    :base_speed => 0:0.01:1,
    :cohere_factor => 0:0.01:1,
    :separation => 0:0.1:10,
    :separate_factor => 0:0.01:1,
    :match_factor => 0:0.01:1,
    :flee_factor => 0:0.01:1,
    :agility => 0:0.01:1,
    :visual_distance => 0:1:20
  )
  fontsize_theme = Theme(fontsize=40)
  set_theme!(fontsize_theme)

  fig, abmobs = abmexploration(model;
    (agent_step!)=agent_step!, (model_step!)=model_step!, ac=ac, am=am, as=40,
    params=properties,
    adata, alabels=["Rock", "Paper", "Scissors"],
    plotkwargs=(; markersize=0)
  )
  fig
end

model = imodel()
run_sim(model)


# Static testing
function initialise_test()
  space = ContinuousSpace((20, 20))
  model = ABM(Hand, space)
  return model
end

vel = Tuple(rand(model.rng, 2) * 2 .- 1)
speed = 0.5
visual_distance = 5.0

model = initialise_test()

add_agent!((10, 10), Rock, model, vel, speed, visual_distance)
add_agent!((10, 10), Paper, model, vel, speed, visual_distance)

model.agents

close = interacting_pairs(model, 20, :types)
close

for pair in close
  @info "Pair $pair is close"
  op = overpower(pair...)
  if isnothing(op)
    continue
  end
  a, b = pair
  change, other = op == 1 ? (a, b) : (b, a)
  @info "Changed agent $change to $other"
  add_agent!(change.pos, typeof(other), model, change.vel, change.speed, change.visual_distance)
  kill_agent!(change, model)
end

model.agents
