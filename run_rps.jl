includet("rps.jl")


# Initialise the model
model = initialise(;
  rps=(20, 20, 20),
  speed=0.7,
  cohere_factor=0.01,
  separation=7.0,
  separate_factor=0.25,
  match_factor=0.05,
  flee_factor=0.1,
  agility=0.3,
  visual_distance=10.0,
  cluster_size=20,
  extent=(100, 100)
)

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

# Interactive Plotting

rock(a) = a.hand isa Rock
paper(a) = a.hand isa Paper
scissors(a) = a.hand isa Scissors
adata = [(rock, count), (paper, count), (scissors, count)]

fig, abmobs = abmexploration(model;
  (agent_step!)=agent_step!, (model_step!)=model_step!, ac=ac, am=am, as=20,
  adata, alabels=["Rock", "Paper", "Scissors"],
  plotkwargs=(; markersize=0)
)
fig

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
