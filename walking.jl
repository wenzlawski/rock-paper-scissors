using Agents
using InteractiveDynamics
using StatsBase
using GLMakie

# # Random walking algorithm simulations
# Using Agents.js

"""Using the Agents.jl package:

1. Define the space the agent will live in.
2. Define the agent type. (@agent)
3. Simulation tuning properties.
4. Functions to govern the time evolution of the ABM.
5. Visualise the model, animate its evolution.
6. Collect data in Vectors.
"""

# We are going to start with uniform random walks
# Starting with a 2D grid random walk.

n_size = 5
n_walkers = 5

# Initialising the random model space and Agents

function initialize()
  space2d = GridSpace((n_size, n_size))
  model = ABM(GridAgent{2}, space2d)
  poss = sample(0:n_size^2-1, n_walkers, replace=false)
  xys = map(p -> (p ÷ n_size + 1, p % n_size + 1), poss)
  for p in xys
    add_agent!(
      p, model
    )
  end
  return model
end

# Step the agent 

a_step!(walker, model) = walk!(walker, rand, model; ifempty=true)

# Initialise the model

model = initialize()

# Visualising the plot

figure, = abmplot(model)
figure

fig, ax, abmobs = abmplot(model;
  (agent_step!)=a_step!)
fig

abmvideo(
  "walking.mp4", model, agent_step!;
  framerate=4, frames=20,
  title="Random walking"
)

