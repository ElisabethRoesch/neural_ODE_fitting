using DifferentialEquations, Flux, Optim, DiffEqFlux, DiffEqParamEstim, Plots
u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
dudt = Chain(x -> x.^3,
             Dense(2,50,tanh),
             Dense(50,2))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=t)

function predict_n_ode(p)
  n_ode(u0,p)
end

# function loss_n_ode(p)
#     pred = predict_n_ode(p)
#     loss = sum(abs2,ode_data .- pred)
#     loss,pred
# end
#

function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
node_two_stage = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)

loss_n_ode = node_two_stage.cost_function
#loss_n_ode(n_ode.p)
function los(p)
  l = loss_n_ode(p)
  pred = predict_n_ode(p)
  return l,pred
end
los(n_ode.p)
cb = function (p,l,pred) #callback function to observe training
  # display(l)
  # pred = predict_n_ode(p)
  # # plot current prediction against data
  # pl = scatter(t,ode_data[1,:],label="data")
  # scatter!(pl,t,pred[1,:],label="prediction")
  # display(plot(pl))
  print("hi")
end

# Display the ODE with the initial parameter values.
cb(n_ode.p,los(n_ode.p)...)

res1 = DiffEqFlux.sciml_train!(los, n_ode.p, ADAM(0.05), cb = cb, maxiters = 100)
