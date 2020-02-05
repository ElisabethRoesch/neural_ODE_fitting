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

function loss_n_ode(p)
    pred = predict_n_ode(p)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

loss_n_ode(n_ode.p) # n_ode.p stores the initial parameters of the neural ODE


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
cb(n_ode.p,loss_n_ode(n_ode.p)...)

res1 = DiffEqFlux.sciml_train!(loss_n_ode, n_ode.p, ADAM(0.05), cb = cb, maxiters = 100)





function dudt(du,u,p,t)
  du[1]= 1.1*u[1]
  du[2]= 1.1*u[2]
  return du
