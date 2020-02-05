
using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim
u0 = Float32[1.5; 0.]
datasize = 200
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u.^3)'true_A)'
end
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
dudt = Chain((x->x.^3),
        Dense(2,50,tanh),
       Dense(50,2))
ps = Flux.params(dudt)
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
two_stage_loss_fct()=loss_n_ode.cost_function(ps)
esti =loss_n_ode.estimated_solution
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 1500
verify = 50
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.001)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
cb1 = function ()
    println("hi")
end
@time Flux.train!(L2_loss_fct, ps, data1, opt1, cb = cb1)
