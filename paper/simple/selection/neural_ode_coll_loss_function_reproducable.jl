
using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @save
using Flux: glorot_uniform
mutable struct saver
    losses::Array{Float64,1}
    l2s::Array{Float64,1}
    times::Array{Dates.Time,1}
    count_epochs::Int128
end
function saver(n_epochs)
    losses = zeros(n_epochs)
    l2s = zeros(n_epochs)
    times = fill(Dates.Time(Dates.now()),n_epochs)
    count_epochs = 0
    return saver(losses,l2s,times,count_epochs)
end
function update_saver(saver, loss_i, l2_i, time_i)
    epoch_i = saver.count_epochs
    saver.losses[epoch_i] = loss_i
    saver.l2s[epoch_i] = l2_i
    saver.times[epoch_i] = time_i
end
u0 = Float32[1.5; 0.]
datasize = 100
tspan = (0.0f0, 3.f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= ((u.^3)'true_A)'
end
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
scatter(t, ode_data[1,:], label="Observation: species 1", grid = "off")
scatter!(t, ode_data[2,:], label="Observation: species 2", xlab = "time", ylab="Species")
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
species = ["X","Y"]
test = [1,2]
plot(ode_data[test[1],:], ode_data[test[2],:], color = "green", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "Path in state space", color = "green")
plot!(esti[test[1],:], esti[test[2],:], color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
scatter!(esti[test[1],:], esti[test[2],:], label = "Path in state space", color = "red")
scatter(t, ode_data[1,:], label = "Observation: Species 1", grid = "off",legend =:topleft)
scatter!(t, ode_data[2,:], label = "Observation: Species 2")
scatter!(t, esti[1,:], label = "Estimation: Species 1")
scatter!(t, esti[2,:], label = "Estimation: Species 2")
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 800
verify = 50# for <verify>th epoch the L2 is calculated
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.0001)
sa = saver(n_epochs)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
# Callback function to observe two stage training.
cb1 = function ()
    println(sa.count_epochs)
    sa.count_epochs = sa.count_epochs +  1
    if mod(sa.count_epochs-1, verify)==0
        update_saver(sa, Tracker.data(two_stage_loss_fct()),Tracker.data(L2_loss_fct()),Dates.Time(Dates.now()))
        # println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
        pred = n_ode(u0)
        a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
        scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
        plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
        scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")
        display(a)
        savefig(string("paper/simple/model1_default_run5/", sa.count_epochs,"te_fit_in_statespace.pdf"))
        @save string("paper/simple/model1_default_run5/", sa.count_epochs,"te_dudt.bson") dudt
    else
        update_saver(sa, Tracker.data(two_stage_loss_fct()),0,Dates.Time(Dates.now()))
        # println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
    end

end
@time Flux.train!(two_stage_loss_fct, ps, data1, opt1, cb = cb1)
pred = n_ode(u0)
a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")
display(a)
savefig(string("paper/simple/model1_default_run5/", sa.count_epochs,"te_fit_in_statespace.pdf"))
@save string("paper/simple/model1_default_run5/", sa.count_epochs,"te_dudt.bson") dudt


# scatter(t, ode_data[1,:], label = "data", grid = "off")
# scatter!(t, ode_data[2,:], label = "data")
# plot!(t, Flux.data(pred[1,:]), label = "prediction")
# plot!(t, Flux.data(pred[2,:]), label = "prediction")
# header = string("col losses: ", sa.times[end] - sa.times[1])
# plot(range(1,stop=length(sa.l2s)),sa.l2s,label = "l2s", grid = "off")
# plot!(range(1,stop=length(sa.losses)),sa.losses,width  =2, label = header)
