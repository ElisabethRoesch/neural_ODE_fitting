
using Flux, DiffEqFlux, DifferentialEquations,DiffEqParamEstim, Plots, Optim, Dates
using BSON: @save
print("done")
# Structure to observe training
mutable struct saver_l2
    losses::Array{Float64,1}
    times::Array{Dates.Time,1}
    count_epochs::Int128
end
function saver_l2(n_epochs)
    losses = zeros(n_epochs)
    times = fill(Dates.Time(Dates.now()),n_epochs)
    count_epochs = 0
    return saver_l2(losses,times,count_epochs)
end
function update_saver(saver_l2, loss_i, time_i)
    epoch_i = saver_l2.count_epochs
    saver_l2.losses[epoch_i] = loss_i
    saver_l2.times[epoch_i] = time_i
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
ode_data = Array(solve(prob,Tsit5(),saveat = t))
scatter(t, ode_data[1,:], label="Observation: species 1", grid = "off", xlab= "Time", ylab= "Abundance")
scatter!(t, ode_data[2,:], label="Observation: species 2")
dudt = Chain(x -> x.^3,
       Dense(2,50, tanh),
       Dense(50,2))
ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7, abstol = 1e-9)
n_epochs = 800
verify = 50
species = ["X","Y"]
test = [1,2]
sa_l2 = saver_l2(n_epochs)
L2_loss_fct() = sum(abs2, ode_data .- n_ode(u0))
cb = function ()
    sa_l2.count_epochs = sa_l2.count_epochs +  1
    xx=Tracker.data(L2_loss_fct())
    update_saver(sa_l2,xx , Dates.Time(Dates.now()))
    println("\"", xx, "\" \"", Dates.Time(Dates.now()), "\";")
    if mod(sa_l2.count_epochs-1, verify)==0
        pred = n_ode(u0)
        a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
        scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
        plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
        scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")
        display(a)
        savefig(string("paper/simple/L2/", sa_l2.count_epochs,"te_fit_in_statespace.pdf"))
        @save string("paper/simple/L2/", sa_l2.count_epochs,"te_dudt.bson") dudt
    end
end

opt1 = Descent(0.001)
data = Iterators.repeated((), n_epochs)
@time Flux.train!(L2_loss_fct, ps, data, opt1, cb = cb)
pred = n_ode(u0)
a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")
display(a)
savefig(string("paper/simple/L2/", sa_l2.count_epochs,"te_fit_in_statespace.pdf"))
@save string("paper/simple/L2/", sa_l2.count_epochs,"te_dudt.bson") dudt

using JLD
JLD.save("paper/simple/L2/savelosses.jld", "sal2_losses", sa_l2.losses)
JLD.save("paper/simple/L2/savetimes.jld", "l2_times", sa_l2.times)

#save("data.jld", "data", r)
#load("data.jld")["data"]

# pred = n_ode(u0)
# scatter(t, ode_data[1,:], label = "Observation 1", color = "blue", grid = "off",xlab= "Time", ylab= "Abundance", legend=:top)
# scatter!(t, ode_data[2,:], label = "Observation 2", color = "orange")
# plot!(t, Flux.data(pred[1,:]), label = "Prediction 1", color = "blue")
# plot!(t, Flux.data(pred[2,:]), label = "Prediction 2", color = "orange")
# header = string("l2 losses:",sa_l2.times[end]-sa_l2.times[1])
# plot(range(1, stop = length(sa_l2.losses)), sa_l2.losses, width = 2, label = header, grid = "off")
