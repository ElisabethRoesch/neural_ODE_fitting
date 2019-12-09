
using Flux, DiffEqFlux, DifferentialEquations,DiffEqParamEstim, Plots, Optim, Dates
using BSON: @save
print("done")
# Structure to observe training
mutable struct saver
    losses::Array{Float64,1}
    times::Array{Dates.Time,1}
    count_epochs::Int128
end
function saver(n_epochs)
    losses = zeros(n_epochs)
    times = fill(Dates.Time(Dates.now()),n_epochs)
    count_epochs = 0
    return saver(losses,times,count_epochs)
end
function update_saver(saver, loss_i, time_i)
    epoch_i = saver.count_epochs
    saver.losses[epoch_i] = loss_i
    saver.times[epoch_i] = time_i
end
u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0, 1.5f0)
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
n_epochs = 1000
sa_l2 = saver(n_epochs)
L2_loss_fct() = sum(abs2, ode_data .- n_ode(u0))
cb = function ()
    sa_l2.count_epochs = sa_l2.count_epochs +  1
    xx=Tracker.data(L2_loss_fct())
    update_saver(sa_l2,xx , Dates.Time(Dates.now()))
    println("\"", xx, "\" \"", Dates.Time(Dates.now()), "\";")
end

pred = n_ode(u0)
scatter(t, ode_data[1,:], label = "Observation 1", color = "blue", grid = "off",xlab= "Time", ylab= "Abundance", legend=:top)
scatter!(t, ode_data[2,:], label = "Observation 2", color = "orange")
plot!(t, Flux.data(pred[1,:]), label = "Prediction 1", color = "blue")
plot!(t, Flux.data(pred[2,:]), label = "Prediction 2", color = "orange")

opt = ADAM(0.1)
data = Iterators.repeated((), n_epochs)
@time Flux.train!(L2_loss_fct, ps, data, opt, cb = cb)

pred = n_ode(u0)
scatter(t, ode_data[1,:], label = "Observation 1", color = "blue", grid = "off",xlab= "Time", ylab= "Abundance", legend=:top)
scatter!(t, ode_data[2,:], label = "Observation 2", color = "orange")
plot!(t, Flux.data(pred[1,:]), label = "Prediction 1", color = "blue")
plot!(t, Flux.data(pred[2,:]), label = "Prediction 2", color = "orange")
#savefig("sogood.png")
#@save "model_l2_1000_epochs.bson" dudt
header = string("l2 losses:",sa_l2.times[end]-sa_l2.times[1])
plot(range(1, stop = length(sa_l2.losses)), sa_l2.losses, width = 2, label = header, grid = "off")
test_u0 = [0.1 , 5.5]
pred_t = n_ode(test_u0)
prob_t = ODEProblem(trueODEfunc, test_u0, tspan)
ode_data_t = Array(solve(prob_t, Tsit5(), saveat = t))
scatter(t, ode_data_t[1,:], label = "data")
scatter!(t, ode_data_t[2,:], label = "data")
plot!(t, Flux.data(pred_t[1,:]), label = "prediction")
plot!(t, Flux.data(pred_t[2,:]), label = "prediction")
