
using Flux, DiffEqFlux, DifferentialEquations,DiffEqParamEstim, Plots, Optim, Dates
using BSON: @save
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
u0 = Float32[2.; 0.]
datasize = 50
tspan = (0.0f0, 7.f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end
prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
species1 = "Van der Pol oscillator"
species2 = "Dummy data";

plot(ode_data[1,:], ode_data[2,:],
    label = "",
    xlab = "X", ylab = "Y", grid = "off",framestyle = :box,
    color = "brown")
scatter!(ode_data[1,:], ode_data[2,:],label = "", color = "brown")
#savefig("paper/vdP/Obs_statespace.pdf")
scatter(t, ode_data[1,:], label="", color ="red", grid = "off",framestyle = :box)
scatter!(t, ode_data[2,:], label="", color ="blue")
plot!(t, ode_data[1,:], label="", color ="red")
plot!(t, ode_data[2,:], label="", color ="blue")
#savefig("paper/vdP/Obs_time.pdf")
dudt = Chain(Dense(2,50,tanh),
        Dense(50,50,tanh),
        Dense(50,50,tanh),
        Dense(50,2))

ps = Flux.params(dudt)
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat = t, reltol = 1e-7, abstol = 1e-9)
n_epochs = 351
verify = 50
test = [1,2]
sa_l2 = saver_l2(n_epochs)
L2_loss_fct() = sum(abs2, ode_data .- n_ode(u0))
cb = function ()
    sa_l2.count_epochs = sa_l2.count_epochs +  1
    xx=Tracker.data(L2_loss_fct())
    update_saver(sa_l2,xx , Dates.Time(Dates.now()))
    println("Epoch: ",sa_l2.count_epochs,"\tL2 loss: ", round(xx,digits=3))
    if mod(sa_l2.count_epochs-1, verify)==0
        pred = n_ode(u0)
        a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
        scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
        plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
        scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")
        display(a)
        #savefig(string("paper/vdP/L2/", sa_l2.count_epochs,"te_fit_in_statespace.pdf"))
        @save string("paper/vdP/L2/", sa_l2.count_epochs,"te_dudt.bson") dudt
    end
end


test=[1,2]
species = ["X","Y"]
#pred = n_ode(u0)
# a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
# scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
# plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
# scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")

data = Iterators.repeated((), n_epochs)
opt1 = Descent(0.0001)
@time Flux.train!(L2_loss_fct, ps, data, opt1, cb = cb)
using JLD
JLD.save("paper/vdP/L2/savelosses.jld", "losses", sa_l2.losses)
JLD.save("paper/vdP/L2/savetimes.jld", "times", sa_l2.times)

pred = n_ode(u0)
scatter(t, ode_data[1,:], label="", color ="red", grid = "off",framestyle = :box)
scatter!(t, ode_data[2,:], label="", color ="blue")
plot!(t, ode_data[1,:], label="", color ="red")
plot!(t, ode_data[2,:], label="", color ="blue")
#
scatter!(t,Flux.data(pred[test[1],:]), label="", color ="green", grid = "off",framestyle = :box)
scatter!(t, Flux.data(pred[test[2],:]), label="", color ="brown")
plot!(t,Flux.data(pred[test[1],:]), label="", color ="green")
plot!(t, Flux.data(pred[test[2],:]), label="", color ="brown")

a=2
# plot(ode_data[test[1],:], ode_data[test[2],:],
#     label = "",
#     xlab = species[test[1]], ylab = species[test[2]], grid = "off", framestyle = :box,
#     color = obs_c)
# scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "", color = obs_c)
# plot!(esti[test[1],:], esti[test[2],:], label = "", color = est_c)
# scatter!(esti[test[1],:], esti[test[2],:], label = "", color = est_c)



# as = range(-3, step = 0.2, stop = 3)
# bs = range(-3, step = 0.2, stop = 3)
# cords = Array{Tuple{Real,Float64},1}(undef,length(as)*length(bs))
# m = 1
# for a in as
#     for b in bs
#         cords[m]=(a,b)
#         global m=m+1
#     end
# end
# grads = []
# for i in cords
#     cord = [i[1], i[2]]
#     #grad = Flux.data(dudt(cord))
#     grad = trueODEfunc([0.,0.], cord, 1, 1)
#     tuple = (grad[1], grad[2])
#     push!(grads, tuple)
# end



# quiv_plt=quiver(cords, quiver=grads,  size = (500,500), grid = :off,framestyle = :box)
# plot!(ode_data[test[1],:], ode_data[test[2],:], ylim = (-3,3), xlim = (-3,3), linewidth =4, color = "red",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off",framestyle = :box)
# display(quiv_plt)
# savefig("paper/vdP/observation_quiver.pdf")
