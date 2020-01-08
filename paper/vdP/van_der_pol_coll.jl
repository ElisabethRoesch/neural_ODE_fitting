using  Plots, Optim, Dates, DiffEqParamEstim, Flux, DiffEqFlux, OrdinaryDiffEq
using BSON: @save
u0 = Float32[2.; 0.]
datasize = 200
tspan = (0.0f0, 7.f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end
est_c = "#D6B656"
pred_col_c = "#82B366"
pred_l2_c = "#9673A6"
obs_c = "#6C8EBF"
col=pred_col_c
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
dudt = Chain(Dense(2,50,tanh),
        Dense(50,50,tanh),
        Dense(50,50,tanh),
        Dense(50,2))
# Parameters of the model which are to be learnt. They are: W1 (2x50), b1 (50), W2 (50x2), b2 (2)
ps = Flux.params(dudt)
# Getting loss function from two stage collocation function
function node_two_stage_function(model, x, tspan, saveat, ode_data,
            args...; kwargs...)
  dudt_(du,u,p,t) = du .= model(u)
  prob_fly = ODEProblem(dudt_,x,tspan)
  two_stage_method(prob_fly, saveat, ode_data)
end
loss_n_ode = node_two_stage_function(dudt, u0, tspan, t, ode_data, Tsit5(), reltol=1e-7, abstol=1e-9)
#  loss function
two_stage_loss_fct() = loss_n_ode.cost_function(ps)
# Defining anonymous function for the neural ODE with the model. in: u0, out: solution with current params.
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 3501
verify = 50 # for <verify>th epoch the L2 is calculated
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.0001)
sa = saver(n_epochs)
L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))
# Callback function to observe two stage training.
cb1 = function ()
    sa.count_epochs = sa.count_epochs +  1
    if mod(sa.count_epochs-1, verify)==0
        xx1 = Tracker.data(two_stage_loss_fct())
        xx2 = Tracker.data(L2_loss_fct())
        update_saver(sa, xx1, xx2, Dates.Time(Dates.now()))
        println("Epoch: ",sa.count_epochs,"\tCol loss: ", round(xx1,digits=3), "\tL2 control: ", round(xx2,digits=3))
        #pred = n_ode(u0)
        #a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
        #scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
        #plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
        #scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")
        #display(a)
        # as = range(-3, step = 0.2, stop = 3)
        # bs = range(-3, step = 0.2, stop = 3)
        # cords = Array{Tuple{Real,Float64},1}(undef,length(as)*length(bs))
        # m = 1
        # for a in as
        #     for b in bs
        #         cords[m] = (a,b)
        #         m = m+1
        #     end
        # end
        # grads = []
        # for i in cords
        #     cord = [i[1], i[2]]
        #     grad = Flux.data(dudt(cord))
        #     tuple = (grad[1], grad[2])
        #     push!(grads, tuple)
        # end
        #quiv_plt=quiver(cords, size = (500,500), quiver=grads, grid = :off,framestyle = :box)
        #plot!(ode_data[test[1],:], ode_data[test[2],:], ylim = (-3,3), xlim = (-3,3), linewidth =4, color = "red",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
        #display(quiv_plt)
        pred = n_ode(u0)
        a = plot(ode_data[1,:], ode_data[2,:],
            label = "", ylim = (-3,3), xlim = (-3,3) ,xticks= ([-1,1],["",""]), yticks=  ([-1,1],["",""]), size=(500,500), margin=5Plots.mm,
            xlab = "X",linewidth=3, ylab = "Y", grid = "off", framestyle = :box,
            color = obs_c,markerstrokecolor = obs_c)
        scatter!(markerstrokecolor = obs_c, ode_data[1,:], ode_data[2,:], label = "", color = obs_c)
        plot!(Flux.data(pred[1,:]),linewidth=3, Flux.data(pred[2,:]), color = col, label = "")
        scatter!(markerstrokecolor = col, Flux.data(pred[1,:]), Flux.data(pred[2,:]), label = "", color = col)
        display(a)
        @save string("paper/vdP/col/", sa.count_epochs,"te_dudt.bson") dudt
        #savefig(string("paper/vdP/", sa.count_epochs, "_statespace.pdf"))
    else
        update_saver(sa, Tracker.data(two_stage_loss_fct()),0,Dates.Time(Dates.now()))
        # println("\"",Tracker.data(two_stage_loss_fct()),"\" \"",Dates.Time(Dates.now()),"\";")
    end
end

test=[1,2]
species = ["X","Y"]
pred = n_ode(u0)
# a = plot(ode_data[test[1],:], ode_data[test[2],:], color = "green",xlab = species[test[1]], ylab = species[test[2]], label = "", legend=:bottomright, grid = "off")
# scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "A", color = "green")
# plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = "red", xlab = species[test[1]], ylab = species[test[2]], label = "", grid = "off")
# scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "B", color = "red")


# train n_ode with collocation method
@time Flux.train!(two_stage_loss_fct, ps, data1, opt1, cb = cb1)


scatter(t, ode_data[1,:], label="", color ="red", grid = "off",framestyle = :box)
scatter!(t, ode_data[2,:], label="", color ="blue")
plot!(t, ode_data[1,:], label="", color ="red")
plot!(t, ode_data[2,:], label="", color ="blue")
#
scatter!(t,Flux.data(pred[test[1],:]), label="", color ="green", grid = "off",framestyle = :box)
scatter!(t, Flux.data(pred[test[2],:]), label="", color ="brown")
plot!(t,Flux.data(pred[test[1],:]), label="", color ="green")
plot!(t, Flux.data(pred[test[2],:]), label="", color ="brown")

esti =loss_n_ode.estimated_solution


using JLD
JLD.save("paper/vdP/col/savelosses.jld", "losses", sa.losses)
JLD.save("paper/vdP/col/savetimes.jld", "times", sa.times)
JLD.save("paper/vdP/col/savel2s.jld", "times", sa.l2s)

plot(ode_data[test[1],:], ode_data[test[2],:],
    label = "",
    xlab = species[test[1]], ylab = species[test[2]], grid = "off", framestyle = :box,
    color = obs_c)
scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "", color = obs_c)
plot!(esti[test[1],:], esti[test[2],:], label = "", color = est_c)
scatter!(esti[test[1],:], esti[test[2],:], label = "", color = est_c)



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
