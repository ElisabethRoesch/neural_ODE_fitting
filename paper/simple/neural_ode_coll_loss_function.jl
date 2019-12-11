
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


scatter!(esti[test[1],:], esti[test[2],:], label = "Path in state space", color = "red")
scatter(t, ode_data[1,:], label = "Observation: Species 1", grid = "off",legend =:topleft)
scatter!(t, ode_data[2,:], label = "Observation: Species 2")
scatter!(t, esti[1,:], label = "Estimation: Species 1")
scatter!(t, esti[2,:], label = "Estimation: Species 2")
n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
n_epochs = 800
verify = 50# for <verify>th epoch the L2 is calculated
data1 = Iterators.repeated((), n_epochs)
opt1 = Descent(0.001)
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
        #savefig(string("paper/simple/col/", sa.count_epochs,"te_fit_in_statespace.pdf"))
        #@save string("paper/simple/col/", sa.count_epochs,"te_dudt.bson") dudt
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
#savefig(string("paper/simple/col/", sa.count_epochs,"te_fit_in_statespace.pdf"))
#@save string("paper/simple/col/", sa.count_epochs,"te_dudt.bson") dudt


est_c = "#D6B656"
pred_col_c = "#82B366"
pred_l2_c = "#9673A6"
obs_c = "#6C8EBF"


plot(ode_data[test[1],:], ode_data[test[2],:],
    label = "", ylim = (-1.7,1.7), xlim = (-1.7,1.7) ,xticks= [-1.0,1], yticks= [-1,1], size=(500,500), margin=5Plots.mm,
    xlab = species[test[1]], ylab = species[test[2]], grid = "off",framestyle = :box,
    color = obs_c)
scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "", color = obs_c)
savefig("paper/simple/Obs.pdf")



plot(ode_data[test[1],:], ode_data[test[2],:],
    label = "", ylim = (-1.7,1.7), xlim = (-1.7,1.7) ,xticks= [-1.0,1], yticks= [-1,1], size=(500,500), margin=5Plots.mm,
    xlab = species[test[1]], ylab = species[test[2]], grid = "off", framestyle = :box,
    color = obs_c)
scatter!(ode_data[test[1],:], ode_data[test[2],:], label = "", color = obs_c)
plot!(esti[test[1],:], esti[test[2],:], label = "", color = est_c)
scatter!(esti[test[1],:], esti[test[2],:], label = "", color = est_c)
plot!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), color = pred_col_c, label = "")
scatter!(Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "", color = pred_col_c)
savefig("paper/simple/Obs_Esti_Pred.pdf")



using JLD
#JLD.save("paper/simple/col/losses.jld", "col_losses", sa.losses)
#JLD.save("paper/simple/col/times.jld", "col_times", sa.times)

# scatter(t, ode_data[1,:], label = "data", grid = "off")
# scatter!(t, ode_data[2,:], label = "data")
# plot!(t, Flux.data(pred[1,:]), label = "prediction")
# plot!(t, Flux.data(pred[2,:]), label = "prediction")
# header = string("col losses: ", sa.times[end] - sa.times[1])
# plot(range(1,stop=length(sa.l2s)),sa.l2s,label = "l2s", grid = "off")

labels = [ "Loss 1", "Loss 2", "Delta"]
labels = [ "", "", ""]

#Plots.scalefontsizes(0.8)
selection = range(1,step = 10, stop =800)
pl_1_x=range(1,stop=length(sa_l2.losses))[selection]
pl_1_y=log.(sa_l2.losses)[selection]
pl_2_x=range(1,stop=length(sa.losses))[selection]
pl_2_y= log.(sa.losses)[selection]
plot(pl_1_x,pl_1_y, color = pred_l2_c, margin=5Plots.mm, width =2, label =labels[1],  grid = "off")
scatter!([pl_1_x[1],pl_1_x[end]],[pl_1_y[1],pl_1_y[end]], color = pred_l2_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")
plot!(pl_2_x, pl_2_y, color = pred_col_c, width=2, label = labels[2], xlab = "Training epoch", ylab= "Log(Loss)", grid = "off")
scatter!([pl_2_x[1],pl_2_x[end]],[pl_2_y[1],pl_2_y[end]], color = pred_col_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")
plot!(range(1,step = 50,stop=800), linestyle = :dash, log.(sa.l2s[range(1,step = 50,stop=800)]),color = pred_col_c, width = 2, label = labels[3], grid = "off")
vline!([51,251,401,551,751], linewidth = 2,color = "brown", label = "")
savefig("paper/simple/selection/loss_noe_legend.pdf") #plotting every 10th for visu


t1 = sa_l2.times-sa_l2.times[1]
t2 = sa.times- sa.times[1]
t2_end = t2[end]
t3 =  t2[range(1,step = 50,stop=800)]
s_t1 = t1[selection]
s_t2 = t2[selection]
a = []
for tt in s_t1
    w= round(Millisecond(tt), Second)
    push!(a,w)
end
b = []
for tt in s_t2
    w= round(Millisecond(tt), Second)
    push!(b,w)
end
c = []
for tt in t3
    w= round(Millisecond(tt), Second)
    push!(c,w)
end
cc=string(b[end])[1:3]
end_t= parse(Float64,cc)
end_t
plot(a, size =(200, 400), color = pred_l2_c, xlim = (-5,end_t+10), xticks=([0,end_t],["0",string(end_t)]), log.(sa_l2.losses)[selection], margin=2Plots.mm, width =2, label="",  grid="off")
plot!(b, color = pred_col_c, log.(sa.losses)[selection], width=2, label="", xlab="Time [Sec]", ylab="Log(Loss)", grid="off")
plot!(c, log.(sa.l2s[range(1,step = 50,stop=800)]), linestyle =:dash, color = pred_col_c, label="")
scatter!([b[1],b[end]],[log.(sa.losses)[1],log.(sa.losses)[end]], color = pred_col_c, label ="")
scatter!([a[1]],[log.(sa_l2.losses)[1]], color = pred_l2_c, label ="")
savefig("paper/simple/selection/timezoom.pdf")
