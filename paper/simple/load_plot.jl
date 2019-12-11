using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
pred_col_c = "#82B366"
pred_l2_c = "#9673A6"
obs_c = "#6C8EBF"
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
key_list = [51,251,401,551,751]
col = pred_l2_c
col = pred_col_c

foldername = "L2_001"
foldername = "model1_default_run3"
#Plots.scalefontsizes(0.8)
for key_t in key_list
    @load string("paper/simple/",foldername,"/", key_t, "te_dudt.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    a = plot(ode_data[test[1],:], ode_data[test[2],:],
        label = "", ylim = (-1.7,1.7), xlim = (-1.7,1.7) ,xticks= [-1.0,1], yticks= [-1,1], size=(500,500), margin=5Plots.mm,
        xlab = species[test[1]],linewidth=6, ylab = species[test[2]], grid = "off", framestyle = :box,
        color = obs_c,markerstrokecolor = obs_c)
    scatter!(markerstrokecolor = obs_c, ode_data[test[1],:], ode_data[test[2],:], label = "", color = obs_c)
    plot!(Flux.data(pred[test[1],:]),linewidth=3, Flux.data(pred[test[2],:]), color = col, label = "")
    #scatter!(markerstrokecolor = col, Flux.data(pred[test[1],:]), Flux.data(pred[test[2],:]), label = "", color = col)
    display(a)
    #savefig(string("paper/simple/selection/", foldername, "_",key_t,"te_fit_selected.pdf"))
end
