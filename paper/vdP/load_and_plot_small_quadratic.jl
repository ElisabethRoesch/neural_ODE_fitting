using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
pred_col_c = "#82B366"
pred_l2_c = "#C698DB"
obs_c = "#696969"
u0 = Float32[2.; 0.]
datasize = 200
tspan = (0.0f0, 14.f0)
t = range(tspan[1], tspan[2], length = datasize)
function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end

prob = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))
key_list = Array(range(1,step=50,stop=n_epochs))

col = pred_l2_c
col = pred_col_c

foldername = "col"
#Plots.scalefontsizes(0.8)
for key_t in key_list
    @load string("paper/vdP/",foldername,"/", key_t, "te_dudt.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    a = plot(ode_data[1,:], ode_data[2,:],
        label = "", ylim = (-3,3), xlim = (-3,3) ,xticks= ([-1,1],["",""]), yticks=  ([-1,1],["",""]), size=(500,500), margin=5Plots.mm,
        xlab = "X",linewidth=3, ylab = "Y", grid = "off", framestyle = :box,
        color = obs_c,markerstrokecolor = obs_c)
    scatter!(markerstrokecolor = obs_c, ode_data[1,:], ode_data[2,:], label = "", color = obs_c)
    plot!(Flux.data(pred[1,:]),linewidth=3, Flux.data(pred[2,:]), color = col, label = "")
    scatter!(markerstrokecolor = col, Flux.data(pred[1,:]), Flux.data(pred[2,:]), label = "", color = col)
    display(a)
    savefig(string("paper/vdP/",foldername,"/plots/",key_t,"te_fit_selected.pdf"))
end

scatter(markerstrokecolor = obs_c, ode_data[1,:], ode_data[2,:], label = "", color = obs_c)

ode_data[1,:]
