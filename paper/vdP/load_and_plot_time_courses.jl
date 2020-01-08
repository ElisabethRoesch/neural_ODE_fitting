using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
pred_col_c = "#82B366"
pred_l2_c = "#C698DB"
obs_c = "#696969"
u0 = Float32[2.; 0.]
datasize = 200
tspan = (0.0f0, 7.f0)
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
    @load string("paper/vdP/", foldername, "/", key_t, "te_dudt.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan, Tsit5(), saveat=t, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    a= scatter(t, ode_data[1,:], label="", color =obs_c, grid = "off",framestyle = :box)
    scatter!(t, ode_data[2,:], label="", color =obs_c)
    plot!(t, ode_data[1,:], label="", color =obs_c)
    plot!(t, ode_data[2,:], label="", color =obs_c)
    scatter!(t,Flux.data(pred[test[1],:]), label="", color =col, grid = "off",framestyle = :box)
    scatter!(t, Flux.data(pred[test[2],:]), label="", color =col)
    plot!(t,Flux.data(pred[test[1],:]), label="", color =col)
    plot!(t, Flux.data(pred[test[2],:]), label="", color =col)
    display(a)
    savefig(string("paper/vdP/", foldername, "/plots/time_course_", key_t, "te_fit_selected.pdf"))
end
