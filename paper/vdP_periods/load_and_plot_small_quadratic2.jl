using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
pred_col_c = "#82B366"
pred_l2_c = "#C698DB"
# grey
ref_c = "#696969"
# red
stop_c = "#920005"
u0 = Float32[2.; 0.]

function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end

col = pred_l2_c
col = pred_col_c
symbols = [:solid, :dash, :dashdot, :dashdotdot, :dot]
foldernames = ["col_periods_short_50", "col_periods_short_55",
                "col_periods_short_60", "col_periods_short_65",
                "col_no_noise", "col_periods_short_75",
                "col_periods_short_80", "col_periods_short_85"]
lengths = [5.0, 5.5, 6.0, 6.5 ,7.0, 7.5,8.0, 8.5]
alphas = [1.]
colors = ["#920005", "#920005", "#920005", "#920005", "#6C8EBF", "#6C8EBF", "#6C8EBF", "#6C8EBF"]

#scatter!([ode_data[1,:][1]], [ode_data[2,:][1]], color = ref_c, markerstrokecolor ="white", markercolor = "grey",  label = "")
for i in 1:length(foldernames)
  datasize = 200
  tspan = (0.0f0, lengths[i])
  t = range(tspan[1], tspan[2], length = datasize)
  prob = ODEProblem(trueODEfunc, u0, tspan)
  ode_data = Array(solve(prob,Tsit5(),saveat=t))
    a = plot(ode_data[1,:], ode_data[2,:],
         label = "", ylim = (-3,3), xlim = (-3,3) ,xticks= ([-1,1],["",""]), yticks=  ([-1,1],["",""]), size=(200,200), margin=5Plots.mm,
         xlab = "",linewidth=3, ylab = "", grid = "off", framestyle = :box,
         color =  ref_c)
    key_t = 3501
    foldername = foldernames[i]
    @load string("/Users/eroesch/github/neural_ODE_fitting/paper/vdP/", foldername, "/", key_t, "te_dudt.bson") dudt
    tspan = (0.0f0, lengths[i])
    t = range(tspan[1], tspan[2], length = datasize)
    prob = ODEProblem(trueODEfunc, u0, tspan)
    ode_data_t = Array(solve(prob,Tsit5(),saveat=t))
    #scatter!([ode_data_t[1,:][end]], [ode_data_t[2,:][end]], markerstrokecolor ="white", markercolor = colors[i], linewidth=3, color = obs_c, label = "")
    tspan_test = (0.0f0, 35.)
    t_test = range(tspan_test[1], tspan_test[2], length = datasize*5)
    n_ode = x->neural_ode(dudt, x, tspan_test, Tsit5(), saveat=t_test, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    a = scatter(ode_data[1,:], ode_data[2,:],
         label = "", ylim = (-3,3), xlim = (-3,3) ,xticks= ([-1,1],["",""]), yticks=  ([-1,1],["",""]), size=(200,200), margin=5Plots.mm,
         xlab = "",linewidth=3, ylab = "", grid = "off", framestyle = :box,
         color =  ref_c)
    plot!(Flux.data(pred[1,:]), Flux.data(pred[2,:]), linewidth = 3, color = pred_col_c, label = "")
    scatter!([ode_data_t[1,:][end]], [ode_data_t[2,:][end]], label = "", markerstrokecolor = "white",linewidth = 3, color = stop_c)
    scatter!([ode_data_t[1,:][1]], [ode_data_t[2,:][1]], label = "",markerstrokecolor = "white", linewidth = 3, color = ref_c)

    #scatter!(markerstrokecolor = col, Flux.data(pred[1,:]), Flux.data(pred[2,:]), label = "", color = col)
    display(a)
    #savefig(string("paper/vdP/periods/", foldername, "_state_space.pdf"))
    savefig(string("paper/vdP/periods/figure/",foldername,key_t,".pdf"))
end
