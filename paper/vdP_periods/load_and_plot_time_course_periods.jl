using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
# green
pred_col_c = "#82B366"
#grey
obs_c = "#696969"
# grey
ref_c = "#C698DB"
# red
stop_c = "#920005"

u0 = Float32[2.; 0.]
datasize = 200
tested_periods = 5 # this is not exact.

function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end

final_epoch = 3501
lengths = [5.0, 5.5, 6.0, 6.5 ,7.0, 7.5,8.0, 8.5]
foldernames = ["col_periods_short_50", "col_periods_short_55",
                "col_periods_short_60", "col_periods_short_65",
                "col_no_noise", "col_periods_short_75",
                "col_periods_short_80", "col_periods_short_85"]

ref_starts = [150,160,175,190,200,212,230,245]
for i in 1:length(foldernames)
    foldername = foldernames[i]
    key_t = string(final_epoch)
    start_test = 0.
    stop_test = 35.
    tspan_test = (start_test, stop_test)
    t_test = range(tspan_test[1], tspan_test[2], length = datasize*tested_periods)
    prob_ref = ODEProblem(trueODEfunc, u0, tspan_test)
    ode_data_ref = Array(solve(prob_ref, Tsit5(), saveat = t_test))

    start_train = 0.
    stop_train = lengths[i]
    tspan_train = (start_train, stop_train)
    t_train = range(tspan_train[1], tspan_train[2], length = datasize)
    prob = ODEProblem(trueODEfunc, u0, tspan_train)
    ode_data = Array(solve(prob, Tsit5(), saveat = t_train))

    @load string("/Users/eroesch/github/neural_ODE_fitting/paper/vdP/", foldername, "/", key_t, "te_dudt.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan_test, Tsit5(), saveat=t_test, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    lime = maximum(ode_data)
    a=  scatter(t_train, ode_data[1,:], label = "", color = obs_c,
            markerstrokewidth=0.001, grid = "off", framestyle = :box,
            ylim = (-3.5,3.5), size = (500,100), markersize = 2.35,
            yticks = ([-lime,0,lime],["","",""]),
            xticks = ([0,stop_train, stop_test],["","",""]))
    scatter!(t_train, ode_data[2,:], label="", color = obs_c, markerstrokewidth=0.001, markersize = 2.35)
    plot!(t_test,Flux.data(pred[1,:]), linewidth = 2, label = "", color = pred_col_c)
    plot!(t_test, Flux.data(pred[2,:]),linewidth = 2, label = "", color = pred_col_c)
    plot!(t_test[ref_starts[i]:end], ode_data_ref[1,:][ref_starts[i]:end], linestyle = :dash, label = "", linewidth = 2, color = obs_c)
    plot!(t_test[ref_starts[i]:end], ode_data_ref[2,:][ref_starts[i]:end], linestyle = :dash, label = "", linewidth = 2, color = obs_c)
    scatter!([t_train[1]], [ode_data[1,:][1]], label = "", markerstrokecolor = "white", markercolor = obs_c,  markersize = 2.35)
    scatter!([t_train[1]], [ode_data[2,:][1]], label = "", markerstrokecolor = "white", markercolor = obs_c,  markersize = 2.35)
    scatter!([stop_train], [ode_data[1,:][end]], label = "", markerstrokecolor = "white", markercolor = stop_c,  markersize = 2.35)
    scatter!([stop_train], [ode_data[2,:][end]], label = "", markerstrokecolor = "white", markercolor = stop_c,  markersize = 2.35)

    display(a)
    #savefig(string("paper/vdP/", foldername, "/plots/time_course_", key_t, "te_fit_selected_testing.pdf"))
    savefig(string("paper/vdP/periods/", foldername,"ts.pdf"))
end
