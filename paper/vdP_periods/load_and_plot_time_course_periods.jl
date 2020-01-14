using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
# green
pred_col_c = "#82B366"
#grey
obs_c = "#696969"
# grey
ref_c = "#C698DB"
# red
red_c = "#920005"

u0 = Float32[2.; 0.]
datasize = 200
tested_periods = 5 # this is not exact.
start_test = 0.
stop_test = 35.

tspan_test = (start_test, stop_test)

t_test = range(tspan_test[1], tspan_test[2], length = datasize*tested_periods)

function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end

final_epoch = 3501

foldernames = ["col_periods_short_75","col_no_noise","col_periods_short_65", "col_periods_short_6", "col_periods_short_55", "col_periods_short_5"]

for i in 1:length(foldernames)
    foldername = foldernames[i]
    key_t = string(final_epoch)
    @load string("/Users/eroesch/github/neural_ODE_fitting/paper/vdP/", foldername, "/", key_t, "te_dudt.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan_test, Tsit5(), saveat=t_test, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    #prob = ODEProblem(trueODEfunc, u0, tspan_train)
    prob_ref = ODEProblem(trueODEfunc, u0, tspan_test)

    #ode_data = Array(solve(prob,Tsit5(),saveat=t_train))
    #lime = maximum(ode_data)
    ode_data_ref = Array(solve(prob_ref,Tsit5(),saveat=t_test))
    #a=  scatter(t_train, ode_data[1,:], label = "", color = obs_c,
            markerstrokewidth=0.001, grid = "off", framestyle = :box,
            ylim = (-3.5,3.5), size = (500,100), markersize = 2.35,
            yticks= ([-lime,0,lime],["","",""]),
            xticks= ([0,end_train, end_test],["","",""]))
    #scatter!(t_train, ode_data[2,:], label="", color = obs_c, markerstrokewidth=0.001, markersize = 2.35)
    plot(t_test,Flux.data(pred[1,:]), linewidth = 2, label = "", color = pred_col_c)
    plot!(t_test, Flux.data(pred[2,:]),linewidth = 2, label = "", color = pred_col_c)


    #plot!(ref_range, ode_data_ref[1,:][start_ref:end], linestyle = :dash, label = "", linewidth = 2, color = obs_c)
    #plot!(ref_range, ode_data_ref[2,:][start_ref:end], linestyle = :dash, label = "", linewidth = 2, color = obs_c)

    display(a)
    #savefig(string("paper/vdP/", foldername, "/plots/time_course_", key_t, "te_fit_selected_testing.pdf"))
    savefig(string("paper/vdP/periods/", foldername,".pdf"))
end



a = range(1, step=0.4, stop =4)
fieldnames(a)
a[3:end]-a[4:end]
a[end]
a.ref
a.len
a.offset
a.step
