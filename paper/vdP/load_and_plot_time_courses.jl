using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
# green
pred_col_c = "#82B366"
# lila blass
#pred_l2_c = "#C698DB"
#grey
obs_c = "#696969"
# grey
ref_c = "#C698DB"
# grey and red
cols = ["#696969","#920005","#920005","#920005"]
#markerlinewidh low or line off for obs schatter
col = pred_l2_c
col = pred_col_c
alphas = [1.,1.,1.,1.]
u0 = Float32[2.; 0.]
datasize = 200
end_train = 7.
tested_periods = 5
end_test = end_train*tested_periods

tspan_train = (0.0f0, end_train)
tspan_test = (0.0f0, end_test)

t_train = range(tspan_train[1], tspan_train[2], length = datasize)
t_test = range(tspan_test[1], tspan_test[2], length = datasize*tested_periods)

function trueODEfunc(du, u, p, t)
  du[1] = u[2]
  du[2] = (1-u[1]^2)*u[2]-u[1]
  return du
end


n_epochs=4401
key_list = Array(range(1,step=50,stop=n_epochs))
key_list = [4401]



foldernames = [ "col_no_noise", "col_low_noise", "col_medium_noise", "col_high_noise"]

noises = [0.0,0.1,0.2,0.5]

#Plots.scalefontsizes(0.8)
for i in 1:length(foldernames)
        foldername = foldernames[i]
        noise = noises[i]
    key_t = "4401"

    @load string("paper/vdP/", foldername, "/", key_t, "te_dudt_re.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan_test, Tsit5(), saveat=t_test, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    prob = ODEProblem(trueODEfunc, u0, tspan_train)
    prob_ref = ODEProblem(trueODEfunc, u0, tspan_test)

    ode_data = Array(solve(prob,Tsit5(),saveat=t_train))
    lime = maximum(ode_data)
    noise = rand(size(ode_data)[1],size(ode_data)[2]).*noise
    ode_data = ode_data.+noise
    ode_data_ref = Array(solve(prob_ref,Tsit5(),saveat=t_test))
    a=  scatter(t_train, ode_data[1,:], label = "", color = cols[i], alpha = alphas[i], markerstrokewidth=0.001, grid = "off",framestyle = :box,
            ylim = (-3.5,3.5), size = (500,100), markersize = 2.35,
            yticks= ([-lime,0,lime],["","",""]),
            xticks= ([0,end_train, end_test],["","",""]))
    scatter!(t_train, ode_data[2,:], label="", color = cols[i], alpha = alphas[i], markerstrokewidth=0.001, markersize = 2.35)
    #plot!(t_train, ode_data[1,:], label="", color =obs_c)
    #plot!(t_train, ode_data[2,:], label="", color =obs_c)
    #scatter!(t_test,Flux.data(pred[test[1],:]), label="", color =col, grid = "off",framestyle = :box)
    #scatter!(t_test, Flux.data(pred[test[2],:]), label="", color =col)
    plot!(t_test,Flux.data(pred[1,:]), linewidth = 2, label = "", color = pred_col_c)
    plot!(t_test, Flux.data(pred[2,:]),linewidth = 2, label = "", color = pred_col_c)
    plot!(t_test[200:end], ode_data_ref[1,:][200:end], linestyle = :dash, label = "", linewidth = 2, color = obs_c)
    plot!(t_test[200:end], ode_data_ref[2,:][200:end], linestyle = :dash, label = "", linewidth = 2, color = obs_c)

    display(a)
    #savefig(string("paper/vdP/", foldername, "/plots/time_course_", key_t, "te_fit_selected_testing.pdf"))
    savefig(string("paper/vdP/noise_figure_selected_sub_plots/", foldername,"reruns_new.pdf"))
end
