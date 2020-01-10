using Flux, DiffEqFlux, OrdinaryDiffEq, DiffEqParamEstim, Plots, Dates
using BSON: @load
pred_col_c = "#82B366"
pred_l2_c = "#C698DB"
obs_c = "#696969"
ref_c = "#920005"
# nice red: "#920005"
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

prob = ODEProblem(trueODEfunc, u0, tspan_train)
prob_ref = ODEProblem(trueODEfunc, u0, tspan_test)

ode_data = Array(solve(prob,Tsit5(),saveat=t_train))
noise = rand(size(ode_data)[1],size(ode_data)[2]).*0.
ode_data = ode_data.+noise
ode_data_ref = Array(solve(prob_ref,Tsit5(),saveat=t_test))
key_list = Array(range(1,step=50,stop=n_epochs))

lime = maximum(ode_data)
col = pred_l2_c
col = pred_col_c
foldernames = [ "col", "col_low_noise", "col_medium_noise", "col_high_noise"]
l2_test_losses = []
#Plots.scalefontsizes(0.8)
for foldername in foldernames
    key_t = 3501
    @load string("paper/vdP/", foldername, "/", key_t, "te_dudt.bson") dudt
    n_ode = x->neural_ode(dudt, x, tspan_test, Tsit5(), saveat=t_test, reltol=1e-7, abstol=1e-9)
    pred = n_ode(u0)
    l2_test_loss  = log(Flux.data(sum(abs2, ode_data_ref .- n_ode(u0))))
    push!(l2_test_losses, l2_test_loss)
    println(l2_test_loss)
end
#l2_test_losses = [0.85900104, 6.9583673, 8.634295, 7.919167]
a = scatter([1], [l2_test_losses[1]], xlim = (0,2), size = (100,400),
       xticks = ([0],[""]), yticks = ([0],[""]), grid = "off", color = ref_c, width  = 2, label = "")
scatter!([1], [l2_test_losses[2]], color = ref_c, markershape = :utriangle, width = 2, label = "")
scatter!([1], [l2_test_losses[3]], color = ref_c, width = 2, markershape = :dtriangle, label = "")
scatter!([1], [l2_test_losses[4]], color = ref_c, width = 2, markershape = :diamond , label = "")
#plot!(s
#savefig(string("paper/vdP/", foldername, "/plots/time_course_", key_t, "te_fit_selected_testing.pdf"))
savefig(string("paper/vdP/noise_figure_selected_sub_plots/", foldername,"tests.pdf"))
