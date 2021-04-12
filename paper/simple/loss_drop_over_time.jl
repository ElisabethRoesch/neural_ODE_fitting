using Plots, JLD, Dates

#labels = [ "Col", "L2"]
labels = [ "", ""]
foldername_col = "col"
foldername_l2 = "L2"
pred_color_col = "#82B366"
pred_color_l2 = "#800080"

col_losses = JLD.load(string("paper/simple/",foldername_col,"/savelosses.jld"))["col_losses"]
col_times = JLD.load(string("paper/simple/",foldername_col,"/savetimes.jld"))["col_times"]

l2_losses = JLD.load(string("paper/simple/",foldername_l2,"/savelosses.jld"))["sal2_losses"]
l2_times = JLD.load(string("paper/simple/",foldername_l2,"/savetimes.jld"))["l2_times"]

col_losses[end]
l2_losses[end]

step_size = 50
n = length(l2_losses)
selection = Array(range(1,step = 10, stop = n))
t_col_selection = col_times[selection]
t_l2_selection = l2_times[selection]

t_plot_col = []
w = t_col_selection[1]-t_col_selection[1]
w_rounded = round(Millisecond(w), Second)
push!(t_plot_col, w_rounded)
for t in 2:length(t_col_selection)
    w = t_col_selection[t]-t_col_selection[t-1]
    w_rounded = round(Millisecond(w), Second)
    push!(t_plot_col, t_plot_col[end] +w_rounded)
end

t_plot_l2 = []
w = t_l2_selection[1]-t_l2_selection[1]
w_rounded = round(Millisecond(w), Second)
push!(t_plot_l2, w_rounded)
for t in 2:length(t_l2_selection)
    w = t_l2_selection[t]-t_l2_selection[t-1]
    w_rounded = round(Millisecond(w), Second)
    push!(t_plot_l2, t_plot_l2[end] + w_rounded)
end
cutoff = parse(Float32,string(t_plot_col[end])[1:3])
plot(t_plot_col, log.(col_losses)[selection],
    size = (200, 400),
    grid = "off", margin = 5Plots.mm, width = 2, tickfontcolor = "white",
    color = pred_color_col, label = "", xlim = [0., cutoff])
plot!(t_plot_l2, log.(l2_losses)[selection],
    width=2, label="",
    color = pred_color_l2)

scatter!([t_plot_col[1],t_plot_col[end]],[log.(col_losses)[1],log.(col_losses)[end]], color = pred_color_col, label ="")
scatter!([t_plot_l2[1]],[log.(l2_losses)[1]], color = pred_color_l2, label ="")
savefig("paper/simple/selection/loss_drop_over_time.pdf")
