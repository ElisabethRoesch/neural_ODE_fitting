
using Plots, JLD, Dates

#labels = [ "Col", "L2"]
labels = [ "", ""]
foldername_col = "col"
foldername_l2 = "L2"
pred_color_col = "#82B366"
pred_color_l2 = "#800080"

#col_l2_control = JLD.load(string("paper/simple/",foldername_col,"/savel2s.jld"))["col_l2s"]
col_losses = JLD.load(string("paper/simple/",foldername_col,"/savelosses.jld"))["col_losses"]
col_times = JLD.load(string("paper/simple/",foldername_col,"/savetimes.jld"))["col_times"]

l2_losses = JLD.load(string("paper/simple/",foldername_l2,"/savelosses.jld"))["sal2_losses"]
l2_times = JLD.load(string("paper/simple/",foldername_l2,"/savetimes.jld"))["l2_times"]

plot(col_times, log.(col_losses), size =(200, 400), color = pred_color_l2, margin=5Plots.mm, width =2, label="",  grid="off")
plot!(l2_times, log.(l2_losses), color = pred_color_col, width=2, label="", xlab="Time [Sec]", ylab="Log(Loss)", grid="off")



step_size = 50
n = length(l2_losses)
selection = Array(range(1,step = 10, stop = n))
t_col_selection = col_times[selection]
t_l2_selection = l2_times[selection]
a = []

w = t_l2_selection[1]-t_l2_selection[1]
w_rounded = round(Millisecond(w), Second)
push!(a, w_rounded)
for t in 2:length(t_l2_selection)
    w = t_l2_selection[t]-t_l2_selection[t-1]
    w_rounded = round(Millisecond(w), Second)
    push!(a,w_rounded)
end
b = []
push!(b,t_l2_selection[1]-t_l2_selection[1])
for tt in t_col_selection
    w= round(Millisecond(tt), Second)
    push!(b,w+b[end])
end

plot(a, log.(col_l2_control)[selection],
    size = (200, 400), grid = "off", margin = 5Plots.mm, width = 2,
    color = pred_color_l2, label = "")
plot!(b, color = pred_color_col, log.(sa.col_losses)[selection], width=2, label="", xlab="Time [Sec]", ylab="Log(Loss)", grid="off")
scatter!([b[1],b[end]],[log.(sa.col_losses)[1],log.(sa.col_losses)[end]], color = pred_color_col, label ="")
scatter!([a[1]],[log.(sa_l2.col_losses)[1]], color = pred_l2_c, label ="")
savefig("paper/simple/selection/loss_drop_over_time.png")
