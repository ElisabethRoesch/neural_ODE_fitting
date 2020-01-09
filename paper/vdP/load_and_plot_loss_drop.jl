using Plots, JLD

labels = [ "", "", ""]
foldername = "col_noise"
l2s = JLD.load(string("paper/vdP/",foldername,"/savel2s.jld"))["times"]
losses = JLD.load(string("paper/vdP/",foldername,"/savelosses.jld"))["losses"]
times = JLD.load(string("paper/vdP/",foldername,"/savetimes.jld"))["times"]
n = length(l2s)
step_size = n/5
selection = range(1,step = 50, stop =n)
selection_snips = Array(range(500,step = step_size, stop =n))
#pl_1_x=range(1,stop=length(sa_l2.losses))[selection]
#pl_1_y=log.(sa_l2.losses)[selection]
pl_2_x=range(1,stop=length(losses))[selection]
pl_2_y= log.(losses)[selection]
#plot(pl_1_x,pl_1_y, color = pred_l2_c, margin=5Plots.mm, width =2, label =labels[1],  grid = "off")
#scatter!([pl_1_x[1],pl_1_x[end]],[pl_1_y[1],pl_1_y[end]], color = pred_l2_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")
plot(pl_2_x, pl_2_y, color = pred_col_c, width=2, label = labels[2], xlab = "Training epoch", ylab= "Log(Loss)", grid = "off")
scatter!([pl_2_x[1],pl_2_x[end]],[pl_2_y[1],pl_2_y[end]], color = pred_col_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")
plot!(selection, linestyle = :dash, log.(l2s[selection]),color = pred_col_c, width = 2, label = labels[3], grid = "off")
vline!(selection_snips, linewidth = 2,color = "brown", label = "")
savefig(string("paper/vdP/", foldername, "/selection/loss_no_legend.png"))
