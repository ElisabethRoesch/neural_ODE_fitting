using Plots, JLD

labels = [ "", "", ""]
foldername = "col"
foldername1 = "col_noise"
foldername2 = "col_low_noise"
foldername3 = "col_very_low_noise"


l2s = JLD.load(string("paper/vdP/",foldername,"/savel2s.jld"))["times"]
losses = JLD.load(string("paper/vdP/",foldername,"/savelosses.jld"))["losses"]
times = JLD.load(string("paper/vdP/",foldername,"/savetimes.jld"))["times"]

l2s1 = JLD.load(string("paper/vdP/",foldername1,"/savel2s.jld"))["times"]
losses1 = JLD.load(string("paper/vdP/",foldername1,"/savelosses.jld"))["losses"]
times1 = JLD.load(string("paper/vdP/",foldername1,"/savetimes.jld"))["times"]

l2s2 = JLD.load(string("paper/vdP/",foldername2,"/savel2s.jld"))["l2s"]
losses2 = JLD.load(string("paper/vdP/",foldername2,"/savelosses.jld"))["losses"]
times2 = JLD.load(string("paper/vdP/",foldername2,"/savetimes.jld"))["times"]

l2s3 = JLD.load(string("paper/vdP/", foldername3, "/savel2s.jld"))["l2s"]
losses3 = JLD.load(string("paper/vdP/",  foldername3, "/savelosses.jld"))["losses"]
times3 = JLD.load(string("paper/vdP/", foldername3, "/savetimes.jld"))["times"]

n = length(l2s)
step_size = n/5
selection = range(1,step = 50, stop =n)
selection_snips = Array(range(500,step = step_size, stop =n))
#pl_1_x=range(1,stop=length(sa_l2.losses))[selection]
#pl_1_y=log.(sa_l2.losses)[selection]
pl_2_x = range(1,stop=length(losses))[selection]
pl_2_y = log.(losses)[selection]

pl_2_x1 = range(1,stop=length(losses1))[selection]
pl_2_y1 = log.(losses1)[selection]

pl_2_x2 = range(1,stop=length(losses2))[selection]
pl_2_y2 = log.(losses2)[selection]

pl_2_x3 = range(1,stop=length(losses3))[selection]
pl_2_y3 = log.(losses3)[selection]

#plot(pl_1_x,pl_1_y, color = pred_l2_c, margin=5Plots.mm, width =2, label =labels[1],  grid = "off")
#scatter!([pl_1_x[1],pl_1_x[end]],[pl_1_y[1],pl_1_y[end]], color = pred_l2_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")
plot(pl_2_x, pl_2_y, color = pred_col_c, width=2, label = labels[2], xlab = "Training epoch", ylab= "Log(Loss)", grid = "off")
scatter!([pl_2_x[1],pl_2_x[end]],[pl_2_y[1],pl_2_y[end]], color = pred_col_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")

plot(pl_2_x, pl_2_y, color = pred_col_c, width=2, label = labels[2], xlab = "Training epoch", ylab= "Log(Loss)", grid = "off")
scatter!([pl_2_x[1],pl_2_x[end]],[pl_2_y[1],pl_2_y[end]], color = pred_col_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")


plot(pl_2_x, pl_2_y, color = pred_col_c, width=2, label = labels[2], xlab = "Training epoch", ylab= "Log(Loss)", grid = "off")
scatter!([pl_2_x[1],pl_2_x[end]],[pl_2_y[1],pl_2_y[end]], color = pred_col_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")



#plot!(selection, linestyle = :dash, log.(l2s[selection]),color = pred_col_c, width = 2, label = labels[3], grid = "off")
vline!(selection_snips, linewidth = 2,color = "brown", label = "")
savefig(string("paper/vdP/", foldername, "/selection/loss_no_legend.png"))
