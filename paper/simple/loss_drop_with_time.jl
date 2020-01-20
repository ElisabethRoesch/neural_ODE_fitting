using Plots, JLD, Dates

labels = [ "Col", "L2"]
labels = [ "", ""]
foldername = "col"
foldername1 = "L2"
pred_col_c = "#82B366"
pred_col_2 = "#800080"

l2s = JLD.load(string("paper/simple/",foldername,"/savel2s.jld"))["col_l2s"]
losses = JLD.load(string("paper/simple/",foldername,"/savelosses.jld"))["col_losses"]
times = JLD.load(string("paper/simple/",foldername,"/savetimes.jld"))["col_times"]

losses1 = JLD.load(string("paper/simple/",foldername1,"/savelosses.jld"))["sal2_losses"]
times1 = JLD.load(string("paper/simple/",foldername1,"/savetimes.jld"))["l2_times"]

n = length(l2s)

selection = Array(range(1,step = 1, stop =n))
#selection_snips = Array(range(1,step = step_size, stop =n))
#pl_1_x=range(1,stop=length(sa_l2.losses))[selection]
#pl_1_y=log.(sa_l2.losses)[selection]
pl_2_x = range(1,stop=length(losses))[selection]
pl_2_y = log.(losses)[selection]

pl_2_x1 = range(1,stop=length(losses1))[selection]
pl_2_y1 = log.(losses1)[selection]


# grey and red
cols = ["green","purple"]

#map(x -> 1/x, pl_2_y3)
#plot(pl_1_x,pl_1_y, color = pred_l2_c, margin=5Plots.mm, width =2, label =labels[1],  grid = "off")
#scatter!([pl_1_x[1],pl_1_x[end]],[pl_1_y[1],pl_1_y[end]], color = pred_l2_c, margin=5Plots.mm, width  =2, label ="",  grid = "off")
a = plot(pl_2_x, pl_2_y, width = 2.5, color = cols[1], label = labels[1])
plot!(pl_2_x1, pl_2_y1, width = 2.5, color = cols[2],label = labels[2],
    grid = "off", xlab = "Training epoch",
    ylab = "Log(Loss)", margin = 5Plots.mm, legend = :bottomleft)

#scatter!([ pl_2_x[end]], [pl_2_y[end]], color = pred_col_c, width  = 2, label = "")
#scatter!([pl_2_x1[end]], [pl_2_y1[end]], color = pred_col_2, markershape = :utriangle, width = 2, label = "")
#scatter!([pl_2_x2[end]], [pl_2_y2[end]], color = pred_col_2, width = 2, markershape = :dtriangle, label = "")
#scatter!([pl_2_x3[end]], [pl_2_y3[end]], color = pred_col_2, width = 2, markershape = :diamond , label = "")
#plot!(selection, linestyle = :dash, log.(l2s[selection]),color = pred_col_c, width = 2, label = labels[3], grid = "off")
#vline!(selection_snips, linewidth = 2,color = "brown", label = "")


savefig(string("paper/simple/selection/redo_lossdrop.pdf"))
