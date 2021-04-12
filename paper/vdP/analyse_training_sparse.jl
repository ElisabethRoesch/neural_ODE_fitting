using Plots, JLD, Dates

#labels = [ "Col", "L2"]
labels = [ "", ""]
foldername_50 = "col_periods_short_50"
foldername_60 = "col_periods_short_60"
foldername_75 = "col_periods_short_75"
foldername_80 = "col_periods_short_80"
foldername_85 = "col_periods_short_85"
6/7
folder_names = [foldername_50,foldername_60, foldername_75,foldername_80, foldername_85]

col_losses_all_folders = []
col_times_all_folders = []
for foldername_col in folder_names
    col_losses = JLD.load(string("paper/vdP/",foldername_col,"/savelosses.jld"))
    col_times = JLD.load(string("paper/vdP/",foldername_col,"/savetimes.jld"))
    push!(col_losses_all_folders, col_losses)
    push!(col_times_all_folders, col_times)
end

all_ends = []
for ends in 1:5
    push!(all_ends, col_losses_all_folders[ends]["losses"][end])
end

tines = [50,60,75,80,85]
scatter(tines,all_ends, grid = "off")
plot!(tines,all_ends)

print(all_ends)
