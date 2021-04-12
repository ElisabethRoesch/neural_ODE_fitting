using Plots, JLD, Dates

#labels = [ "Col", "L2"]
labels = [ "", ""]
foldername_1 = "col_no_noise"
foldername_2 = "col_low_noise"
foldername_3 = "col_medium_noise"
foldername_4 = "col_high_noise"

folder_names = [foldername_1, foldername_2, foldername_3, foldername_4]

col_losses_all_folders = []
col_times_all_folders = []
for foldername_col in folder_names
    col_losses = JLD.load(string("paper/vdP/",foldername_col,"/savelosses.jld"))
    col_times = JLD.load(string("paper/vdP/",foldername_col,"/savetimes.jld"))
    push!(col_losses_all_folders, col_losses)
    push!(col_times_all_folders, col_times)
end

all_ends = []
for ends in 1:4
    push!(all_ends, col_losses_all_folders[ends]["losses"][end])
end

scatter(all_ends, grid = "off")
plot!(all_ends)

print(round.(all_ends,))
