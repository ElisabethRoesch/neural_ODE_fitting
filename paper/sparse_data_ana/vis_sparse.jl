using Plots

select = [1,3,4,5,6]

data_sizes = [10, 25, 50, 100, 200, 300]
col_vals = [0.29, 0.51, 0.56, 0.83, 1.23, 1.40]
l2_vals = [0.003, 0.40, 1.49, 1.82, 4.40, 7.29]

rel_col_vals = col_vals./data_sizes
rel_l2_vals = l2_vals./data_sizes

p_abs = scatter(data_sizes, col_vals, color = "red", label = "Loss 1", title = "absolute values")
plot!(data_sizes, col_vals, color = "red", label = "")
scatter!(data_sizes, l2_vals, color = "blue", label = "Loss 2")
plot!(data_sizes, l2_vals, color = "blue", label = "")


p_rel = scatter(data_sizes, rel_col_vals, color = "black", grid = "off", xlab = "Data size", ylab = "Loss", label = "")
plot!(data_sizes, rel_col_vals, color = "black", label = "")


scatter!(data_sizes, rel_l2_vals, color = "blue", label = "Loss 2")
plot!(data_sizes, rel_l2_vals, color = "blue", label = "")



plot(p_abs, p_rel)
savefig("paper/sparse_data_ana/App.pdf")
