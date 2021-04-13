using Plots

data_sizes = [10, 25, 50, 100, 200, 300]
col_vals = [0.29, 0.51, 0.45, 0.83, 1.23, 1.40]
l2_vals = [0.003, 0.40, 1.49, 1.82, 4.40, ]

rel_col_vals = col_vals./data_sizes
rel_l2_vals = l2_vals./data_sizes

p_abs = scatter(data_sizes, col_vals)
scatter!(data_sizes, l2_vals)

p_rel = scatter(data_sizes, rel_col_vals)
scatter!(data_sizes, rel_l2_vals)


plot(p_abs, p_rel)
savfig("paper/sparse_data_ana/loss_vals.png")
