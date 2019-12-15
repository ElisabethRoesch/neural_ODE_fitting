using DiffEqParamEstim: decide_kernel
kernel_function = decide_kernel(:Epanechnikov)
kernel_function = decide_kernel(:Uniform)
kernel_function = decide_kernel(:Triangular)
