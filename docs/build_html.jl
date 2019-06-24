using Weave,Plots, DSP
#HTML
weave("/Users/eroesch/Documents/phd/brave_new_world/Lisis_loss_function_comparison/docs/n_ode.jmd",
  out_path="/Users/eroesch/Documents/phd/brave_new_world/Lisis_loss_function_comparison/docs",
  doctype = "md2html")
#pdf
#weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
#  out_path=:pwd,
#  doctype = "md2pdf")
  #Markdown
#weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
#      doctype="pandoc",
#      out_path=:pwd)
