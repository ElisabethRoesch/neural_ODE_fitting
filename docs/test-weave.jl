using Weave,Plots, DSP
#HTML
weave(joinpath(dirname(pathof(Weave)), "/Users/eroesch/Documents/phd/brave_new_world/Lisis_loss_function_comparison/docs", "diffeqtest.jmd"),
  out_path=:pwd,
  doctype = "md2html")
#pdf
weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
  out_path=:pwd,
  doctype = "md2pdf")
  #Markdown
weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
      doctype="pandoc",
      out_path=:pwd)
