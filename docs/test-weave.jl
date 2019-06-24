using Weave
#HTML
weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
  out_path=:pwd,
  doctype = "md2html")
#pdf
weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
  out_path=:pwd,
  doctype = "md2pdf")
  #Markdown
weave(joinpath(dirname(pathof(Weave)), "../examples", "FIR_design.jmd"),
      doctype="pandoc"
      out_path=:pwd)
