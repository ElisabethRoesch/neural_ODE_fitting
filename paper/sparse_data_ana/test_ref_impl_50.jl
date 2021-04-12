using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0 = Float32[2.0; 0.0]
datasize = 50
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps)) .+ 0.1randn(2,datasize)

du,u = collocate_data(data,tsteps,EpanechnikovKernel())

scatter(tsteps,data')
plot!(tsteps,u',lw=5)
savefig("paper/sparse_data_ana/colloc_50.png")
plot(tsteps,du')
savefig("paper/sparse_data_ana/colloc_du_50.png")

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(2, 50, tanh),
                  FastDense(50, 2))

function loss(p)
    cost = zero(first(p))
    for i in 1:size(du,2)
      _du = dudt2(@view(u[:,i]),p)
      dui = @view du[:,i]
      cost += sum(abs2,dui .- _du)
    end
    sqrt(cost)
end

pinit = initial_params(dudt2)
callback = function (p, l)
  return false
end

result_neuralode = DiffEqFlux.sciml_train(loss, pinit,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 10000)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)
nn_sol = prob_neuralode(u0, result_neuralode.minimizer)
scatter(tsteps,data')
plot!(nn_sol)
# get final loss value
val = result_neuralode.minimum
savefig("paper/sparse_data_ana/colloc_trained_50_loss_$val.png")
@save "paper/sparse_data_ana/50_dudt_stage1.bson" dudt2

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, data .- pred)
    return loss
end

@time numerical_neuralode = DiffEqFlux.sciml_train(loss_neuralode, result_neuralode.minimizer,
                                                ADAM(0.05), cb = callback,
                                                maxiters = 300)

nn_sol = prob_neuralode(u0, numerical_neuralode.minimizer)
scatter(tsteps,data')
plot!(nn_sol,lw=5)
savefig("post_trained_50.png")
@save "paper/sparse_data_ana/50_dudt_stage2.bson" dudt2
