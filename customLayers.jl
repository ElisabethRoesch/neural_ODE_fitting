
struct sinLayer{a,b,d,e,n,i}
  A::a
  k::b
  ϕ::d
  c::e
  norm::n
  in::i
end

function sinLayer(in::Integer,norm::Int64)
  return sinLayer(param(rand(MersenneTwister(1234),Float32,in)),param(rand(MersenneTwister(1234),Float32,in)),param(rand(MersenneTwister(1234),Float32,in)),param(rand(MersenneTwister(1234),Float32,in)),norm,in)
end


Flux.@treelike sinLayer

function (a::sinLayer)(x::AbstractArray)
  A, k, ϕ, c, norm, in = a.A, a.k, a.ϕ, a.c, a.norm, a.in
  space=collect(1:in)
  if(norm==1)
    ret = @. A*sin(k*x+ϕ)+c

  elseif(norm==2)
    ret = @. x+A*sin(k*x+ϕ)+c
  elseif(norm==3)
    ret = @. x*(A*sin(k*x+ϕ)+c)
  elseif(norm==4)
    ret = @. (2*A*sin(k*x+ϕ)+c)*0.1^x
  end
  ret
end




struct polyLayer{w,d}
  W::w
  deg::d
end

function polyLayer(in::Integer, deg::Integer)
  return polyLayer(param(rand(Float32,in,deg+1)),deg)
end

#polyLayer(in) = polyLayer(in, 1)


Flux.@treelike polyLayer

function (a::polyLayer)(x::AbstractArray)
  W, deg = a.W, a.deg
  #display(x)
  ret=W[:,1]
  if(deg>=1)
    ret = @.ret + W[:,2]*x
    if(deg>=2)
      ret = @.ret + W[:,3]*x^2
      if(deg>=3)
        ret = @.ret + W[:,4]*x^3
        if(deg>=4)
          ret = @.ret + W[:,5]*x^4
          if(deg>=5)
            ret = @.ret + W[:,6]*x^4
            if(deg>=6)
              ret = @.ret + W[:,7]*x^6
              if(deg>=7)
                ret = @.ret + W[:,8]*x^7
              end
            end
          end
        end
      end
    end
  end
  ret
end


struct crossMul{w,i}
  W::w
  in::i
end

function crossMul(in::Integer)
  return crossMul(param(rand(MersenneTwister(1234),Float32,in^2)),in)
end


Flux.@treelike crossMul

function (a::crossMul)(x::AbstractArray)
  W, in = a.W, a.in
  ret = reshape(x * transpose(x),in^2).*W
  ret
end

struct dupeL{i}
  in::i
end

function dupe(in::Integer)
  return dupeL(in)
end


Flux.@treelike dupeL

function (a::dupeL)(x::AbstractArray)
  in = a.in
  collect(Iterators.flatten(Iterators.repeated(x,in)))[1:in]

end







struct errAct{a,b}
  A::a
  B::b
end
pa=[]
function errAct(in::Integer)
  global pa = fill(0,in)
  return errAct(param(rand(Float32,in)),param(collect(range(1f0, length=in, stop=8f0))))
end


Flux.@treelike errAct

function (a::errAct)(x::AbstractArray)
  A, B = a.A, a.B
 # display(pa)
 # diplay(A)
  println(sum(pa.-A))
  if(abs(maximum(A))>30 || abs(maximum(B))>30 || abs(minimum(A)) < 0.000001 || abs(minimum(B)) < 0.000001)
    print(maximum(A))
    print(" ")
    print(minimum(A))
    print(" ")
    print(maximum(B))
    print(" ")
    print(minimum(B))
    println("")
  end
  @. 1/(MathConstants.e^(2*A*x-B)^2)
  global pa = A
end
