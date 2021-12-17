export BayesianIDAPBC, mappolicy, margpolicy

const Vdmax = 24.94412861168683

const BayesianIDAPBC = IDAPBCVariants{:InformativeBayesian}()

_hardσ(x) = Vdmax*NNlib.hardσ(x-3)
function MLBasedESC.derivative(::typeof(_hardσ))
    return (x)->abs(x-3) <= 3 ? Vdmax/6 : zero(x)
end


function MLBasedESC.IDAPBCProblem(::ReactionWheelPendulum,
    ::IDAPBCVariants{:InformativeBayesian}
)
    Md⁻¹ = PSDMatrix(2, ()->[31.622776601683793,0,0,22.360679774997898])
    Vd = FastChain(
        FastDense(2, 10, elu; bias=false),
        FastDense(10, 10, elu; bias=false),
        FastDense(10, 1, _hardσ; bias=false),
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    ps = paramstack(P)
    μv = DiffEqFlux.initial_params(Vd)
    Σv = DiffEqFlux.initial_params(Vd)
    return P, (ps, μv, Σv)
end

function mappolicy(::IDAPBCVariants{:InformativeBayesian}, P::IDAPBCProblem, θ; 
    umax=Inf, kv=1
)
    ps, μv, Σv = θ
    Md = first(unstack(P, ps))
    newps = [Md; μv]
    function (x::AbstractVector)
        clamp(controller(P,x,newps,kv=kv), -umax, umax)
    end
end

function posterior(μ::AbstractVector, Σ::AbstractVector)
    distribution = map(μ, Σ) do _1,_2
        Distributions.Normal(_1, LogExpFunctions.softplus(_2))
    end
    return DistributionsAD.arraydist(distribution)
end

function margpolicy(::IDAPBCVariants{:InformativeBayesian}, P::IDAPBCProblem, θ, N; 
    umax=Inf, kv=1
)
    ps, μv, Σv = θ
    Md = first(unstack(P, ps))
    postdist = posterior(μv, Σv)
    function (x::AbstractVector)
        effort = mapreduce(+, eachcol(rand(postdist,N))) do sample
            newps = [Md; sample]
            controller(P,x,newps,kv=kv)
        end
        return clamp(effort/N, -umax, umax) 
    end
end
