export train!, policyfrom, loadidapbc
export NeuralIDAPBC, DefaultIDAPBC

const Md_groundtruth = [1.0 -1.01; -1.01 1.02025]
const Md⁻¹_groundtruth = inv(Md_groundtruth)

function V(q)
    return m3*(cos(q[1]) - 1)
end
function MLBasedESC.jacobian(::typeof(V), q)
    [ -m3*sin(q[1]), zero(eltype(q)) ]
end

function Vd_groundtruth(q,ps)
    a1,a2,a3 = Md_groundtruth[[1,2,4]]
    k1 = 1/1000
    γ2 = -I1*(a2+a3)/(I2*(a1+a2))
    z = q[2] + γ2*q[1]
    return [I1*m3/(a1+a2)*cos(q[1]) + 1/2*k1*z^2]
end
function MLBasedESC.jacobian(::typeof(Vd_groundtruth), q, ps=nothing)
    ForwardDiff.jacobian(_1->Vd_groundtruth(_1,ps), q)
end

struct IDAPBCVariants{T} end
const NeuralIDAPBC = IDAPBCVariants{:Chain}()
const DefaultIDAPBC = NeuralIDAPBC

function MLBasedESC.IDAPBCProblem(::ReactionWheelPendulum, 
    ::IDAPBCVariants{:Chain}=DefaultIDAPBC
)
    # Md⁻¹ = Md⁻¹_groundtruth
    Md⁻¹ = PSDMatrix(2, ()->[31.622776601683793,0,0,22.360679774997898])

    # Vd = SOSPoly(2,1:2)
    # Vd = FastChain(
    #     inmap,
    #     SOSPoly(4,1:2)
    # )
    Vd = FastChain(
        # inmap, FastDense(4, 10, elu; bias=false),
        FastDense(2, 10, elu; bias=false),
        FastDense(10, 10, elu; bias=false),
        FastDense(10, 1, square; bias=false),
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    ps = paramstack(P)
    return P, ps
end

function loadidapbc(file)
    BSON.@load file idapbc
    Md⁻¹ = eval(idapbc.Mdinv)
    Vd = eval(idapbc.Vd)
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    return P, idapbc.θ
end

function train!(P::IDAPBCProblem, ps; dq=pi/10, kwargs...)
    L1 = PDELossPotential(P)
    data = ([q1,q2] for q1 in -pi:dq:pi for q2 in -pi:dq:pi)
    optimize!(L1,ps,collect(data);kwargs...)
end


function policyfrom(P::IDAPBCProblem; umax=Inf, kv=1)
    u_idapbc(x,p) = begin
        xbar = [rem2pi.(x[1:2], RoundNearest); x[3:end]]
        q1, q2, q1dot, q2dot = xbar
        effort = zero(q1)
        if (1-cos(q1) < 1-cosd(10)) && abs(q1dot) < 5
            effort = -dot(LQR, [sin(q1), sin(q2), q1dot, q2dot])
        else
            effort = controller(P,xbar,p,kv=kv)
        end
        return clamp(effort, -umax, umax)
    end
end

function MLBasedESC.ParametricControlSystem(::ReactionWheelPendulum, 
    prob::IDAPBCProblem; umax=Inf, kv=1
)
    return ParametricControlSystem(
        ReactionWheelPendulum(), 
        policyfrom(prob, umax=umax, kv=kv)
    )
end
