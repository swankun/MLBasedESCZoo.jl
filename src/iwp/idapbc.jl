export train!, policyfrom, loadidapbc
export NeuralIDAPBC, DefaultIDAPBC, TruthIDAPBC

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
const TruthIDAPBC = IDAPBCVariants{:Truth}()
const DefaultIDAPBC = NeuralIDAPBC

MLBasedESC.IDAPBCProblem(r::ReactionWheelPendulum) = MLBasedESC.IDAPBCProblem(r,DefaultIDAPBC)
function MLBasedESC.IDAPBCProblem(::ReactionWheelPendulum, 
    ::IDAPBCVariants{:Chain}
)
    Md⁻¹ = PSDMatrix(2, ()->[31.622776601683793,0,0,22.360679774997898])
    # Md⁻¹ = PSDMatrix(2, ()->[82.47221754422259 0.0; 81.64365569190376 0.9900262236663507])
    # Md⁻¹ = Md⁻¹_groundtruth
    Vd = FastChain(
        # inmap, FastDense(4, 10, elu; bias=false),
        FastDense(2, 10, elu; bias=false),
        FastDense(10, 20, elu; bias=false),
        FastDense(20, 4, elu; bias=false),
        FastDense(4, 1, square; bias=false),
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    ps = paramstack(P)
    return P, ps
end
function MLBasedESC.IDAPBCProblem(::ReactionWheelPendulum, 
    ::IDAPBCVariants{:Truth}
)
    Md⁻¹ = Md⁻¹_groundtruth
    Vd = Vd_groundtruth
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

function saveidapbc(θ, file)
    idapbc = (
        Mdinv = :( PSDMatrix(2, ()->[23.626696, 0.0, 0.0, 34.72419]) ),
        Vd = :(
            FastChain(
                FastDense(2, 10, elu; bias=false),
                FastDense(10, 20, elu; bias=false),
                FastDense(20, 4, elu; bias=false),
                FastDense(4, 1, square; bias=false),
            )
        ),
        θ = θ
    )
    BSON.@save file idapbc
end

function train!(P::IDAPBCProblem, ps; dq=0.1, kwargs...)
    L1 = PDELossPotential(P)
    # data = ([q1,q2] for q1 in -2pi:pi/20:2pi for q2 in -50pi:pi/10:50pi)
    data = ([q1,q2] for q1 in -2pi:pi/5:2pi for q2 in -50pi:pi/2:50pi)
    optimize!(L1,ps,collect(data);kwargs...)
end


function policyfrom(P::IDAPBCProblem; umax=Inf, lqrmax=Inf, kv=1)
    u_idapbc(x,p) = begin
        xbar = [rem2pi.(x[1:2], RoundNearest); x[3:end]]
        q1, q2, q1dot, q2dot = xbar
        effort = zero(q1)
        if (1-cos(q1) < 1-cosd(15)) && abs(q1dot) < 5
            effort = -dot(LQR, [sin(q1), sin(q2), q1dot, q2dot])
            return clamp(effort, -lqrmax, lqrmax)
        else
            effort = controller(P,xbar,p,kv=kv)
            return clamp(effort, -umax, umax)
        end
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

function evaluate(pbc::IDAPBCProblem, θ; kv=1, umax=Inf, lqrmax=umax, kwargs...)
    sys = ParametricControlSystem(ReactionWheelPendulum(), pbc; umax=umax, kv=kv)
    x = simulate(sys, θ; kwargs...)
    policy = policyfrom(pbc; umax=umax, lqrmax=lqrmax) 
    u = map(x->policy(x,θ), eachcol(x))
    return x, u
end

function plot(pbc::IDAPBCProblem, θ; out=true, kwargs...)
    evolution = evaluate(pbc, θ; kwargs...)
    fig = plot(evolution, out=false)
    N = 101
    X = range(-2pi, 2pi, length=N)
    Y = range(-2pi, 2pi, length=N)
    Z = zeros(N,N)
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            x = [X[ix],Y[iy]]
            Z[ix,iy] = pbc[:Vd](x,θ)[1]
        end
    end
    ax, ct = contour(fig[3,2][1,1], X,Y,Z, 
        colormap=:gnuplot, 
        levels=[0,1e-2,1,10,20,40,60,80]
    )
    ax.title = "Hd"
    Colorbar(fig[3,2][1,2], ct)
    # Axis(fig[4,1], title="Hd(t)")
    # Hd = map(x->pbc.Hd(wrap(x),θ)[1], eachcol(first(evolution)))
    # t = range(0, 1, length=size(evolution[1],2))
    # lines!(t, Hd)
    out && save("plots/out.png", fig)
    return fig
end

function plot_uru(pbc::IDAPBCProblem, θ; out=true)
    N = 51
    X = range(-pi, pi, length=N)
    Y = range(-pi, pi, length=N)
    Z = zeros(N,N)
    D = Vector{Tuple{Int,Int}}()
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            # x, u = evaluate(pbc, θ, kv=1.5e-2, x0=[X[ix],0,Y[iy],0], umax=3.0, tf=40.0)
            # x, u = evaluate(pbc, θ, kv=10, x0=[X[ix],0,Y[iy],0], tf=40.0)   # TruthIDAPBC
            x, u = evaluate(pbc, θ, umax=3.0, kv=5e-2, x0=[X[ix],0,Y[iy],0], tf=40.0)   # TruthIDAPBC
            Z[ix,iy] = sum(abs2,u)
            if abs(x[3,end]) > 1e-2
                push!(D, (ix,iy))
            end
        end
    end
    @show Zbar = maximum(Z)
    for d in D
        @show (X[first(d)], Y[last(d)])
        Z[d...] = Zbar
    end
    
    fig = Figure(resolution=(3.167,3).*300)
    majorfontsize = 30*2
    minorfontsize = 24*2
    ax_hm = Axis(
        fig[1,1][1,1], 
        aspect=AxisAspect(1),
        xticks=(range(-pi,pi,step=0.5pi), ["-π", "-π/2", "0", "π/2", "π"]),
        yticks=-3:1:3,
        title=L"u^\top R u", titlesize=majorfontsize, 
        xlabel="Pendulum angle (rad)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        ylabel="Pendulum velocity (rad/s)", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    hm = heatmap!(ax_hm, X,Y,Z, colormap=:RdPu_4, 
        # colorrange=(0.0,200),
        colorrange=(0.0,2000),
    )
    Colorbar(fig[1,1][1,2], hm, 
        # label="(× 1E+04)", labelsize=30, 
        ticklabelfont="CMU Serif", ticklabelsize=20*2,
        # tickformat=(xs)->["$(floor(Int,x/1e4))×10⁴" for x in xs]
    )
    out && save("plots/uRu.png", fig)
    return fig
end

function plot_Vd(pbc::IDAPBCProblem, θ)
    fig = Figure()
    N = 501
    X = range(-1.5pi, 1.5pi, length=N)
    # Y = range(-pi, pi, length=N)
    Y = range(-4pi, 4pi, length=N)
    # Y = range(-20, 20, length=N)
    ax, ct = contour(fig[1,1][1,1],
        X,Y,(x,y)->pbc[:Vd]([x;y],θ)[1],
        color=:black,
        # levels=15,
        levels=[0,0.4,10,20,40,60],
    )
    # Colorbar(fig[1,1][1,2], ct)
    save("plots/Vd.png", fig)
    fig
end
