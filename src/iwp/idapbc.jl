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
    # Md⁻¹ = PSDMatrix(2, ()->[4.688072309384954 0.5; 0.5 15.339299776947408])
    # Md⁻¹ = PSDMatrix(2, ()->[31.622776601683793,0,0,22.360679774997898])
    # Md⁻¹ = PSDMatrix(2, ()->[82.47221754422259 0.0; 81.64365569190376 0.9900262236663507])
    initps = 1 ./ MLBasedESC.Flux.glorot_uniform(4)
    Md⁻¹ = PSDMatrix(2, ()-> initps)
    # Md⁻¹ = Md⁻¹_groundtruth
    # Md⁻¹ = FastChain(
    #     FastDense(2, 8, elu; bias=true),
    #     FastDense(8, 4; bias=true),
    #     MLBasedESC.posdef
    # )
    Vd = FastChain(
        # inmap, FastDense(4, 10, tanh; bias=false),
        FastDense(2, 8, tanh; bias=false),
        FastDense(8, 16, tanh; bias=false),
        FastDense(16, 4, tanh; bias=false),
        FastDense(4, 1, square; bias=false),
    )
    # Vd = SOSPoly(2, 1:1)
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

function saveidapbc(P, θ, file)
    vds = *(
        "FastChain(",
        ("FastDense($(l.in),$(l.out),$(l.σ);bias=$(l.bias))," for l in P.Vd.layers)...,
        ")"
    )
    idapbc = (
        Mdinv = quote
            PSDMatrix(2, ()->$(θ[1:4]))
        end,
        # Mdinv = :( Md⁻¹_groundtruth ),
        Vd = Meta.parse(vds),
        θ = θ,
        Kv = 2e-3
    )
    BSON.@save file idapbc
end

function train!(P::IDAPBCProblem, ps; dq=0.1, kwargs...)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    # data = ([q1,q2] for q1 in -pi:pi/10:pi for q2 in -pi:pi/10:pi) |> collect
    # data = ([q1,q2] for q1 in -2pi:pi/5:2pi for q2 in -10pi:pi/2:10pi)
    data = ([q1,q2] for q1 in -2pi:pi/10:2pi for q2 in range(-50,50,length=101)) |> collect
    # data = ([q1,q2] for q1 in -pi:pi/10:pi for q2 in -pi:pi/10:pi)
    append!(data, [q1,q2] for q1 in -pi/4:pi/40:pi/4 for q2 in -pi/4:pi/40:pi/4)
    # optimize!(L2,ps,collect(data);kwargs...)
    optimize!(L1,ps,collect(data);kwargs...)
end


function policyfrom(P::IDAPBCProblem; umax=Inf, lqrmax=Inf, kv=1)
    u_idapbc(x,p) = begin
        xbar = [rem2pi.(x[1:2], RoundNearest); x[3:end]]
        # xbar = [x[1]; x[2]; x[3:end]]
        q1, q2, q1dot, q2dot = xbar
        effort = zero(q1)
        # if (1-cos(q1) < 1-cosd(10)) && abs(q1dot) < pi/4
        #     # xbar[1] = sin(q1)
        #     # xbar[2] = sin(q2)
        #     effort = -dot(LQR, xbar)
        #     return clamp(effort, -lqrmax, lqrmax)
        # else
            effort = controller(P,xbar,p,kv=kv)
            return clamp(effort, -umax, umax)
        # end
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

function publication_plot(pbc::IDAPBCProblem, θ; kv=1.2e-3, tf=6.0)
    kwargs = (
        umax = Inf,
        tf = tf,
        n = 1001,
        x0 = [3.,0,0,0],
        kv = kv,
        # kv = 1.2e-3,
        # kv = 5.0,
    )
    evolution = evaluate(pbc, θ; kwargs...)
    traj, _ = evolution
    Hd = map(eachcol(traj)) do x
        q = x[1:2]
        qdot = x[3:4]
        momentum = MLBasedESC._M⁻¹(pbc,q) \ qdot
        return hamiltoniand(pbc, [q;momentum], θ)
    end
    Hd[1:12] .= Hd[12]
    t = range(0, kwargs.tf, length=size(traj,2))
    majorfontsize = 36*1.5
    minorfontsize = 24*1.5
    fig = Figure(resolution=(1800,500))
    ax1 = Axis(fig[1,1],
        xlabel="Time (s)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        ylabel=L"Pendulum angle (radians)$\:$", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    lines!(ax1, t, traj[1,:], color=:black, linewidth=2)
    ax1.yticks = -3:3
    ylims!(-1.5, 3.5)

    ax2 = Axis(fig[1,2],
        xlabel="Time (s)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        ylabel=L"Rotor angle (radians) $\:$", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    lines!(ax2, t, traj[2,:], color=:black, linewidth=2)
    # ax2.yticks = -60:20:60
    # ylims!(-65, 65)

    ax3 = Axis(fig[1,3],
        xlabel="Time (s)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        ylabel=L"Learned Hamiltonian $H_d^{\;\theta}$", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    lines!(ax3, t, Hd, color=:black, linewidth=2)
    # ax2.yticks = -60:20:60
    # ylims!(-65, 65)
    save("plots/out.png", fig)
    save("plots/idapbc_iwp_evolution.eps", fig)
    return fig
end

plot_uru(::IDAPBCVariants{:Truth}) = plot_uru(IDAPBCProblem(ReactionWheelPendulum(), TruthIDAPBC)...)
plot_uru(::IDAPBCVariants{:Chain}) = plot_uru(loadidapbc("src/iwp/models/publications/idapbc_uru.bson")...)
function plot_uru(pbc::IDAPBCProblem, θ; out=true)
    N = 51
    X = range(-pi, pi, length=N)
    Y = range(-pi, pi, length=N)
    Z = zeros(N,N)
    D = Vector{Tuple{Int,Int}}()
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            # x, u = evaluate(pbc, θ, kv=1.5e-2, x0=[X[ix],0,Y[iy],0], umax=3.0, tf=40.0)
            # x, u = evaluate(pbc, θ, kv=2.5e-3, x0=[X[ix],0,Y[iy],0], umax=1.0, tf=40.0)
            # x, u = evaluate(pbc, θ, kv=10, x0=[X[ix],0,Y[iy],0], tf=40.0)   # TruthIDAPBC
            # x, u = evaluate(pbc, θ, umax=3.0, kv=5e-2, x0=[X[ix],0,Y[iy],0], tf=40.0)   # TruthIDAPBC
            # x, u = evaluate(pbc, θ, kv=1.5e-2, x0=[X[ix],0,Y[iy],0], umax=3.0, tf=40.0) # publications/idapbc_uru.bson
            x, u = evaluate(pbc, θ, kv=4e-3, x0=[X[ix],0,Y[iy],0], umax=1.0, tf=40.0) # publications/idapbc_tanh_Vd.bson
            Z[ix,iy] = sum(abs2,u)
            q1 = x[1,end]
            q1dot = x[3,end]
            if !(1-cos(q1) < 1-cosd(20)) && !(abs(q1dot) < 5)
                @show (q1, q1dot)
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
        xticks=piticks(0.5),
        yticks=-3:1:3,
        title=L"u^\top R u", 
        # titlesize=majorfontsize, 
        # xlabel="Pendulum angle (rad)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        # ylabel="Pendulum velocity (rad/s)", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        # xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        # yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    hm = heatmap!(ax_hm, X,Y,Z, colormap=:RdPu_4, 
        colorrange=(0.0,200),
        # colorrange=(0.0,2000), # TruthIDAPBC
    )
    Colorbar(fig[1,1][1,2], hm, 
        # label="(× 1E+04)", labelsize=30, 
        # ticklabelfont="CMU Serif", ticklabelsize=20*2,
        # tickformat=(xs)->["$(floor(Int,x/1e4))×10⁴" for x in xs]
    )
    out && save("plots/uRu.svg", fig)
    out && save("plots/uRu.png", fig)
    return fig
end

function plot_Vd(pbc::IDAPBCProblem, θ)
    fig = Figure()
    N = 101
    X = range(-pi, pi, step=pi/100)
    # Y = range(-4pi, 4pi, step=pi/100)
    # X = range(-2pi, 2pi, step=pi/50)
    Y = range(-50, 50, length=N)
    majorfontsize = 36*1.5
    minorfontsize = 24*1.5
    ax = Axis(fig[1,1][1,1],
        xticks=(range(-pi,pi,step=0.5pi), [L"$-\pi$ ", L"-\pi/2", L"0", L"\pi/2", L"$\pi$ "]),
        # title=L"u^\top R u", titlesize=majorfontsize, 
        xlabel=L"Pendulum angle $q_1$ (radians)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        ylabel=L"Rotor angle $q_2$ (radians)", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    ct = contour!(ax,
        X,Y,(x,y)->pbc[:Vd]([x;y],θ)[1],
        # color=:black,
        colormap=:rust,
        linewidth=1.5,
        enable_depth=false,
        # levels=5,
        levels=vcat(0,0.0001/2,0.01,0.1),
        # levels=vcat(0,1,3,20,100, 200, 600, 1000, 2000)
    )
    Colorbar(fig[1,1][1,2], ct,
        ticklabelfont="CMU Serif", ticklabelsize=30)
    save("plots/idapbc_Vd.png", fig)
    save("plots/idapbc_Vd.eps", fig)
    fig
end
