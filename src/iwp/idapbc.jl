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

function sumofsquares(x::AbstractVector, ::Any=nothing)
    sum(abs2, x)
end
function MLBasedESC.jacobian(::typeof(sumofsquares), x, ::Any=nothing)
    2x
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
const NeuralIDAPBCSoSNet = IDAPBCVariants{:SoSNet}()
const TruthIDAPBC = IDAPBCVariants{:Truth}()
const DefaultIDAPBC = NeuralIDAPBC

MLBasedESC.IDAPBCProblem(r::ReactionWheelPendulum) = MLBasedESC.IDAPBCProblem(r,DefaultIDAPBC)
function MLBasedESC.IDAPBCProblem(::ReactionWheelPendulum, 
    ::IDAPBCVariants{:Chain}
)
    # Md⁻¹ = PSDMatrix(2, ()->[4.688072309384954 0.5; 0.5 15.339299776947408])
    # Md⁻¹ = PSDMatrix(2, ()->[82.47221754422259 0.0; 81.64365569190376 0.9900262236663507])
    Md⁻¹ = PSDMatrix(2, ()->[31.622776601683793,0,0,22.360679774997898])
    Vd = FastChain(
        FastDense(2, 8, elu, bias=false),
        FastDense(8, 4, elu, bias=false),
        FastDense(4, 1, square, bias=false),
    )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    ps = paramstack(P)
    return P, ps
end
function MLBasedESC.IDAPBCProblem(::ReactionWheelPendulum, ::IDAPBCVariants{:SoSNet})
    p = BSON.load("src/iwp/models/neuralidapbc_sosnet.bson")[:paramvec]
    Vd = FastChain(
        FastDense(2, 12, elu; bias=false),
        FastDense(12, 8, elu; bias=false),
        FastDense(8, 4, elu; bias=false),
        sumofsquares
    )
    Md⁻¹ = PSDMatrix(2, () -> 1 ./ MLBasedESC.Flux.glorot_uniform(4) )
    P = IDAPBCProblem(2,M⁻¹,Md⁻¹,V,Vd,G,G⊥)
    return P, p
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
    vds = ["FastChain("]
    for l in P.Vd.layers
        if isa(l, FastDense)
            push!(vds,"FastDense($(l.in),$(l.out),$(l.σ);bias=$(l.bias)),")
        else
            fname_full = replace(string(typeof(l)), "typeof" => "")
            fname_scope = last(split(fname_full, "."))
            push!(vds, fname_scope)
        end
    end
    vds = *(vds...)
    idapbc = (
        Mdinv = quote
            PSDMatrix(2, ()->$(θ[1:4]))
        end,
        Vd = Meta.parse(vds),
        θ = θ,
        Kv = 1.0
    )
    BSON.@save file idapbc
end

function train!(P::IDAPBCProblem, ps; dq=0.1, kwargs...)
    L1 = PDELossPotential(P)
    L2 = PDELossKinetic(P)
    # data = ([q1,q2] for q1 in -2pi:pi/20:2pi for q2 in -2pi:pi/20:2pi) |> collect
    # data = ([q1,q2] for q1 in -2pi:pi/20:2pi for q2 in range(-50, 50, length=201)) |> collect
    data = ([q1,q2] for q1 in -4pi:pi/20:4pi for q2 in range(-50,50,length=201)) |> collect
    # data = ([q1,q2] for q1 in -pi:pi/20:pi for q2 in -pi:pi/20:pi) |> collect
    # data = ([q1,q2] for q1 in -2pi:pi/20:2pi for q2 in range(-50pi,50pi,length=201)) |> collect
    # data = ([q1,q2] for q1 in range(-100,100,length=201) for q2 in range(-100,100,length=201)) |> collect
    append!(data, [q1,q2] for q1 in -pi/3:pi/30:pi/3 for q2 in -pi/3:pi/30:pi/3)
    append!(data, [q1,0] for q1 in range(pi-pi/6, pi+pi/6, step=pi/30))
    append!(data, [q1,0] for q1 in range(-pi-pi/6, -pi+pi/6, step=pi/30))
    append!(data, [0,q2] for q2 in range(pi-pi/6, pi+pi/6, step=pi/30))
    append!(data, [0,q2] for q2 in range(-pi-pi/6, -pi+pi/6, step=pi/30))
    optimize!(L1,ps,data;kwargs...)
    # optimize!(L2,ps,collect(data);kwargs...)
end


function policyfrom(P::IDAPBCProblem; umax=Inf, lqrmax=Inf, kv=1)
    u_idapbc(x,p) = begin
        xbar = [rem2pi.(x[1:2], RoundNearest); x[3:end]]
        # xbar = [x[1]; x[2]; x[3:end]]
        # xbar = [
        #     rem(x[1], 2*2pi, RoundNearest)
        #     rem(x[2], 2*2pi, RoundNearest)
        #     x[3]
        #     x[4]
        # ]
        q1, q2, q1dot, q2dot = xbar
        effort = zero(q1)
        if false#(1-cos(q1) < 1-cosd(30)) && abs(q1dot) < 5
            # xbar[1] = sin(q1)
            # xbar[2] = sin(q2)
            effort = -dot(LQR, xbar)
            return clamp(effort, -lqrmax, lqrmax)
        else
            effort = controller(P,xbar,p,kv=kv)
            return clamp(effort, -umax, umax)
        end
        # q1, q2, q1dot, q2dot = x
        # dist = norm([1-cos(q1), 0.5*q1dot/10])
        # # dist = 1 - cq1
        # eta0 = exp(-(5*dist)^2)
        # eta1 = 1 - eta0
        # lqr = -dot(LQR, xbar)
        # swing = controller(P,xbar,p)
        # effort = eta0*lqr + eta1*swing
        # return clamp(effort, -umax, umax)
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
    sys = ParametricControlSystem(ReactionWheelPendulum(), pbc, umax=umax, kv=kv)
    x = simulate(sys, θ; kwargs...)
    policy = policyfrom(pbc; umax=umax, lqrmax=lqrmax, kv=kv) 
    u = map(x->policy(x,θ), eachcol(x))
    return x, u
end

function plot(pbc::IDAPBCProblem, θ; out=true, kwargs...)
    evolution = evaluate(pbc, θ; kwargs...)
    fig = plot(evolution, out=false)
    # N = 101
    # X = range(-2pi, 2pi, length=N)
    # Y = range(-2pi, 2pi, length=N)
    # Z = zeros(N,N)
    # Threads.@threads for ix in 1:length(X)
    #     for iy in 1:length(Y) 
    #         x = [X[ix],Y[iy]]
    #         Z[ix,iy] = pbc[:Vd](x,θ)[1]
    #     end
    # end
    # ax, ct = contour(fig[3,2][1,1], X,Y,Z, 
    #     colormap=:gnuplot, 
    #     levels=[0,1e-2,1,10,20,40,60,80]
    # )
    # ax.title = "Hd"
    # Colorbar(fig[3,2][1,2], ct)
    Axis(fig[3,2], title="Hd(t)")
    Hd = map(eachcol(first(evolution))) do x
        p = inv(pbc.M⁻¹)*x[3:4]
        qp = [x[1:2]; p]
        # qp = [x[1]; 0; p]
        hamiltoniand(pbc, qp, θ)
    end
    t = range(0, 1, length=size(evolution[1],2))
    lines!(t, Hd)
    out && save("plots/idapbcout.png", fig)
    return fig
end

function publication_plot(pbc::IDAPBCProblem, θ; kv=1.2e-3, tf=6.0)
    x0s = [
        # [3.,0,0,0],
        [1.,-55,0,0],
        [1.25,-66,0,0],
        [1.25,-66,0,0],
        [1.5,-80,0,0],
        [1.75,-95,0,0],
        [2.0,-110,0,0],
        [2.25,-123,0,0],
        [2.5,-140,0,0],
        [2.75,-150,0,0],
        [3.,-160,0,0],
    ] |> reverse
    linecolors = [:black; [Symbol("gray$(i)") for i=range(30, step=5, length=length(x0s)-1)]]
    fig = Figure(resolution=(1200,400).*1.1)
    ax1 = Axis(fig[1,1],
        xlabel=L"Pendulum angle $q_1$ (rad)",
        ylabel=L"$\dot{q}_1$ (rad/s)", 
        palette=(
            color=linecolors,
        )
    )
    ax2 = Axis(fig[1,2],
        xlabel="Time (s)", 
        ylabel=L"Hamiltonian $H_d^{\;\theta}$", 
        palette=(
            color=linecolors,
        )
    )
    for x0 in x0s
        kwargs = (
            umax = Inf,
            tf = tf,
            n = 1001,
            x0 = x0,
            kv = kv,
        )
        evolution = evaluate(pbc, θ; kwargs...)
        traj, _ = evolution
        Hd = map(eachcol(traj)) do x
            q = x[1:2]
            qdot = x[3:4]
            momentum = MLBasedESC._M⁻¹(pbc,q) \ qdot
            return hamiltoniand(pbc, [q;momentum], θ)
        end
        t = range(0, kwargs.tf, length=size(traj,2))
        lines!(ax1, traj[1,:], traj[3,:], linewidth=2)
        # lines!(ax1, t, traj[1,:], linewidth=2)
        lines!(ax2, t, Hd, linewidth=2)
    end
    ax1.yticks = -3:3
    ylims!(ax1, -3.25, 1.25)
    # ylims!(ax1, -0.5, 3.25)
    # ax2.yticks = 0:0.05:0.25
    # ylims!(ax2, -0.0125, 0.26)
    save("plots/out.png", fig)
    # save("plots/idapbc_iwp_evolution.eps", fig)
    return fig
end

plot_uru(::IDAPBCVariants{:Truth}) = plot_uru(IDAPBCProblem(ReactionWheelPendulum(), TruthIDAPBC)...)
plot_uru(::IDAPBCVariants{:Chain}) = plot_uru(loadidapbc("src/iwp/models/publications/idapbc_uru.bson")...)
function plot_uru(pbc::IDAPBCProblem, θ; out=true)
    N = 21
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
            x, u = evaluate(pbc, θ, kv=1.5e-2, x0=[X[ix],0,Y[iy],0], umax=3.0, tf=20.0) # publications/idapbc_uru.bson
            # x, u = evaluate(pbc, θ, kv=4e-3, x0=[X[ix],0,Y[iy],0], umax=1.0, tf=40.0) # publications/idapbc_tanh_Vd.bson (I2=0.5I2)
            Z[ix,iy] = sum(abs2,u)
            q1 = x[1,end]
            q1dot = x[3,end]
            if !(1-cos(q1) < 1-cosd(20)) && !(abs(q1dot) < 5)
                @warn "Failed to catch with q0 = $((q1, q1dot))."
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
    )
    hm = heatmap!(ax_hm, X,Y,Z, colormap=:RdPu_4, 
        colorrange=(0.0,200),
        # colorrange=(0.0,2000), # TruthIDAPBC
    )
    Colorbar(fig[1,1][1,2], hm)
    out && save("plots/uRu.svg", fig)
    out && save("plots/uRu.png", fig)
    return fig
end
function plot_Vd(pbc::IDAPBCProblem, θ)
    fig = Figure(resolution=1.25.*(800,400))
    plot_Vd!(fig, pbc::IDAPBCProblem, θ)
    save("plots/idapbc_Vd.png", fig)
    save("plots/idapbc_Vd.eps", fig)
end
function plot_Vd!(fig, pbc::IDAPBCProblem, θ)
    N = 101
    X = Y = range(-pi, pi, step=pi/101)
    # Y = range(-2pi, 2pi, step=pi/100)
    # Y = range(-20pi, 20pi, step=pi/2)
    Y = range(-50, 50, length=201)
    majorfontsize = 36*1.5
    minorfontsize = 24*1.5
    ax = Axis(fig[1,1][1,1],
        xticks=(range(-pi,pi,step=0.5pi), [L"$-\pi$ ", L"-\pi/2", L"0", L"\pi/2", L"$\pi$ "]),
        # yticks=(range(-2pi,2pi,step=pi), [L"$-2\pi$ ", L"-\pi", L"0", L"\pi", L"$2\pi$ "]),
        yticks=-50:25:50,
        xlabel=L"Pendulum angle $q_1$ (rad)", 
        ylabel=L"Rotor angle $q_2$ (rad)", 
    )
    ct = contour!(ax,
        X,Y,(x,y)->pbc[:Vd]([x;y],θ)[1],
        colormap=:grays,
        linewidth=2,
        # levels=20,
        levels=[0.001, 0.005, 0.01, 0.05, 0.1, collect(0.2:0.1:1)...],
        # levels=vcat(0,0.0001,0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3),
        # levels=vcat(0,1,3,20,100, 200, 600, 1000, 2000),
        colorrange=(0,0.85),
    )
    ylims!(-55,55)
    fig
end

function Hd_q1(pbc::IDAPBCProblem, ps)
    fig = Figure()
    ax1 = Axis(fig[1, 1], yticklabelcolor = :blue,
        topspinevisible = true, rightspinevisible = true,
        xticks=piticks(0.5))
    ax2 = Axis(fig[1, 1], yticklabelcolor = :red, yaxisposition = :right)
    hidespines!(ax2)
    hidexdecorations!(ax2)
    lines!(ax1, -pi:pi/50:pi, (x)->hamiltoniand(pbc, [x,0,0,0], ps)[1], color=:blue)
    lines!(ax2, -pi:pi/50:pi, (x)->Vd_groundtruth([x,0], ps)[1], color=:red)
    save("plots/out.png", fig)
end 

function contour_u(pbc::IDAPBCProblem, ps; kwargs...)
    fig = Figure(resolution=1.25.*(800,400))
    contour_u!(fig, pbc, ps; kwargs...)
    save("plots/idapbc_u.png", fig)
    save("plots/idapbc_u.eps", fig)
end
function contour_u!(fig, pbc::IDAPBCProblem, ps; kv=0.0005)
    N = 101
    X = Y = range(-pi, pi, step=pi/50)
    # Y = range(-2pi, 2pi, step=pi/100)
    # Y = range(-20pi, 20pi, step=pi/2)
    Y = range(-10, 10, length=101)
    majorfontsize = 36*1.5
    minorfontsize = 24*1.5
    ax = Axis(fig[1,1][1,1],
        xticks=(range(-pi,pi,step=0.5pi), [L"$-\pi$ ", L"-\pi/2", L"0", L"\pi/2", L"$\pi$ "]),
        # levels=10,
        xlabel=L"Pendulum angle $q_1$ (rad)", 
        ylabel=L"$\dot{q}_1$ (rad/s)", 
        colorrange=(-4,4)
    )
    ct = contourf!(ax,
        X,Y,(x,y)->controller(pbc,[x;0;y;0],ps,kv=kv),
        colormap=:grays,
    )
    Colorbar(fig[1,1][1,2], limits=(-4,4), colormap=:grays)
    # ylims!(-55,55)
    fig
end

function two_contours(pbc::IDAPBCProblem, ps)
    fig = Figure(resolution=1.1.*(1200,400))
    plot_Vd!(fig[1,1], pbc, ps)
    contour_u!(fig[1,2], pbc, ps)
    save("plots/idapbc_contours.png", fig)
    save("plots/idapbc_contours.eps", fig)
    fig
end

function animate(::ReactionWheelPendulum)
    # Simulation setup
    pbc, ps = IDAPBCProblem(ReactionWheelPendulum(), NeuralIDAPBCSoSNet)
    x0s = [
        [1.,-55,0,0],
        [1.25,-66,0,0],
        [1.75,-95,0,0],
        [2.25,-123,0,0],
        [2.75,-150,0,0],
        [3.,-160,0,0],
    ] 
    x0 = x0s[3]
    kwargs = (
        umax = Inf,
        tf = 6.0,
        n = 1001,
        x0 = x0,
        kv = 0.1,
    )
    evolution = evaluate(pbc, ps; kwargs...)
    traj, _ = evolution
    q1 = traj[1,:]
    q2 = traj[2,:] #.- traj[2,end]
    q1dot = traj[3,:]
    q2dot = traj[4,:]
    Hd = map(eachcol(traj)) do x
        q = x[1:2]
        qdot = x[3:4]
        momentum = MLBasedESC._M⁻¹(pbc,q) \ qdot
        return hamiltoniand(pbc, [q;momentum], ps)
    end
    timevec = range(0, kwargs.tf, length=size(traj,2))

    # shape decomposition
    l1 = 0.17f0
    l2 = 0.025f0
    link1xy(θ) = reverse(l1.*sincos(θ))
    rotor(θ) = Circle(Point2f(link1xy(θ)...), 1f0)

    # Animation setup
    f = Figure(resolution=round.(Int, (170.66,96)./2.54.*20))
    fps = 60
    indpersec = round(Int, (1/fps)/(kwargs.tf/(kwargs.n-1)), RoundUp)

    # Phase plot
    ax1 = Axis(f[1, 1][1,1],
        xlabel=L"Pendulum angle $q_1$ (rad)",
        ylabel=L"$\dot{q}_1$ (rad/s)", 
    )
    metaplt1 = lines!(ax1, q1, q1dot, color=:white)
    timestep = Node(1)
    phase_points = Node(Point2f[(q1[1],q1dot[1])])
    lineplt1 = lines!(ax1, phase_points, color=:black)
    mx1 = @lift([q1[$timestep]])
    my1 = @lift([q1dot[$timestep]])
    scplt1 = scatter!(ax1, mx1, my1, strokewidth=2, strokecolor=:black, color=:white)

    # Pendulum plot
    ax2 = Axis(f[1, 2],
        topspinevisible = !true,
        rightspinevisible = !true,
        leftspinevisible = !true,
        bottomspinevisible = !true,
        aspect=AxisAspect(1),
    )
    hidedecorations!(ax2)
    metaplt2 = lines!(ax2, 1.1*[-l1, l1], 1.1*[-l1 ,l1], color=:white)
    mx2 = @lift([
        0., -l1*sin(q1[$timestep])
    ])
    my2 = @lift([
        0., l1*cos(q1[$timestep])
    ])
    # ex2 = @lift(-l1*sin(q1[$timestep]))
    # ey2 = @lift( l1*cos(q1[$timestep]))
    circle = @lift( Circle(Point2f(-l1*sin(q1[$timestep]), l1*cos(q1[$timestep])), l2) )
    circlelinex = @lift([
        -l1*sin(q1[$timestep]), 
        -l1*sin(q1[$timestep])-l2*sin(q1[$timestep]+q2[$timestep])
    ])
    circleliney = @lift([
        l1*cos(q1[$timestep]), 
        l1*cos(q1[$timestep])+l2*cos(q1[$timestep]+q2[$timestep])
    ])
    lplt2 = lines!(ax2, mx2, my2, linewidth=10, color=:gray)
    # scplt3 = scatter!(ax2, ex2, ey2, markersize=20)
    poly!(ax2, circle, color = :pink)
    lines!(ax2, circlelinex, circleliney, color=:black, linewidth=1.0)
    scatter!(ax2, [0.,], [0.,], color=:green, markersize=20)

    # Hd plot
    ax3 = Axis(f[1, 1][2,1],
        xlabel="Time (s)", 
        ylabel=L"Hamiltonian $H_d^{\;\theta}$",
    )
    metaplt2 = lines!(ax3, timevec, Hd, color=:white)
    Hd_points = Node(Point2f[(timevec[1],Hd[1])])
    lineplt3 = lines!(ax3, Hd_points, color=:black)
    mx3 = @lift([timevec[$timestep]])
    my3 = @lift([Hd[$timestep]])
    scplt3 = scatter!(ax3, mx3, my3, strokewidth=2, strokecolor=:black, color=:white, glowcolor=:pink)

    # Animate    
    timestamps = range(1, 1001, step=indpersec)
    record(f, "plots/test_anim.mp4", timestamps, framerate=fps) do t
        phase_points[] = push!(phase_points[], Point2f(q1[t], q1dot[t]))
        Hd_points[] = push!(Hd_points[], Point2f(timevec[t], Hd[t]))
        timestep[] = t
    end
end
