export neuralpbc_system, setup_problem, trainloss, train!
export evaluate, plot

function zeroshift(q::AbstractVector, ::Any=nothing)
    return [
        q[1]
        1-q[2]
        q[3]
        1-q[4]
        q[5]
        q[6]
    ]
end
function MLBasedESC.jacobian(::typeof(zeroshift), q,::Any=nothing)
    diagm([1,-1,1,-1,1,1])
end

function MLBasedESC.NeuralPBC(::ReactionWheelPendulum)
    Hd = FastChain(
        FastDense(6, 12, elu, bias=true),
        FastDense(12, 3, elu, bias=true),
        FastDense(3, 1, bias=true)
        # zeroshift,
        # FastDense(6, 18, tanh, bias=false),
        # FastDense(18, 12, tanh, bias=false),
        # FastDense(12, 1, square, bias=false)
    )
    return NeuralPBC(6,Hd)
end

function policyfrom(P::NeuralPBC; umax=Inf, lqrmax=umax)
    u_neuralpbc(x,p) = begin
        sq1, cq1, sq2, cq2, q1dot, q2dot = x
        dist = norm([1-cq1, 0.5*q1dot/10])
        # dist = 1 - cq1
        eta0 = exp(-(5*dist)^2)
        eta1 = 1 - eta0
        lqr = -dot(LQR, [sq1, sq2, q1dot, q2dot])
        swing = P(x,p)
        effort = eta0*lqr + eta1*swing
        return clamp(effort, -umax, umax)
        # if (1-cq1 < 1-cosd(30)) && abs(q1dot) < 5
        #     effort = -dot(LQR, [sq1, sq2, q1dot, q2dot])
        #     return clamp(effort, -lqrmax, lqrmax)
        # else
        #     effort = P(x,p)
        #     return clamp(effort, -umax, umax)
        # end
    end
end

function quadneuralnet(x, f, ps)
    Q = f(x,ps)
    dot(x, Q*x)
end

function MLBasedESC.ParametricControlSystem(::ReactionWheelPendulum, 
    prob::NeuralPBC; kwargs...
)
    return ParametricControlSystem{false}(
        eomwrap, policyfrom(prob; kwargs...), 6
    )
end

function saveweights(pbc::NeuralPBC, ps)
    filename = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    parentdir = "/tmp/jl_MLBasedESCZoo/NeuralPBC/"
    !isdir(parentdir) && run(`mkdir -p $parentdir`)
    # parentdir = "src/iwp/models/"
    # filename = "neuralpbc_1ring_candidate"
    weight_filepath = parentdir * filename * ".bson"
    BSON.@save weight_filepath pbc ps
end


function train!(::ReactionWheelPendulum, pbc::NeuralPBC, ps; 
    tf=3.0, dt=0.1, umax=1.0, lqrmax=1.5,
    batchsize=4, maxiters=1000, replaybuffer=5
)
    sys = ParametricControlSystem(ReactionWheelPendulum(), pbc, umax=umax, lqrmax=lqrmax)
    # sys = ParametricControlSystem{false}(
    #     eomwrap, (x,θ)->pbc(x,θ), 6
    # )
    dist(x) = begin
        sq1, cq1, sq2, cq2, q1dot, q2dot = x
        return 4(1-cq1) + (q1dot^2)/4 + (q2dot^2)/5
    end
    loss = SetDistanceLoss(dist, wrap(zeros(4)), 1/100)
    loss_tspan = range(tf/2, tf, step=dt)
    losstypestr = typeof(loss).name.name |> string
    tspan = (zero(tf), tf)
    prob = ODEProblem(sys, ps, tspan)
    optimizer = Flux.ADAM()
    sampler(batchsize, ps) = customdagger(prob, batchsize, ps, tf=2tf)
    data = reduce(vcat, sampler(batchsize, ps) for _=1:replaybuffer)
    push!(data, wrap([pi,0,0,0]))
    push!(data, wrap([-pi,0,0,0]))
    lastsave = Dates.now()
    for nepoch in 1:maxiters
        data = vcat(last(data,(replaybuffer-1)*batchsize), sampler(batchsize, ps))
        dataloader = Flux.Data.DataLoader(data; batchsize=batchsize, shuffle=true)
        max_batch = round(Int, dataloader.imax / dataloader.batchsize, RoundUp)
        epochloss = 0
        nbatch = 1
        for batch in dataloader
            gs, ls = gradient(loss, prob, batch, ps; dt=loss_tspan)
            epochloss = +(epochloss, ls)
            if !isnothing(gs) && !any(isnan, gs)
                Flux.Optimise.update!(optimizer, ps, gs)
            end
            MLBasedESC.bstatus(nbatch, max_batch, ls)
            nbatch += 1
            if iszero(nbatch % 10) || (nbatch == max_batch)
                plot(pbc, ps, umax=umax, lqrmax=lqrmax, tf=tf, 
                    x0=unwrap(first(batch)));
            end
        end
        if Dates.now()-lastsave > Minute(1)
            saveweights(pbc, ps)
            lastsave = Dates.now()
        end
        MLBasedESC.estatus(losstypestr, nepoch, epochloss/nbatch, maxiters)
    end
end

function customdagger(::ReactionWheelPendulum, pbc::NeuralPBC, n, θ, tf; kwargs...)
    sys = ParametricControlSystem(ReactionWheelPendulum(), pbc; kwargs...)
    tspan = (zero(tf), tf)
    prob = ODEProblem(sys, θ, tspan)
    return customdagger(prob, n, θ; tf=tf)
end
function customdagger(prob, n, θ; tf=last(prob.tspan))
    xstar = zeros(4)
    Σ = LinearAlgebra.I
    D = MvNormal(xstar, Σ)
    if rand(Bool)
        return map(wrap, eachcol(rand(D,n)))
    else
        xinit = rand(D)
        xinit[1:2] *= pi/2
        xinit[3:4] .= 0
        x0 = wrap(xinit)
        tspan = (zero(tf), tf)
        tsave = range(first(tspan), last(tspan), length=n)
        x = trajectory(remake(prob, tspan=tspan), x0, θ, saveat=tsave)
        return collect(Vector{eltype(x)},eachcol(x))
    end
end

function evaluate(pbc::NeuralPBC, θ; umax=Inf, lqrmax=umax, kwargs...)
    policy = policyfrom(pbc, umax=umax, lqrmax=lqrmax)
    wpolicy(x, ps) = begin
        policy(wrap(x), ps)
    end
    sys = ParametricControlSystem{false}(eom, wpolicy,4)
    x = simulate(sys, θ; kwargs...)
    u = map(_1->wpolicy(_1,θ), eachcol(x))
    return x, u
end


function plot(evolution::Tuple{AbstractMatrix,AbstractVector}; out=true)
    #=
    Usage: 
    plot(evaluate(pbc, ps, umax=2.0, tf=20.0, x0=[pi*randn(), pi*randn(),0,0]))
    =#
    traj, ctrl = evolution
    t = range(0, 1, length=size(traj,2))
    fig = Figure(size=(800,800))
    labels = ("q1","q2","q1dot","q2dot")
    for (x, xstr, ij) = zip(eachrow(traj), labels, Iterators.product(1:2,1:2))
        Axis(fig[ij...], title=xstr)
        lines!(t, x)
    end
    Axis(fig[3,1], title="control")
    lines!(t, ctrl)
    # Axis(fig[3,1], title="Phase space")
    # lines!(traj[1,:], traj[3,:])
    out && save("plots/out3.png", fig)
    return fig
end
function plot(pbc::NeuralPBC, θ; out=true, kwargs...)
    evolution = evaluate(pbc, θ; kwargs...)
    # traj, _ = evaluate(pbc, θ; kwargs...)
    # ham = map(eachcol(traj)) do x0
    #     x0bar = wrap(x0) 
    #     pbc.Hd(x0bar, θ)[1]
    # end
    # evolution = (traj, ham)
    fig = plot(evolution, out=false)
    X = range(-pi, pi, length=50)
    Y = range(-pi, pi, length=50)
    Z = zeros(50,50)
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            x = wrap([X[ix],Y[iy],0,0])
            # x = wrap([X[ix],0,Y[iy],0])
            Z[ix,iy] = pbc.Hd(x,θ)[1]
        end
    end
    zmin = minimum(Z)
    zmax = maximum(Z)
    ax, ct = contour(fig[3,2][1,1], X,Y,Z, colormap=:gnuplot,
        levels=range(zmin, zmax, length=10)
    )
    ax.title = "Hd"
    Colorbar(fig[3,2][1,2], ct)
    # Axis(fig[4,1], title="Hd(t)")
    # Hd = map(x->pbc.Hd(wrap(x),θ)[1], eachcol(first(evolution)))
    # t = range(0, 1, length=size(evolution[1],2))
    # lines!(t, Hd)
    out && save("plots/out3.png", fig)
    return fig
end

function contour_Hd_bayesian()
    Hd = FastChain(
        FastDense(6, 3, elu),
        FastDense(3, 3, elu),
        FastDense(3, 1)
    )
    pbc = NeuralPBC(6,Hd)
    raw = BSON.load("/tmp/normalm3_with1_1_4.bson")
    ps = raw[:hα]
    contour_Hd(pbc, ps)
end
function contour_Hd(pbc::NeuralPBC, θ)
    N = 101
    X = range(-pi, pi, length=N)
    Y = range(-pi, pi, length=N)
    Z = zeros(N,N)
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            x = wrap([X[ix],Y[iy],0,0])
            # x = wrap([X[ix],0,Y[iy],0])
            Z[ix,iy] = pbc.Hd(x,θ)[1]
        end
    end
    fig = Figure(resolution=(4,3).*200, figure_padding=0,)
    zmin = minimum(Z)
    zmax = maximum(Z)
    ax = Axis(fig[1,1],
        xlabel=L"$q_1$",
        ylabel=L"$q_2$",
        xlabelpadding=0.0,
        ylabelpadding=0.0,
        xticks=piticks(1),
        yticks=piticks(1),
        title=L"Level sets of $H_d$"
    )
    # hidespines!(ax)
    hidedecorations!(ax, 
        # ticks=false,
        ticklabels=false, 
        label=false
    )
    # tightlimits!(ax)
    ct = contour!(X,Y,Z, colormap=:grays, linewidth=2.0,
        levels=10
        # levels=[0,0.001,0.005,0.01,0.05,0.1,1,10,100]
    )
    # Colorbar(fig[1,1][1,2], ct)
    save("plots/neuralpbc_contour.eps", fig)
    save("plots/neuralpbc_contour.png", fig)
end

function plot_uru(pbc::NeuralPBC, θ; out=true)
    # ps = BSON.load("src/iwp/models/neuralpbc_20211228.bson")[:ps]
    # ps = BSON.load("src/iwp/models/neuralpbc_20220124.bson")[:ps]
    N = 21
    X = range(-pi, pi, length=N)
    Y = range(-pi, pi, length=N)
    Z = zeros(N,N)
    D = Vector{Tuple{Int,Int,Matrix{Float64}}}()
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            if 1-cos(X[ix]) > 1-cosd(30) && abs(Y[iy]) > 5
                continue
            end
            x, u = evaluate(pbc, θ, x0=[X[ix],0,Y[iy],0]; umax=2.0, tf=30.0)
            Z[ix,iy] = sum(abs2,u)
            q1 = x[1,end]
            q1dot = x[3,end]
            if !(1-cos(q1) < 1-cosd(20)) && !(abs(q1dot) < 5)
                @warn "Failed to catch x0=$([X[ix],0,Y[iy],0]). Terminal state xf=$(x[:,end])."
                push!(D, (ix,iy,x))
            end
        end
    end
    @show Zbar = maximum(Z)
    @show length(D)
    for d in D
        ix, iy, x = d
        # x0 = (X[ix], Y[iy])
        # @show x0
        # @show x[:,end]
        Z[ix,iy] = Zbar
    end
    
    fig = Figure(resolution=(3.167,3).*300)
    majorfontsize = 30*2
    minorfontsize = 24*2
    ax_hm = Axis(
        fig[1,1][1,1], 
        aspect=AxisAspect(1),
        # xticks=(range(-pi,pi,step=0.5pi), ["-π", "-π/2", "0", "π/2", "π"]),
        xticks=piticks(0.5),
        yticks=-3:1:3,
        title=L"u^\top R u", 
        # titlesize=majorfontsize, 
        # xlabel="Pendulum angle (rad)", xlabelfont="CMU Serif", xlabelsize=minorfontsize,
        # ylabel="Pendulum velocity (rad/s)", ylabelfont="CMU Serif", ylabelsize=minorfontsize,
        # xticklabelfont="CMU Serif", xticklabelsize=minorfontsize,
        # yticklabelfont="CMU Serif", yticklabelsize=minorfontsize,
    )
    hm = heatmap!(ax_hm, X,Y,Z, 
        colormap=:RdPu_4, 
        # colorrange=(0.0,20.0)
    )
    Colorbar(fig[1,1][1,2], hm, 
        # label=L"u^\top Ru", labelsize=30, 
        # ticklabelfont="CMU Serif", ticklabelsize=minorfontsize
    )
    out && save("plots/uRu.png", fig)
    out && save("plots/uRu.svg", fig)
    return fig
end

function Hd_q1(pbc::NeuralPBC, ps)
    fig = Figure()
    ax = Axis(fig[1,1])
    lines!(-pi:pi/50:pi, (x)->pbc.Hd(wrap([x,0,0,0]), ps)[1])
    lines!(-pi:pi/50:pi, (x)->Vd_groundtruth([x,0,0,0], ps)[1])
    save("plots/out.png", fig)
end 