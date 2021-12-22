export neuralpbc_system, setup_problem, trainloss, train!
export evaluate, plot

function MLBasedESC.NeuralPBC(::ReactionWheelPendulum)
    Hd = FastChain(
        FastDense(6, 10, elu, bias=true),
        FastDense(10, 5, elu, bias=true),
        FastDense(5, 1, bias=true)
    )
    return NeuralPBC(6,Hd)
end

function policyfrom(P::NeuralPBC; umax=Inf, lqrmax=umax)
    u_neuralpbc(x,p) = begin
        sq1, cq1, sq2, cq2, q1dot, q2dot = x
        if (1-cq1 < 1-cosd(10)) && abs(q1dot) < 1
            effort = -dot(LQR, [sq1, sq2, q1dot, q2dot])
            return clamp(effort, -lqrmax, lqrmax)
        else
            effort = P(x,p)
            return clamp(effort, -umax, umax)
        end
    end
end


function MLBasedESC.ParametricControlSystem(::ReactionWheelPendulum, 
    prob::NeuralPBC; kwargs...
)
    return ParametricControlSystem{false}(
        eomwrap, policyfrom(prob; kwargs...), 6
    )
end


function train!(::ReactionWheelPendulum, pbc::NeuralPBC, ps; 
    tf=3.0, dt=0.1, umax=0.5, lqrmax=1.5,
    batchsize=4, maxiters=1000, replaybuffer=batchsize*5
)
    sys = ParametricControlSystem(ReactionWheelPendulum(), pbc, umax=umax, lqrmax=lqrmax)
    dist(x) = begin
        sq1, cq1, sq2, cq2, q1dot, q2dot = x
        return (1-cq1) + q1dot^2 + q2dot^2
    end
    loss = SetDistanceLoss(dist, wrap(zeros(4)), 0.1)
    losstypestr = typeof(loss).name.name |> string
    tspan = (zero(tf), tf)
    prob = ODEProblem(sys, ps, tspan)
    optimizer = Flux.ADAM()
    # sampler = MLBasedESC.CustomDagger(prob, zeros(4), wrap)
    sampler(batchsize, ps) = customdagger(prob, batchsize, ps, tf=6.0)
    data = reduce(vcat, sampler(batchsize, ps) for _=1:replaybuffer)
    for nepoch in 1:maxiters
        data = vcat(last(data,replaybuffer-batchsize), sampler(batchsize, ps))
        dataloader = Flux.Data.DataLoader(data; batchsize=batchsize, shuffle=true)
        max_batch = round(Int, dataloader.imax / dataloader.batchsize, RoundUp)
        epochloss = 0
        nbatch = 1
        for batch in dataloader
            gs, ls = gradient(loss, prob, batch, ps; dt=dt)
            epochloss = +(epochloss, ls)
            if !isnothing(gs)
                Flux.Optimise.update!(optimizer, ps, gs)
            end
            MLBasedESC.bstatus(nbatch, max_batch, ls)
            nbatch += 1
        end
        plot(pbc, ps, umax=umax, lqrmax=lqrmax, tf=20.0, 
            x0=[3,0*randn(),0randn(),0randn()]);
        MLBasedESC.estatus(losstypestr, nepoch, epochloss/nbatch, maxiters)
    end
end

function customdagger(prob, n, θ; tf=last(prob.tspan))
    xstar = zeros(4)
    Σ = LinearAlgebra.I
    D = MvNormal(xstar, Σ)
    if rand(Bool)
        return map(wrap, eachcol(rand(D,n)))
    else
        xinit = rand(D)
        xinit[1:2] *= pi
        x0 = wrap(xinit)
        tspan = (zero(tf), tf)
        tsave = range(first(tspan), last(tspan), length=101)
        x = trajectory(remake(prob, tspan=tspan), x0, θ, saveat=tsave)
        return collect.(rand(collect(eachcol(x)), n))
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
    fig = Figure()
    labels = ("q1","q2","q1dot","q2dot")
    for (x, xstr, ij) = zip(eachrow(traj), labels, Iterators.product(1:2,1:2))
        Axis(fig[ij...], title=xstr)
        lines!(t, x)
    end
    Axis(fig[3,1], title="control")
    lines!(t, ctrl)
    out && save("out.png", fig)
    return fig
end
function plot(pbc::NeuralPBC, θ; out=true, kwargs...)
    #=
    Usage: 
    plot(evaluate(pbc, ps, umax=2.0, tf=20.0, x0=[pi*randn(), pi*randn(),0,0]))
    =#
    evolution = evaluate(pbc, θ; kwargs...)
    fig = plot(evolution, out=false)
    X = range(-pi, pi, length=50)
    Y = range(-pi, pi, length=50)
    Z = zeros(50,50)
    Threads.@threads for ix in 1:length(X)
        for iy in 1:length(Y) 
            x = wrap([X[ix],Y[iy],0,0])
            Z[ix,iy] = pbc.Hd(x,θ)[1]
        end
    end
    # Z = [pbc.Hd(wrap([x,y,0,0]),θ)[1] for x in X, y in Y]
    Axis(fig[3,2], title="Hd")
    contour!(X,Y,Z)
    out && save("out.png", fig)
    return fig
end