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

function policyfrom(P::NeuralPBC; umax=Inf)
    u_neuralpbc(x,p) = begin
        sq1, cq1, sq2, cq2, q1dot, q2dot = x
        if (1-cq1 < 1-cosd(10)) && abs(q1dot) < 5
            effort = -dot(LQR, [sq1, sq2, q1dot, q2dot])
        else
            effort = P(x,p)
        end
        return clamp(effort, -umax, umax)
    end
end


function MLBasedESC.ParametricControlSystem(::ReactionWheelPendulum, 
    prob::NeuralPBC; umax=Inf
)
    return ParametricControlSystem{false}(
        eomwrap, policyfrom(prob, umax=umax), 6
    )
end


function train!(::ReactionWheelPendulum, pbc::NeuralPBC, ps; 
    tf=3.0, dt=0.1, batchsize=4, maxiters=1000, replaybuffer=10
)
    sys = ParametricControlSystem(ReactionWheelPendulum(), pbc, umax=2.0)
    dist(x) = begin
        sq1, cq1, sq2, cq2, q1dot, q2dot = x
        return (1-cq1) + q1dot^2 + q2dot^2
    end
    loss = SetDistanceLoss(dist, wrap(zeros(4)), 0.1)
    tspan = (zero(tf), tf)
    prob = ODEProblem(sys, ps, tspan)
    optimizer = Flux.ADAM()
    # sampler = MLBasedESC.CustomDagger(prob, zeros(4), wrap)
    sampler(batchsize, ps) = customdagger(prob, batchsize, ps, tf=6.0)
    data = reduce(vcat, sampler(batchsize, ps) for _=1:replaybuffer)
    for i in 1:maxiters
        data = vcat(last(data,replaybuffer-batchsize), sampler(batchsize, ps))
        dataloader = Flux.Data.DataLoader(data; batchsize=batchsize, shuffle=true)
        epochloss = 0
        for batch in dataloader
            gs, ls = gradient(loss, prob, batch, ps; dt=dt)
            epochloss = max(epochloss, ls)
            if !isnothing(gs)
                Flux.Optimise.update!(optimizer, ps, gs)
            end
        end
        @info "Loss = $(epochloss)"
    end
end

function customdagger(prob, n, θ; tf=last(prob.tspan))
    xstar = zeros(4)
    Σ = LinearAlgebra.I
    D = MvNormal(xstar, Σ)
    if rand(Bool)
        return map(wrap, eachcol(rand(D,n)))
    else
        x0 = wrap(2*rand(D))
        tspan = (zero(tf), tf)
        tsave = range(first(tspan), last(tspan), length=101)
        x = trajectory(remake(prob, tspan=tspan), x0, θ, saveat=tsave)
        return collect.(rand(collect(eachcol(x)), n))
    end
end

function evaluate(pbc::NeuralPBC, θ; umax=Inf, kwargs...)
    policy = policyfrom(pbc, umax=umax)
    wpolicy(x, ps) = begin
        policy(wrap(x), ps)
    end
    sys = ParametricControlSystem{false}(eom, wpolicy,4)
    x = simulate(sys, θ; kwargs...)
    u = map(_1->wpolicy(_1,θ), eachcol(x))
    return x, u
end

function plot(traj::AbstractMatrix, index)
    t = range(0, 1, length=size(traj,2))
    save("out.png", lines(t, traj[index, :]))
end
