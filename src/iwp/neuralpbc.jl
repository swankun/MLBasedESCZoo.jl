export neuralpbc_system, setup_problem, trainloss, train!

function neuralpbc_system(::ReactionWheelPendulum)
    Hd = FastChain(
        FastDense(6, 10, elu, bias=false),
        FastDense(10, 5, elu, bias=false),
        FastDense(5, 1, bias=false)
    )
    pbc = NeuralPBC(6,Hd)
    policy(x,ps) = controller(pbc, x, ps)
    sys = ParametricControlSystem{!true}(eomwrap,policy,6)
    return sys, pbc
end


function trainloss(prob, x0, ps, n=101)
    tsave = range(prob.tspan..., length=n)
    x = MLBasedESC.trajectory(prob, x0, ps; 
        saveat=tsave, 
        sensealg=DiffEqFlux.ReverseDiffAdjoint()
    )
    mapreduce(min, eachcol(x)) do state
        sq1, cq1, sq2, cq2, q1dot, q2dot = state
        dist = (1-cq1) + q1dot^2 + q2dot^2
        ifelse(dist <= 0.1, zero(dist), dist)
    end
end

function train!(prob::ODEProblem, ps; n=101)
    optimizer = Flux.ADAM()
    for i = 1:100
        x0 = [
            0.3pi*(2rand()-1), 
            0.3pi*(2rand()-1), 
            1/2*(2rand()-1),
            1/2*(2rand()-1)
        ] |> wrap
        gs = Flux.gradient(_3->trainloss(prob,x0,_3,n), ps)
        Flux.Optimise.update!(optimizer, ps, gs[1])
        if iszero(i % 10)
            @info "Batch $i"
        end
    end
end

function setup_problem(sys, ps, tf)
    tspan = (zero(tf), tf)
    ODEProblem(sys, ps, tspan)
end
