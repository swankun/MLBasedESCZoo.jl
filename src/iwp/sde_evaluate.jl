export SDEProblem, evaluate_stochastic

function DifferentialEquations.SDEProblem(P::IDAPBCProblem, ps, sysps=(I1,I2,m3,1.0); kwargs...)
    Σ = [2pi/1024/4, 2pi/4096/4, 0.05, 0.05]
    _policy = policyfrom(P, umax=2.0, kv=2.5)
    policy(x) = _policy(x,ps)
    ∂u∂x(x) = ForwardDiff.gradient(policy, x)
    function f!(dx,x,p,t)
        u = policy(x)
        q1, q2, q1dot, q2dot = x
        I1, I2, m3, _ = p
        dx[1] = q1dot
        dx[2] = q2dot
        dx[3] = m3*sin(q1)/I1 - u/I1
        dx[4] = u/I2
    end
    function g!(dx,x,p,t)
        I1, I2, _, N = p
        du = sum(∂u∂x(x) .* Σ * N)
        dx[1] = 0
        dx[2] = 0
        dx[3] = -1/I1*du
        dx[4] = 1/I2*du
    end
    x0 = [3.,0,0,0]
    tspan = (0.0, 30.0)
    return SDEProblem{true}(f!, g!, x0, tspan, sysps)
end

function sysparams_dist(i, steps)
    m1 = m3*(1+i*2steps); m2 = m3*(1-i*2steps)
    I11 = I1*(1+i*2steps); I12 = I1*(1-i*2steps)
    I21 = I2*(1+i*2steps); I22 = I2*(1-i*2steps)
    @assert all([m1,m2,I11,I12,I21,I22] .> 0)
    dm3 = MixtureModel([Uniform(m1-m3*steps, m1+m3*steps), Uniform(m2-m3*steps, m2+m3*steps)], [1/2,1/2])
    dI1 = MixtureModel([Uniform(I11-I1*steps, I11+I1*steps), Uniform(I12-I1*steps, I12+I1*steps)], [1/2,1/2])
    dI2 = MixtureModel([Uniform(I21-I2*steps, I21+I2*steps), Uniform(I22-I2*steps, I22+I2*steps)], [1/2,1/2])
    return dI1, dI2, dm3
end

function evaluate_noise_effects(P::IDAPBCProblem, ps, N=10, sysps=(I1,I2,m3,1.0); make_plot=false)
    sde = SDEProblem(P, ps)
    _policy = policyfrom(P, umax=2.0, kv=2.5)
    policy(x) = _policy(x,ps)
    loss = zeros(N)

    x = Array(solve(sde, LambaEulerHeun(), saveat=0.1, p=sysps))
    u = map(policy, eachcol(x))
    loss[1] = sum( @. 1/2 * ( 2(1 - cos(x[1,:])) + x[3,:]^2 + u^2 ) )
    make_plot && plot((x, u))

    Threads.@threads for i = 2:N
        x = Array(solve(sde, LambaEulerHeun(), saveat=0.1, p=sysps))
        u = map(policy, eachcol(x))
        loss[i] = sum( @. 1/2 * ( 2(1 - cos(x[1,:])) + x[3,:]^2 + u^2 ) )
    end

    return loss
end

function evaluate_stochastic(P::IDAPBCProblem, ps)
    N = 20
    M = 10
    sysparams = fill((I1,I2,m3,1.0), M)
    loss = zeros(N,M)
    # for j = 1:M
    #     sysparams[j] = (I1, I2, m3, j) 
    #     loss[:,j] .= evaluate_noise_effects(P, ps, N, sysparams[j])
    # end
    # for (j, m3bar) in enumerate(range(0.8m3, 1.2m3, length=M))
    #     sysparams[j] = (I1, I2, m3bar, 1.0) 
    #     loss[:,j] .= evaluate_noise_effects(P, ps, N, sysparams[j])
    # end
    for (j, I1bar) in enumerate(range(0.8I1, 1.2I1, length=M))
        sysparams[j] = (I1bar, I2, m3, 1.0) 
        loss[:,j] .= evaluate_noise_effects(P, ps, N, sysparams[j])
    end
    return sysparams, loss
end


function bandplot(res::Tuple{T1,T2}, dim=3) where 
    {T1<:Vector{NTuple{4,Float64}}, T2<:Matrix{Float64}}
    ps, loss = res
    stat = mapreduce(vcat, eachcol(loss)) do l
        μ = mean(l)
        σ = std(l)
        μ, σ
    end

    # Interpolate
    xpoints = getindex.(ps, dim)
    xitp = range(first(xpoints), last(xpoints), length=length(xpoints))
    μ = CubicSplineInterpolation(xitp, first.(stat))
    σ = CubicSplineInterpolation(xitp, last.(stat))
    x = range(first(xitp), last(xitp), length=101) 
    μv = μ(x)
    σv = σ(x)

    # Plot
    fig = Figure()
    ax = Axis(fig[1,1])
    band!(ax, x, μv+σv, μv-σv,
        color=Makie.RGBA{Float32}(0.2f0,0.3f0,0.5f0,0.2f0),
        # transparency=!true,
        # diffuse=Float32[0.1,0.1,0.1],
        # backlight=0.5f0,
        # shading=true,
    )
    lines!(ax, x, μv)
    save("plots/bandplot.png", fig)
end
