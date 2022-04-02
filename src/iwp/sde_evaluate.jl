export SDEProblem, evaluate_stochastic

function DifferentialEquations.SDEProblem(P::IDAPBCProblem, ps, sysps=(I1,I2,m3,1.0); kwargs...)
    Σ = [2pi/1024/4, 2pi/4096/4, 2pi/1024/4, 2pi/4096/4] .* [1, 1, 10, 10]
    _policy = policyfrom(P, umax=2.0, kv=1.0)
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
    tspan = (0.0, 10.0)
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
    _policy = policyfrom(P, umax=2.0, kv=1.0)
    policy(x) = _policy(x,ps)
    loss = zeros(N)

    x = Array(solve(sde, LambaEulerHeun(), saveat=0.1, p=sysps))
    u = map(policy, eachcol(x))
    loss[1] = sum( @. 1/2 * ( 2(1 - cos(x[1,:])) + abs(x[3,:]) + u^2 ) )
    make_plot && plot((x, u))

    Threads.@threads for i = 2:N
        x = Array(solve(sde, LambaEulerHeun(), saveat=0.1, p=sysps))
        u = map(policy, eachcol(x))
        loss[i] = sum( @. 1/2 * ( 2(1 - cos(x[1,:])) + abs(x[3,:]) + u^2 ) )
    end

    return loss
end

function evaluate_stochastic(P::IDAPBCProblem, ps)
    N = 20
    M = 5
    sysparams = fill((I1,I2,m3,1.0), M)
    loss = zeros(N,M)
    for j = 1:M
        D = sysparams_dist(j-1, 0.05)
        Threads.@threads for i = 1:N
            sysparams[j] = tuple(rand.(D)..., 1.0)
            loss[i,j] = evaluate_noise_effects(P, ps, 1, sysparams[j])[1]
        end
    end
    # for j = 1:M
    #     sysparams[j] = (I1, I2, m3, j) 
    #     loss[:,j] .= evaluate_noise_effects(P, ps, N, sysparams[j])
    # end
    # for (j, m3bar) in enumerate(range(0.8m3, 1.2m3, length=M))
    #     sysparams[j] = (I1, I2, m3bar, 1.0) 
    #     loss[:,j] .= evaluate_noise_effects(P, ps, N, sysparams[j])
    # end
    # for (j, I1bar) in enumerate(range(0.8I1, 1.2I1, length=M))
    #     sysparams[j] = (I1bar, I2, m3, 1.0) 
    #     loss[:,j] .= evaluate_noise_effects(P, ps, N, sysparams[j])
    # end
    return sysparams, loss
end


function bandplot(res::Tuple{T1,T2}, dim=3) where 
    {T1<:Vector{NTuple{4,Float64}}, T2}
    ps, stat = res
    N = length(ps)
    # stat = mapreduce(vcat, eachcol(loss)) do l
    #     μ = mean(l)
    #     σ = std(l)
    #     μ, σ
    # end

    # Interpolate
    xpoints = iszero(dim) ? range(0, step=0.1, length=N) : getindex.(ps, dim)
    xitp = range(first(xpoints), last(xpoints), length=length(xpoints))
    μ = CubicSplineInterpolation(xitp, first.(stat))
    σ = CubicSplineInterpolation(xitp, last.(stat))
    x = range(first(xitp), last(xitp), length=101) 
    μv = μ(x)
    σv = σ(x)

    # Plot
    fig = Figure()
    ax = Axis(fig[1,1],
        # title=L"U_{[n-\delta, n+\delta]} \times U_{[\delta+0.1, \delta+0.2]}",
        # xticks=(xpoints, [L"U_{[%$(round(0.9*i,digits=1)),%$(round(1.1*i,digits=1))]}" for i=1:N]),
        # xticklabelrotation=pi/6,
        # xticks=(xpoints, string.(vcat(0,collect(0.15:0.1:0.45)))),
        xlabel="Average error from nominal parameters"
    )
    band!(ax, x, μv+σv, max.(0, μv-σv),
        color=Makie.RGBA{Float32}(0.2f0,0.3f0,0.5f0,0.2f0),
        # transparency=!true,
        # diffuse=Float32[0.1,0.1,0.1],
        # backlight=0.5f0,
        # shading=true,
    )
    lines!(ax, x, μv)
    save("plots/bandplot.png", fig)
end

function load_results()
    avg_error = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0] / 100
    mean_loss_idapbc_bays = [1426.315761739796, 1377.3561011910729, 1758.649813444647, 2329.265210119561, 1617.5074606834935, 2385.9294496193756]
    std_loss_idapbc_bays = [162.8230744271456, 327.51639276177775, 664.9102088777273, 1197.1879844817151, 1295.399728762258, 1287.3320779012363]

    mean_loss_idapbc_deter = [1500.713265363227, 2593.5978501701256, 2378.682175944044, 2708.4706787837267, 9465.571584866691, 7624.2795704683795]
    std_loss_idapbc_deter =  [1951.812038908012, 3554.240174055424, 2859.8706290924383, 4198.077656962136, 6577.378646241127, 7471.971596845208]

    # avg_error = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0] / 100
    # mean_loss_idapbc_bays = [1875.0938900998783, 1675.9833973791526, 1822.6201698397133, 2171.8262754704333, 2033.1851354355458, 2435.4232625524974]
    # std_loss_idapbc_bays = [760.4506704034075, 311.97113336686084, 634.9294210495706, 761.0002155754705, 694.306504815723, 988.7604154360355]
        
    # mean_loss_idapbc_deter = [1507.9908487508005, 2855.224256806579, 3129.553568695179, 5248.157817476253, 6714.3536157007065, 8192.95721648705]
    # std_loss_idapbc_deter = [155.82833052390683, 3351.358194288491, 3194.2180067351237, 4605.997103334661, 6347.886980644229, 8616.814333918048]

    mean_loss_neuralpbc_bays = [1744.3401940707463, 1288.3548822482257, 2693.166161909191, 2622.8607975786663, 3264.6158609181884, 3715.6889048095027]
    std_loss_neuralpbc_bays = [3991.2922221197964, 1848.8110741143594, 2437.9983332388997, 3631.8794618642487, 2757.5582684964816, 4080.0875459324707]
    
    mean_loss_neuralpbc_deter = [2795.2178525105087, 2036.9246126029432, 2447.197947137953, 4049.90161650543, 5762.632707814728, 6809.374679844519]
    std_loss_neuralpbc_deter = [2795.2178525105087, 2036.9246126029432, 2447.197947137953, 4049.90161650543, 5762.632707814728, 6809.374679844519]
    
    xpoints = avg_error
    xitp = range(first(xpoints), last(xpoints), length=length(xpoints))
    x = range(first(xitp), last(xitp), length=101) 
    function itp(arr)
        y = CubicSplineInterpolation(xitp, arr)
        y(x)
    end

    fig = Figure(resolution=(900,300))
    ax = Axis(fig[1,1],
        # title=L"U_{[n-\delta, n+\delta]} \times U_{[\delta+0.1, \delta+0.2]}",
        # xticks=(xpoints, [L"U_{[%$(round(0.9*i,digits=1)),%$(round(1.1*i,digits=1))]}" for i=1:N]),
        # xticklabelrotation=pi/6,
        yticks=0:5000:20000,
        ytickformat=xs->([L"%$(round(x/1e4, digits=1))" for x in xs]),
        xticks=xpoints,
        xlabel="Average error from nominal parameters",
        # ylabel="Performance metric"
    )
    μv = itp(mean_loss_idapbc_bays)
    σv = itp(std_loss_idapbc_bays)
    band!(ax, x, μv+σv, max.(0, μv-σv),
        color=Makie.RGBA{Float32}(0.2f0,0.3f0,0.5f0,0.2f0),
    )
    l1 = lines!(ax, x, μv, linestyle=:dot)
    s1 = scatter!(ax, avg_error, mean_loss_idapbc_bays, marker=:star4, markersize=24)
    μv = itp(mean_loss_idapbc_deter)
    σv = itp(std_loss_idapbc_deter)
    band!(ax, x, μv+σv, max.(0, μv-σv),
        color=Makie.RGBA{Float32}(0.5f0,0.3f0,0.2f0,0.2f0),
    )
    l2 = lines!(ax, x, μv)
    s2 = scatter!(ax, avg_error, mean_loss_idapbc_deter)
    ylims!(ax, -1000, 17000)
    Legend(fig[1,1], [[l1,s1],[l2,s2]], ["Bayesian", "Deterministic"], 
        orientation=:horizontal,
        tellheight = false,
        tellwidth = false,
        halign = :left,
        valign = :top,
        framevisible = false,
    )
    save("plots/bandplot1.png", fig)
    save("plots/bandplot1.eps", fig)

    fig = Figure(resolution=(800,500))
    ax = Axis(fig[1,1],
        # title=L"U_{[n-\delta, n+\delta]} \times U_{[\delta+0.1, \delta+0.2]}",
        # xticks=(xpoints, [L"U_{[%$(round(0.9*i,digits=1)),%$(round(1.1*i,digits=1))]}" for i=1:N]),
        # xticklabelrotation=pi/6,
        ytickformat=xs->([L"%$(round(x/1e4,digits=1))" for x in xs]),
        xticks=xpoints,
        xlabel="Average error from nominal parameters",
        ylabel="Performance metric"
    )
    μv = itp(mean_loss_neuralpbc_bays)
    σv = itp(std_loss_neuralpbc_bays)
    band!(ax, x, μv+σv, max.(0, μv-σv),
        color=Makie.RGBA{Float32}(0.2f0,0.3f0,0.5f0,0.2f0),
    )
    l1 = lines!(ax, x, μv, linestyle=:dot)
    s1 = scatter!(ax, avg_error, mean_loss_neuralpbc_bays, marker=:star4, markersize=24)
    μv = itp(mean_loss_neuralpbc_deter)
    σv = itp(std_loss_neuralpbc_deter)
    band!(ax, x, μv+σv, max.(0, μv-σv),
        color=Makie.RGBA{Float32}(0.5f0,0.3f0,0.2f0,0.2f0),
    )
    l2 = lines!(ax, x, μv)
    s2 = scatter!(ax, avg_error, mean_loss_neuralpbc_deter)
    ylims!(ax, 0, 15000)
    Legend(fig[1,1], [[l1,s1],[l2,s2]], ["Bayesian", "Deterministic"], 
        orientation=:horizontal,
        tellheight = false,
        tellwidth = false,
        halign = :center,
        valign = :top,
        framevisible = false,
    )
    save("plots/bandplot2.png", fig)
    save("plots/bandplot2.eps", fig)
end