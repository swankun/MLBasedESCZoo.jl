export ReactionWheelPendulum, wrap, unwrap, simulate


struct ReactionWheelPendulum end

#======================================
lr = 0.172
mr_old = 0.084 + 4 * 0.12946
mr_new = 0.084 + 1 * 0.12946
I1 = 0.0455 - mr_old*lr^2 + mr_new*lr^2
I2 = 0.00425 - 4*9.484e-4 + 1*9.484e-4
ml = 0.183 - mr_old*lr + mr_new*lr
======================================#
const I1 = 0.0455
const I2 = 0.00425
const m3 = 0.183*9.81
# const I1 = 0.03401
# const I2 = 0.001405
# const m3 = 0.1162*9.81
const b1 = 5/1000 #2.5/1000
const b2 = 1/100  #5/1000
const M⁻¹ = inv(diagm([I1, I2]))
const G = [-1.0, 1.0]
const G⊥ = [1.0 1.0]
const LQR = [-3.8150228674227527, -0.010000000000018733, -0.6570687716257183, -0.0065154336851863896]

function inmap(q,::Any=nothing)
    return [
        1-cos(q[1])
        sin(q[1])
        1-cos(q[2])
        sin(q[2])
    ]
end
function MLBasedESC.jacobian(::typeof(inmap), q,::Any=nothing)
    qbar = inmap(q)
    [
        qbar[2] 0
        1-qbar[1] 0
        0 qbar[4] 
        0 1-qbar[3] 
    ]
end

rpm2rads(rpm) = rpm * 1 / 60 * 2 * pi

function friction_model(velocity)
    # mathworks.com/help/physmod/simscape/ref/translationalfriction.html
    Fbrk = 0.42*0.23 # observed from motor driver -- 0.42 amps
    Fc = 0.67*Fbrk
    breakaway_vel = rpm2rads(1)
    stribeck_vel = breakaway_vel*sqrt(2)
    coulomb_vel = breakaway_vel/10
    f_viscous = b2*velocity
    f_coulumb = Fc*tanh(velocity/coulomb_vel)
    f_stribeck = sqrt(2*exp(1)) * (Fbrk - Fc) * exp(-(velocity/stribeck_vel)^2) * velocity/stribeck_vel
    return f_viscous + f_coulumb + f_stribeck
end

function eom!(dx,x,u)
    q1, q2, q1dot, q2dot = x
    dx[1] = q1dot
    dx[2] = q2dot
    dx[3] = m3*sin(q1)/I1 - u/I1# - b1*q1dot/I1
    dx[4] = u/I2# - friction_model(q2dot)/I2
end
function eom(x,u)
    dx = similar(x)
    eom!(dx,x,u)
    return dx
end

function eomwrap!(dx,x,u)
    sq1, cq1, sq2, cq2, q1dot, q2dot = x
    I1bar = 1I1
    I2bar = 1I2
    m3bar = 1m3
    ϵ = 0.01
    dx[1] = cq1*q1dot - ϵ*sq1*(sq1^2 + cq1^2 - 1)
    dx[2] = -sq1*q1dot - ϵ*cq1*(sq1^2 + cq1^2 - 1)
    dx[3] = cq2*q2dot - ϵ*sq2*(sq2^2 + cq2^2 - 1)
    dx[4] = -sq2*q2dot - ϵ*cq2*(sq2^2 + cq2^2 - 1)
    dx[5] = m3bar*sq1/I1bar - u/I1bar - b1*q1dot/I1bar
    dx[6] = u/I2bar - friction_model(q2dot)/I2bar
end
function eomwrap(x,u)
    dx = similar(x)
    eomwrap!(dx,x,u)
    return dx
end
function wrap(x::AbstractVector)
    q1, q2, q1dot, q2dot = x
    vcat(sincos(q1)..., sincos(q2)..., q1dot, q2dot)
end
function unwrap(x::AbstractVector)
    sq1, cq1, sq2, cq2, q1dot, q2dot = x
    [
        atan(sq1, cq1),
        atan(sq2, cq2),
        q1dot,
        q2dot
    ]
end

function MLBasedESC.ParametricControlSystem(::ReactionWheelPendulum, policy::Function)
    return ParametricControlSystem{true}(eom!,policy,4)
end

function simulate(
    sys::ParametricControlSystem, ps::AbstractVector; 
    x0=[3.,0,0,0], tf=3.0, n=ceil(Int,1+tf/0.1)
)
    t0 = zero(tf)
    prob = ODEProblem(sys, ps, (t0, tf))
    trajectory(prob, x0, ps; saveat=range(t0,tf,length=n))
end
