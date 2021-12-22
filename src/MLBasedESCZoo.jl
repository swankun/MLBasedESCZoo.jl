module MLBasedESCZoo

include("iwp/iwp.jl")
using .ReactionWheelPendulumModule
export ReactionWheelPendulum, wrap, unwrap, simulate
export train!, policyfrom, loadidapbc, NeuralIDAPBC, DefaultIDAPBC
export BayesianIDAPBC, mappolicy, margpolicy
export trainloss, train!, evaluate, plot

end # module
