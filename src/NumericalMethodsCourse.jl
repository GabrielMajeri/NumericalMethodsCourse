module NumericalMethodsCourse

using LinearAlgebra
using Printf: @sprintf

export StoppingCriterion,
    correction,
    residual,
    jacobi,
    gauss_seidel,
    gauss_seidel_backwards,
    jacobi_overrelaxation,
    successive_overrelaxation,
    successive_overrelaxation_backwards,
    symmetric_gauss_seidel,
    symmetric_successive_overrelaxation,
    steepest_descent

include("iterative_algorithm.jl")
include("jacobi.jl")
include("gauss_seidel.jl")
include("jacobi_overrelaxation.jl")
include("successive_overrelaxation.jl")
include("symmetric_methods.jl")
include("steepest_descent.jl")

end
