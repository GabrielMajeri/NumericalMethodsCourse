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
    symmetric_successive_overrelaxation

@enum StoppingCriterion correction residual

"Validate the size of the input parameters given to an iterative method."
function check_inputs_size(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
) where {T<:Number}
    # Extract the matrix dimensions
    m, n = size(A)

    if m != n
        throw(ArgumentError("Coefficient matrix must be square"))
    end

    if size(b) != (n,)
        throw(DimensionMismatch("Bias vector doesn't match matrix size"))
    end

    if size(x⁰) != (n,)
        throw(DimensionMismatch("Initial vector doesn't match matrix size"))
    end
end

include("jacobi.jl")
include("gauss_seidel.jl")
include("jacobi_overrelaxation.jl")
include("successive_overrelaxation.jl")
include("symmetric_methods.jl")

end
