"Enumeration of the criteria which can be used to determine the convergence of an iterative algorithm."
@enum StoppingCriterion correction residual

"Validate the size of the input parameters given to an iterative algorithm."
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

"Validate that a given input matrix is symmetric, otherwise throws an error."
function check_input_symmetric(A::AbstractMatrix)
    if !issymmetric(A)
        throw(ArgumentError("Coefficient matrix must be symmetric"))
    end
end
