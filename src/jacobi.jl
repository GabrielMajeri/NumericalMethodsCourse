struct JacobiAlgorithm <: IterativeAlgorithm end

@kwdef struct JacobiAlgorithmParameters{T<:Number}
    iteration_matrix::AbstractMatrix{T}
    additive_term::AbstractVector{T}
end

function update_solution!(
    ::JacobiAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::JacobiAlgorithmParameters{T},
) where {T<:Number}
    # Apply the Jacobi iteration step,
    #     x_{n + 1} = (I - D^{-1} A) * x_{n} + D^{-1} b
    new_solution .= parameters.iteration_matrix * solution + parameters.additive_term
end

"Computes the solution of a linear system using the Jacobi iteration method."
function jacobi(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
) where {T<:Number}
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)

    # Extract the main diagonal of the matrix A
    D = Diagonal(diag(A))
    # Compute its inverse
    D_inverse = inv(D)

    # Cache the iteration matrix
    iteration_matrix = I - D_inverse * A

    # Cache the D^{-1} * b term which shows up in the iteration
    additive_term = D_inverse * b

    parameters = JacobiAlgorithmParameters(iteration_matrix, additive_term)

    solve_linear_system(
        JacobiAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        parameters,
    )
end
