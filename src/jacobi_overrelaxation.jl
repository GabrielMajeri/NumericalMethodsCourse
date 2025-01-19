"Computes the solution of a linear system using the Jacobi Overrelaxation method."
function jacobi_overrelaxation(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    ω::Float64,
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
    iteration_matrix = I - ω * D_inverse * A

    # Cache the ω * D^{-1} * b term which shows up in the iteration
    additive_term = ω * D_inverse * b

    parameters = JacobiAlgorithmParameters(iteration_matrix, additive_term)

    # We will use the Jacobi Overrelaxation iteration step,
    #     x_{n + 1} = (I - ω D^{-1} A) * x_{n} + ω D^{-1} b
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
