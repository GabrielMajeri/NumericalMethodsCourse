"Computes the solution of a linear system using the Jacobi iteration method."
function jacobi(
    A::AbstractMatrix{T}, b::AbstractVector{T},
    x⁰::AbstractVector{T}, max_iterations::Int,
    tolerance::Float64, criterion::StoppingCriterion) where {T <: Number}

    check_inputs_size(A, b, x⁰)

    # Extract the main diagonal of the matrix A
    D = Diagonal(diag(A))
    # Compute its inverse
    D_inverse = inv(D)

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # Initialize the solution variable with the initial guess
    solution = x⁰

    # Cache the iteration matrix
    iteration_matrix = I - D_inverse * A

    # Cache the D^{-1} * b term which shows up in the iteration
    additive_term = D_inverse * b

    # While the stopping criterion is not met
    while true
        # Apply the Jacobi iteration step,
        #     x_{n + 1} = (I - D^{-1} A) * x_{n} + D^{-1} b
        new_solution = iteration_matrix * solution + additive_term

        # Update the error criteria
        corr = new_solution - solution
        res = b - A * new_solution

        # Update the solution vector
        solution = new_solution
        # Increment the number of iterations
        num_iterations += 1

        # Compute the error vector
        error = similar(b)
        if criterion == correction
            error = corr
        elseif criterion == residual
            error = res
        else
            error("Invalid stopping criterion")
        end

        # Check if we've converged
        if norm(error) <= tolerance
            break
        end

        # Stop if we've reached the maximum number of iterations
        # we've been permitted to use
        if num_iterations >= max_iterations
            break
        end
    end

    # Return the solution the algorithm converged to and the number of iterations it took to reach it
    solution, num_iterations
end
