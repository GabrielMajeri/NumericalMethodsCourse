module NumericalMethodsCourse

using LinearAlgebra

export StoppingCriterion, correction, residual, jacobi

@enum StoppingCriterion correction residual

"Validate the size of the input parameters given to an iterative method."
function check_inputs_size(A::AbstractMatrix{T}, b::AbstractVector{T},
    x⁰::AbstractVector{T}) where {T <: Number}
    m, n = size(A)
    if m != n
        error("Coefficient matrix must be square")
    end

    println(size(b))
    if size(b) != (n,)
        error("Bias vector doesn't match matrix size")
    end

    if size(x⁰) != (n,)
        error("Initial vector doesn't match matrix size")
    end
end

"Compute the inverse of a given matrix using the Jacobi iteration method."
function jacobi(
    A::AbstractMatrix{T}, b::AbstractVector{T},
    x⁰::AbstractVector{T}, max_iterations::Int,
    tolerance::Float64, criterion::StoppingCriterion) where {T <: Number}

    check_inputs_size(A, b, x⁰)

    # Extract the main diagonal of the matrix A
    D = Diagonal(diag(A))
    # Compute its inverse
    D_inverse = inv(D)

    # println("D = ", D)
    # println("D^{-1} = ", D_inverse)

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # Initialize the solution variable with the initial guess
    solution = x⁰

    # Cache the iteration matrix
    iteration_matrix = I - D_inverse * A

    # Cache the D^{-1} * b term which shows up in the iteratior
    additive_term = D_inverse * b

    # While the stopping criterion is not met
    while true
        # println(num_iterations)

        # Apply the Jacobi iteration step,
        #     x_{n + 1} = (I - D^{-1} A) * x_{n} + D^{-1} b
        new_solution = iteration_matrix * solution + additive_term

        # println(new_solution)

        # Update the error criteria
        corr = new_solution - solution
        res = b - A * new_solution

        # Compute the error vector
        error = similar(b)
        if criterion == correction
            error = corr
        elseif criterion == residual
            error = res
        else
            error("Invalid stopping criterion")
        end

        # Update the solution vector
        solution = new_solution
        # Increment the number of iterations
        num_iterations += 1

        # Check if we've converged
        # println(norm(error))
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

end
