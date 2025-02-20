"Computes the solution of a symmetric and positive definite linear system
using the steepest descent algorithm."
function steepest_descent(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    max_iterations::Int,
    tolerance::Float64,
) where {T<:Number}
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)

    # Check that the system matrix is symmetric
    check_input_symmetric(A)

    # We should also ensure that it's positive definite,
    # but that check would be very costly

    # Initialize the solution variable with the initial guess
    solution = x⁰

    # Create a buffer vector for the new solution vector being constructed
    new_solution = similar(x⁰)

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # Compute initial residue vector
    res = b - A * solution
    # The term "A * res" shows up in several places,
    # so we compute it beforehand and cache it
    A_times_res = A * res

    # While the stopping criterion is not yet met
    while true
        @debug "Iteration #$num_iterations"

        # Compute the optimal descent step size
        rate = dot(res, res) / dot(res, A_times_res)

        # Perform a step of the steepest descent algorithm
        new_solution = solution + rate * res

        # Update the residual
        res = res - rate * A_times_res
        A_times_res = A * res
        res_norm = norm(res)
        @debug "Norm of residual error: $(@sprintf "%.4f" res_norm)"

        # Update the solution vector (by making a swap to conserve memory)
        solution, new_solution = new_solution, solution
        # Increment the number of iterations
        num_iterations += 1

        # Check if we're converged
        if res_norm <= tolerance
            break
        end

        # Stop if we've reached the maximum number of iterations
        # we've been permitted to use
        if num_iterations >= max_iterations
            break
        end
    end

    solution, num_iterations
end
