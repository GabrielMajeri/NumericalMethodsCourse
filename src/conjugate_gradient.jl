"Computes the solution of a symmetric and positive definite linear system
using the conjugate gradient algorithm."
function conjugate_gradient(
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

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # Compute initial residue vector
    res = b - A * solution

    # Initial descent direction
    p = res

    rho_old = 0
    rho_new = dot(res, res)
    rho_0 = rho_new

    # We consider that we've reached convergence when
    #   rho_new <= tolerance * rho_0
    rho_tolerance = tolerance * rho_0

    # While the stopping criterion is not yet met
    while true
        @debug "Iteration #$num_iterations"

        if num_iterations > 0
            # Update the descent direction
            beta = rho_new / rho_old
            p = res + beta * p
        end

        # Next element of Krylov basis
        w = A * p

        # Compute the optimal step size
        rate = rho_new / dot(p, w)

        # Perform a step of the conjugate gradient algorithm
        solution = solution + rate * p

        # Update the residual
        res = res - rate * w

        # Update the ρ variables
        rho_old = rho_new
        rho_new = dot(res, res)

        @debug "Square of norm of residual error: $(@sprintf "%.4f" rho_new)"

        # Increment the number of iterations
        num_iterations += 1

        # Stop if we've reached the maximum allowed number of iterations
        if num_iterations >= max_iterations
            break
        end

        # Stop if we're below the desired error tolerance
        if rho_new < rho_tolerance
            break
        end
    end

    solution, num_iterations
end
