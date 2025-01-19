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

"Abstract type used for defining new iterative algorithms for solving linear systems."
abstract type IterativeAlgorithm end

"""Solves the given linear system, using `x⁰` as a starting guess for the solution,
running for at most `max_iterations` steps and stopping when the error metric indicated by `criterion`
has an euclidean norm below `tolerance`.
"""
function solve_linear_system(
    algorithm::Algorithm,
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
    parameters,
) where {Algorithm<:IterativeAlgorithm,T<:Number}
    # Initialize the solution variable with the initial guess
    solution = x⁰

    # Create a buffer vector for the new solution vector being constructed
    new_solution = similar(x⁰)

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # While the stopping criterion is not yet met
    while true
        @debug "Iteration #$num_iterations"

        # Perform a step of the iterative algorithm
        update_solution!(algorithm, new_solution, solution, A, b, parameters)

        # Update the error metrics
        corr = new_solution - solution
        corr_norm = norm(corr)
        @debug "Norm of correction/increment: $(@sprintf "%.4f" corr_norm)"

        res = b - A * new_solution
        res_norm = norm(res)
        @debug "Norm of residual error: $(@sprintf "%.4f" res_norm)"

        # Update the solution vector (by making a swap to conserve memory)
        solution, new_solution = new_solution, solution
        # Increment the number of iterations
        num_iterations += 1

        # Determine the value of the stopping criterion
        error_norm = 0.0
        if criterion == correction
            error_norm = corr_norm
        elseif criterion == residual
            error_norm = err_norm
        else
            error("Invalid stopping criterion")
        end

        # Check if we're converged
        if error_norm <= tolerance
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
