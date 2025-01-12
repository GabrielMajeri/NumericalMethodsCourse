"Computes the solution of a linear system using the Successive Overrelaxation (SOR) method."
function successive_overrelaxation(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    ω::Float64,
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
) where {T<:Number}
    check_inputs_size(A, b, x⁰)

    n = length(b)

    # Initialize the solution variable with the initial guess
    solution = x⁰

    # Create a buffer vector for the new solution vector being constructed
    new_solution = similar(x⁰)

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # While the stopping criterion is not yet met
    while true
        # Compute the new solution vector, component-by-component
        for i = 1:n
            δ₁ = 0
            for j = 1:i-1
                δ₁ += A[i, j] * new_solution[j]
            end

            δ₂ = 0
            for j = i+1:n
                δ₂ += A[i, j] * solution[j]
            end

            new_solution[i] = (ω / A[i, i]) * (b[i] - δ₁ - δ₂) + (1 - ω) * solution[i]
        end

        # Update the error criteria
        corr = new_solution - solution
        res = b - A * new_solution

        # Update the solution vector (by making a swap to conserve memory)
        solution, new_solution = new_solution, solution
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

        # Check if we're converged
        if norm(error) <= tolerance
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

"Computes the solution of a linear system using the descending Successive Overrelaxation (SOR) method."
function successive_overrelaxation_backwards(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    ω::Float64,
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
) where {T<:Number}
    check_inputs_size(A, b, x⁰)

    n = length(b)

    # Initialize the solution variable with the initial guess
    solution = x⁰

    # Create a buffer vector for the new solution vector being constructed
    new_solution = similar(x⁰)

    # Create a counter variable for the number of iterations
    # required for the algorithm to converge
    num_iterations = 0

    # While the stopping criterion is not yet met
    while true
        # Compute the new solution vector, component-by-component
        for i = 1:n
            δ₁ = 0
            for j = 1:i-1
                δ₁ += A[i, j] * solution[j]
            end

            δ₂ = 0
            for j = i+1:n
                δ₂ += A[i, j] * new_solution[j]
            end

            new_solution[i] = (ω / A[i, i]) * (b[i] - δ₁ - δ₂) + (1 - ω) * solution[i]
        end

        # Update the error criteria
        corr = new_solution - solution
        res = b - A * new_solution

        # Update the solution vector (by making a swap to conserve memory)
        solution, new_solution = new_solution, solution
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

        # Check if we're converged
        if norm(error) <= tolerance
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
