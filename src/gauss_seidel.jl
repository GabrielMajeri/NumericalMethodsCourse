struct AscendingGaussSeidelAlgorithm <: IterativeAlgorithm end

function update_solution!(
    ::AscendingGaussSeidelAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::Nothing,
) where {T<:Number}
    n = length(b)

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

        new_solution[i] = (1 / A[i, i]) * (b[i] - δ₁ - δ₂)
    end
end

"Computes the solution of a linear system using the ascending Gauss-Seidel method."
function gauss_seidel(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
) where {T<:Number}
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)

    solve_linear_system(
        AscendingGaussSeidelAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        nothing,
    )
end

struct DescendingGaussSeidelAlgorithm <: IterativeAlgorithm end

function update_solution!(
    ::DescendingGaussSeidelAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::Nothing,
) where {T<:Number}
    n = length(b)

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

        new_solution[i] = (1 / A[i, i]) * (b[i] - δ₁ - δ₂)
    end
end

"Computes the solution of a linear system using the descending Gauss-Seidel method."
function gauss_seidel_backwards(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
) where {T<:Number}
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)

    solve_linear_system(
        DescendingGaussSeidelAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        nothing,
    )
end
