@kwdef struct SuccessiveOverrelaxationAlgorithmParameters
    ω::Float64
end

struct AscendingSuccessiveOverrelaxationAlgorithm <: IterativeAlgorithm end

function update_solution!(
    ::AscendingSuccessiveOverrelaxationAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::SuccessiveOverrelaxationAlgorithmParameters,
) where {T<:Number}
    n = length(b)
    ω = parameters.ω

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
end

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
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)

    parameters = SuccessiveOverrelaxationAlgorithmParameters(ω)

    solve_linear_system(
        AscendingSuccessiveOverrelaxationAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        parameters,
    )
end

struct DescendingSuccessiveOverrelaxationAlgorithm <: IterativeAlgorithm end

function update_solution!(
    ::DescendingSuccessiveOverrelaxationAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::SuccessiveOverrelaxationAlgorithmParameters,
) where {T<:Number}
    n = length(b)
    ω = parameters.ω

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
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)

    parameters = SuccessiveOverrelaxationAlgorithmParameters(ω)

    solve_linear_system(
        DescendingSuccessiveOverrelaxationAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        parameters,
    )
end
