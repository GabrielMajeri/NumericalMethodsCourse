"Validate that a given input matrix is symmetric, otherwise throws an error."
function check_input_symmetric(A::AbstractMatrix)
    if !issymmetric(A)
        throw(ArgumentError("Coefficient matrix must be symmetric"))
    end
end

struct SymmetricGaussSeidelAlgorithm <: IterativeAlgorithm end

@kwdef struct SymmetricGaussSeidelAlgorithmParameters{T<:Number}
    intermediate_solution::AbstractVector{T}
end

function update_solution!(
    ::SymmetricGaussSeidelAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::SymmetricGaussSeidelAlgorithmParameters{T},
) where {T<:Number}
    n = length(b)
    intermediate_solution = parameters.intermediate_solution

    # Compute the new solution vector, component-by-component
    for i = 1:n
        δ₁ = 0
        for j = 1:i-1
            δ₁ += A[i, j] * intermediate_solution[j]
        end

        δ₂ = 0
        for j = i+1:n
            δ₂ += A[i, j] * solution[j]
        end

        intermediate_solution[i] = (1 / A[i, i]) * (b[i] - δ₁ - δ₂)

        δ₁ = 0
        for j = 1:i-1
            δ₁ += A[i, j] * intermediate_solution[j]
        end

        δ₂ = 0
        for j = i+1:n
            δ₂ += A[i, j] * new_solution[j]
        end

        new_solution[i] = (1 / A[i, i]) * (b[i] - δ₁ - δ₂)
    end
end

"Computes the solution of a symmetric linear system using the (ascending) Symmetric Gauss-Seidel method."
function symmetric_gauss_seidel(
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    x⁰::AbstractVector{T},
    max_iterations::Int,
    tolerance::Float64,
    criterion::StoppingCriterion,
) where {T<:Number}
    # Check that the input arguments are valid
    check_inputs_size(A, b, x⁰)
    check_input_symmetric(A)

    # Buffer for holding the intermediate solution vector
    intermediate_solution = similar(x⁰)
    parameters = SymmetricGaussSeidelAlgorithmParameters(intermediate_solution)

    solve_linear_system(
        SymmetricGaussSeidelAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        parameters,
    )
end

struct SymmetricSuccessiveOverrelaxationAlgorithm <: IterativeAlgorithm end

@kwdef struct SymmetricSuccessiveOverrelaxationAlgorithmParameters{T<:Number}
    intermediate_solution::AbstractVector{T}
    ω::Float64
end

function update_solution!(
    ::SymmetricSuccessiveOverrelaxationAlgorithm,
    new_solution::AbstractVector{T},
    solution::AbstractVector{T},
    A::AbstractMatrix{T},
    b::AbstractVector{T},
    parameters::SymmetricSuccessiveOverrelaxationAlgorithmParameters{T},
) where {T<:Number}
    n = length(b)
    intermediate_solution = parameters.intermediate_solution
    ω = parameters.ω

    # Compute the new solution vector, component-by-component
    for i = 1:n
        δ₁ = 0
        for j = 1:i-1
            δ₁ += A[i, j] * intermediate_solution[j]
        end

        δ₂ = 0
        for j = i+1:n
            δ₂ += A[i, j] * solution[j]
        end

        intermediate_solution[i] = (ω / A[i, i]) * (b[i] - δ₁ - δ₂) + (1 - ω) * solution[i]

        δ₁ = 0
        for j = 1:i-1
            δ₁ += A[i, j] * intermediate_solution[j]
        end

        δ₂ = 0
        for j = i+1:n
            δ₂ += A[i, j] * new_solution[j]
        end

        new_solution[i] =
            (ω / A[i, i]) * (b[i] - δ₁ - δ₂) + (1 - ω) * intermediate_solution[i]
    end
end

"Computes the solution of a symmetric linear system using the symmetric
Successive Overrelaxation (SOR) method."
function symmetric_successive_overrelaxation(
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
    check_input_symmetric(A)

    # Buffer for holding the intermediate solution vector
    intermediate_solution = similar(x⁰)
    parameters =
        SymmetricSuccessiveOverrelaxationAlgorithmParameters(intermediate_solution, ω)

    solve_linear_system(
        SymmetricSuccessiveOverrelaxationAlgorithm(),
        A,
        b,
        x⁰,
        max_iterations,
        tolerance,
        criterion,
        parameters,
    )
end
