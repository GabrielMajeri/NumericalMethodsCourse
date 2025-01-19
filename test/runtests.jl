using NumericalMethodsCourse
using LinearAlgebra
using Test

const A1 = [3.0 5.0 3.0; 2.0 2.0 3.0]
const b1 = [1.0; 3.0]

const A2 = [3.0 3.0 -6.0; -4.0 7.0 -8.0; 5.0 7.0 -9.0]
const b2 = [0.0; -5.0; 3.0]

const A3 = [4.0 1.0 1.0; 2.0 -9.0 0.0; 0.0 -6.0 -8.0]
const b3 = [6.0; -7.0; -14.0]

const A4 = [3.0 0.0 4.0; 7.0 4.0 2.0; -1.0 1.0 2.0]
const b4 = [7.0; 13.0; 2.0]

const A5 = Symmetric([3.0 -1.0 0.0; -1.0 3.0 -1.0; 0.0 -1.0 3.0])
const b5 = [2.0; 1.0; 2.0]

const max_iterations = 1_00
const tolerance = 1e-14

"Checks that the output of an iterative algorithm indicates convergence."
function check_convergence(x_out, num_iters_out, x_expected, num_iters_expected)
    if x_out ≉ x_expected
        error("x_out ≉ x_expected: $x_out ≉ $x_expected")
    end
    if num_iters_out ≠ num_iters_expected
        error("num_iters_out ≠ num_iters_expected: $num_iters_out ≠ $num_iters_expected")
    end
    true
end

function check_divergence(x_out, num_iters_out)
    if norm(x_out) < 1000 && num_iters_out ≠ max_iterations
        error("algorithm did not diverge: ($x_out, $num_iters_out)")
    end

    true
end

algorithms = Dict(
    "Jacobi" => (method = jacobi, a3_iterations = 35, a5_iterations = 44),
    "Gauss-Seidel (ascending)" =>
        (method = gauss_seidel, a3_iterations = 10, a5_iterations = 23),
    "Gauss-Seidel (descending)" =>
        (method = gauss_seidel_backwards, a3_iterations = 49, a5_iterations = 65),
)

for (name, (method, a3_iterations, a5_iterations)) ∈ algorithms
    @testset "$name" begin
        @test_throws ArgumentError method(
            A1,
            b1,
            [0.0; 0.0],
            max_iterations,
            tolerance,
            correction,
        )
        @test check_divergence(
            method(A2, b2, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
        )
        @test check_convergence(
            method(A3, b3, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
            A3 \ b3,
            a3_iterations,
        )
        @test check_divergence(
            method(A4, b4, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
        )
        @test check_convergence(
            method(A5, b5, [0.1; 0.2; 0.3], max_iterations, tolerance, correction)...,
            A5 \ b5,
            a5_iterations,
        )
    end
end

relaxation_algorithms = Dict(
    "Jacobi overrelaxation" => (
        method = jacobi_overrelaxation,
        ω = 0.8,
        a3_iterations = 39,
        a5_iterations = 59,
    ),
    "Successive overrelaxation (ascending)" => (
        method = successive_overrelaxation,
        ω = 0.9,
        a3_iterations = 20,
        a5_iterations = 33,
    ),
    "Successive overrelaxation (descending)" => (
        method = successive_overrelaxation_backwards,
        ω = 0.8,
        a3_iterations = 43,
        a5_iterations = 78,
    ),
)

for (name, (method, ω, a3_iterations, a5_iterations)) ∈ relaxation_algorithms
    @testset "$name" begin
        @test_throws ArgumentError method(
            A1,
            b1,
            [0.0; 0.0],
            ω,
            max_iterations,
            tolerance,
            correction,
        )
        @test check_divergence(
            method(A2, b2, [0.0; 0.0; 0.0], ω, max_iterations, tolerance, correction)...,
        )
        @test check_convergence(
            method(A3, b3, [0.0; 0.0; 0.0], ω, max_iterations, tolerance, correction)...,
            A3 \ b3,
            a3_iterations,
        )
        @test check_divergence(
            method(A4, b4, [1.0; 2.0; 3.0], ω, max_iterations, tolerance, correction)...,
        )
        @test check_convergence(
            method(A5, b5, [0.1; 0.2; 0.3], ω, max_iterations, tolerance, correction)...,
            A5 \ b5,
            a5_iterations,
        )
    end
end

@testset "Symmetric Gauss-Seidel" begin
    @test_throws ArgumentError symmetric_gauss_seidel(
        A1,
        b1,
        [0.0; 0.0],
        max_iterations,
        tolerance,
        correction,
    )
    @test_throws ArgumentError symmetric_gauss_seidel(
        [1.0 0.0; 1.0 0.0],
        [1.0; 1.0],
        [0.0; 0.0],
        max_iterations,
        tolerance,
        correction,
    )
    @test check_convergence(
        symmetric_gauss_seidel(
            A5,
            b5,
            [0.1; 0.2; 0.3],
            max_iterations,
            tolerance,
            correction,
        )...,
        A5 \ b5,
        31,
    )
end

@testset "Symmetric successive overrelaxation" begin
    @test_throws ArgumentError symmetric_successive_overrelaxation(
        A1,
        b1,
        [0.0; 0.0],
        0.5,
        max_iterations,
        tolerance,
        correction,
    )
    @test_throws ArgumentError symmetric_successive_overrelaxation(
        [1.0 0.0; 1.0 0.0],
        [1.0; 1.0],
        [0.0; 0.0],
        0.5,
        max_iterations,
        tolerance,
        correction,
    )
    @test check_convergence(
        symmetric_successive_overrelaxation(
            A5,
            b5,
            [0.1; 0.2; 0.3],
            0.9,
            max_iterations,
            tolerance,
            correction,
        )...,
        A5 \ b5,
        37,
    )
end
