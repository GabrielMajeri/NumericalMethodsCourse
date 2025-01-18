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
    if norm(x_out) < 10e3
        error("x_out did not blow up: $x_out")
    end
    if num_iters_out ≠ max_iterations
        error("num_iters_out ≠ max_iterations: $num_iters_out ≠ $max_iterations")
    end
    true
end

@testset "Jacobi" begin
    @test_throws ArgumentError jacobi(
        A1,
        b1,
        [0.0; 0.0],
        max_iterations,
        tolerance,
        correction,
    )
    @test check_divergence(
        jacobi(A2, b2, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
    )
    @test check_convergence(
        jacobi(A3, b3, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
        A3 \ b3,
        35,
    )
    @test check_divergence(
        jacobi(A4, b4, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
    )
    @test check_convergence(
        jacobi(A5, b5, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
        A5 \ b5,
        44,
    )
end

@testset "Gauss-Seidel (ascending)" begin
    @test_throws ArgumentError gauss_seidel(
        A1,
        b1,
        [0.0; 0.0],
        max_iterations,
        tolerance,
        correction,
    )
    @test check_divergence(
        gauss_seidel(A2, b2, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
    )
    @test check_convergence(
        gauss_seidel(A3, b3, [0.0; 0.0; 0.0], max_iterations, tolerance, correction)...,
        A3 \ b3,
        10,
    )
    @test check_convergence(
        gauss_seidel(A5, b5, [0.1; 0.2; 0.3], max_iterations, tolerance, correction)...,
        A5 \ b5,
        23,
    )
end

@testset "Gauss-Seidel (descending)" begin
    @test_throws ArgumentError gauss_seidel_backwards(
        A1,
        b1,
        [0.0; 0.0],
        max_iterations,
        tolerance,
        correction,
    )
    @test check_convergence(
        gauss_seidel_backwards(
            A3,
            b3,
            [0.0; 0.0; 0.0],
            max_iterations,
            tolerance,
            correction,
        )...,
        A3 \ b3,
        49,
    )
    @test check_convergence(
        gauss_seidel_backwards(
            A5,
            b5,
            [0.1; 0.2; 0.3],
            max_iterations,
            tolerance,
            correction,
        )...,
        A5 \ b5,
        65,
    )
end

@testset "Jacobi overrelaxation" begin
    @test_throws ArgumentError jacobi_overrelaxation(
        A1,
        b1,
        [0.0; 0.0],
        0.5,
        max_iterations,
        tolerance,
        correction,
    )
    @test check_convergence(
        jacobi_overrelaxation(
            A5,
            b5,
            [0.1; 0.2; 0.3],
            0.8,
            max_iterations,
            tolerance,
            correction,
        )...,
        A5 \ b5,
        59,
    )
end

@testset "Successive overrelaxation (ascending)" begin
    @test_throws ArgumentError successive_overrelaxation(
        A1,
        b1,
        [0.0; 0.0],
        0.5,
        max_iterations,
        tolerance,
        correction,
    )
    @test check_convergence(
        successive_overrelaxation(
            A5,
            b5,
            [0.1; 0.2; 0.3],
            0.8,
            max_iterations,
            tolerance,
            correction,
        )...,
        A5 \ b5,
        42,
    )
end

@testset "Successive overrelaxation (descending)" begin
    @test_throws ArgumentError successive_overrelaxation_backwards(
        A1,
        b1,
        [0.0; 0.0],
        0.5,
        max_iterations,
        tolerance,
        correction,
    )
    @test check_convergence(
        successive_overrelaxation(
            A5,
            b5,
            [0.1; 0.2; 0.3],
            0.8,
            max_iterations,
            tolerance,
            correction,
        )...,
        A5 \ b5,
        42,
    )
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
