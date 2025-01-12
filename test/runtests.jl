using NumericalMethodsCourse
using Test

const A1 = [3. 5. 3.; 2. 2. 3.]
const b1 = [1.; 3.]

const A5 = [3. -1. 0.; -1. 3. -1.; 0. -1. 3.]
const b5 = [2.; 1.; 2.]

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

@testset "Jacobi" begin
    @test_throws ArgumentError jacobi(A1, b1, [0.; 0.], max_iterations, tolerance, correction)
    @test check_convergence(
        jacobi(A5, b5, [0.; 0.; 0.], max_iterations, tolerance, correction)...,
        A5 \ b5, 44
    )
end

@testset "Gauss-Seidel ascending" begin
    @test_throws ArgumentError gauss_seidel(A1, b1, [0.; 0.], max_iterations, tolerance, correction)
    @test check_convergence(
        gauss_seidel(A5, b5, [0.1; 0.2; 0.3], max_iterations, tolerance, correction)...,
        A5 \ b5, 23
    )
end

@testset "Jacobi overrelaxation" begin
    @test_throws ArgumentError jacobi_overrelaxation(A1, b1, [0.; 0.], 0.5, max_iterations, tolerance, correction)
    @test check_convergence(
        jacobi_overrelaxation(A5, b5, [0.1; 0.2; 0.3], 0.8, max_iterations, tolerance, correction)...,
        A5 \ b5, 59
    )
end
