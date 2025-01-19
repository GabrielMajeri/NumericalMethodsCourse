# Numerical Methods course - Laboratory Homework

This repository contains my solutions (written in [Julia](https://julialang.org/)) for the homework exercises of the _"Numerical methods for nonlinear systems and optimization"_ course, offered at the [FMI-UB](https://fmi.unibuc.ro/) _"Probabilities and Statistics in Finance and Sciences"_ master's program.

Implemented methods:

- [Jacobi](https://en.wikipedia.org/wiki/Jacobi_method)
- [Gauss-Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) (ascending, descending and symmetric)
- [Relaxation-based methods](https://sabs-r3.github.io/scientific-computing/unit_2_linear_algebra/06-jacobi-relaxation-methods/):
    - Jacobi Overrelaxation
    - Successive Overrelaxation (ascending, descending and symmetric)

## Development instructions

To automatically format the code, globally install [the `JuliaFormatter` package](https://domluna.github.io/JuliaFormatter.jl/stable/) and run

```sh
julia -e 'using JuliaFormatter; format(".")'
```

To run the unit tests, use the following command:

```sh
julia --project=. test/runtests.jl
```

To enable debug logging for this package (which will lead to the printing of useful information at each iteration of the algorithms), set the `JULIA_DEBUG` variable:

```sh
export JULIA_DEBUG=NumericalMethodsCourse
```

and then run the code as usual.

## License

The source code in this repo is available under the permissive [MIT license](LICENSE.txt).
