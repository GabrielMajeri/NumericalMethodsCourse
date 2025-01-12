# Numerical Methods course - Laboratory Homework

This repository contains my solutions (written in [Julia](https://julialang.org/)) for the homework exercises of the _"Numerical methods for nonlinear systems and optimization"_ course, offered at the [FMI-UB](https://fmi.unibuc.ro/) _"Probabilities and Statistics in Finance and Sciences"_ master's program.

## Development instructions

To automatically format the code, globally install [the `JuliaFormatter` package](https://domluna.github.io/JuliaFormatter.jl/stable/) and run

```sh
julia -e 'using JuliaFormatter; format(".")'
```

To run the unit tests, use the following command:

```sh
julia --project=. test/runtests.jl
```

## License

The source code in this repo is available under the permissive [MIT license](LICENSE.txt).
