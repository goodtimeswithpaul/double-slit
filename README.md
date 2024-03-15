To explore the results of this project, the following files/folders are important.

1. finitedifference.py is the original code to be optimized. Running it will show a simulation of the double slit experiment.
2. the folder profiling_results shows the results of our various profiling methods (memory profiling, line profiling and cProfiling)
3. the notebook numexpr optimization.ipynb implements and explains the code optimization utilising numexpr, as well as unit tests to ensure that the behaviour of the program isn't affected. The unit tests for the cupy optimization are shown in test_cupy_optimization.py
4. the file finitedifference_cupy.py presents the code optimization utilising cupy
5. the notebook combined_optimization.ipynb shows both optimizations, and measures and plots the performance gains (in terms of program execution time) of both optimizations, compared with that of the original code.
6. The original code is documented (HTML documentation using Sphinx) in the folder original_code_documentation
7. The optimized code (using cupy) is documented in the folder cupy_optimization_documentation
8. The optimized code (using numexpr) is documented in the folder numexpr_optimization_documentation

