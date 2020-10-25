// pi_07_par_omp_for_reduction.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_07_par_omp_for_reduction.cxx -o pi_07.exe

// Usage:
//    ./pi_07.exe


#include <iostream>
#include <omp.h>


double pi_07_par_omp_for_reduction(int num_steps)
{
  double step_size = 1.0 / num_steps;
  double sum       = 0.0;

  #pragma omp parallel for              \
    default      (none)                 \
    firstprivate (num_steps, step_size) \
    reduction    (+:sum)
    for (int i = 0; i < num_steps; ++i) {
      double x = (i + 0.5) * step_size;
      sum      = sum + 4.0 / (1.0 + x * x);
  }

  return sum * step_size;
}


int main()
{
  int num_steps = 1024*1024*1024;
  int max_thrds = omp_get_max_threads();

  double start_time = omp_get_wtime();
  double pi         = pi_07_par_omp_for_reduction(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << max_thrds
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
