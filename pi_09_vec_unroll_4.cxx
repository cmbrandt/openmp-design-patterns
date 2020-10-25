// pi_09_vec_unroll_4.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_09_vec_unroll_4.cxx -o pi_09.exe

// Usage:
//    ./pi_09.exe


#include <iostream>
#include <omp.h>


double pi_09_vec_unroll_4(int num_steps)
{
  double step_size = 1.0 / num_steps;
  double sum       = 0.0;

  for (int i = 0; i < num_steps; i += 4) {
    double x0 = (i + 0.5) * step_size;
    double x1 = (i + 1.5) * step_size;
    double x2 = (i + 2.5) * step_size;
    double x3 = (i + 3.5) * step_size;

    sum = sum + 4.0 / (1.0 + x0 * x0)
              + 4.0 / (1.0 + x1 * x1)
              + 4.0 / (1.0 + x2 * x2)
              + 4.0 / (1.0 + x3 * x3);
  }

  return sum * step_size;
}


int main()
{
  int num_steps = 1024*1024*1024;

  double start_time = omp_get_wtime();
  double pi         = pi_09_vec_unroll_4(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << 1
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
