// pi_10_vec_unroll_8.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_10_vec_unroll_8.cxx -o pi_10.exe

// Usage:
//    ./pi_10.exe


#include <iostream>
#include <omp.h>


double pi_10_vec_unroll_8(int num_steps)
{
  double step_size = 1.0 / num_steps;
  double sum1      = 0.0;
  double sum2      = 0.0;

  for (int i = 0; i < num_steps; i += 8) {
    double x0 = (i + 0.5) * step_size;
    double x1 = (i + 1.5) * step_size;
    double x2 = (i + 2.5) * step_size;
    double x3 = (i + 3.5) * step_size;
    double x4 = (i + 4.5) * step_size;
    double x5 = (i + 5.5) * step_size;
    double x6 = (i + 6.5) * step_size;
    double x7 = (i + 7.5) * step_size;

    sum1 = sum1 + 4.0 / (1.0 + x0 * x0)
                + 4.0 / (1.0 + x1 * x1)
                + 4.0 / (1.0 + x2 * x2)
                + 4.0 / (1.0 + x3 * x3);

    sum2 = sum2 + 4.0 / (1.0 + x4 * x4)
                + 4.0 / (1.0 + x5 * x5)
                + 4.0 / (1.0 + x6 * x6)
                + 4.0 / (1.0 + x7 * x7);
  }

  return (sum1 + sum2) * step_size;
}


int main()
{
  int num_steps = 1024*1024*1024;

  double start_time = omp_get_wtime();
  double pi         = pi_10_vec_unroll_8(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << 1
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
