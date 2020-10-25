// pi_01_seq_loop.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_01_seq_loop.cxx -o pi_01.exe

// Usage:
//    ./pi_01.exe


#include <iostream>
#include <omp.h>


double pi_01_seq_loop(int num_steps)
{
  double step_size = 1.0 / num_steps;
  double sum       = 0.0;

  for (int i = 0; i < num_steps; ++i) {
    double x = (i + 0.5) * step_size;
    sum      = sum + 4.0 / (1.0 + x * x);
  }

  return sum * step_size;
}


int main()
{
  int num_steps = 1024*1024*1024;

  double start_time = omp_get_wtime();
  double pi         = pi_01_seq_loop(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << 1
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
