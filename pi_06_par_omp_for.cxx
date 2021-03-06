// pi_06_par_omp_for.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_06_par_omp_for.cxx -o pi_06.exe

// Usage:
//    ./pi_06.exe


#include <iostream>
#include <tuple>
#include <omp.h>


std::tuple<double, int> pi_06_par_omp_for(int num_steps)
{
  double step_size = 1.0 / num_steps;
  int    chk_thrds = 0;
  double sum       = 0.0;

  #pragma omp parallel             \
    default      (none)            \
    shared       (chk_thrds, sum)  \
    firstprivate (num_steps, step_size)
  {
    int id        = omp_get_thread_num();
    int num_thrds = omp_get_num_threads();

    if (id == 0)
      chk_thrds = num_thrds;

    double temp = 0.0;

    #pragma omp for
      for (int i = 0; i < num_steps; ++i) {
        double x = (i + 0.5) * step_size;
        temp     = temp + 4.0 / (1.0 + x * x);
      }

  #pragma omp critical
    sum = sum + temp;
  } // end omp parallel

  return { sum * step_size, chk_thrds };
}


int main()
{
  int num_steps = 1024*1024*1024;

  double start_time      = omp_get_wtime();
  auto   [pi, num_thrds] = pi_06_par_omp_for(num_steps);
  double stop_time       = omp_get_wtime();

  std::cout << "\nthreads:  " << num_thrds
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
