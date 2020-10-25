// pi_03_par_spmd.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_03_par_spmd.cxx -o pi_03.exe

// Usage:
//    ./pi_03.exe


#include <iostream>
#include <tuple>
#include <vector>
#include <omp.h>


std::tuple<double, int> pi_03_par_spmd(int num_steps)
{
  double step_size = 1.0 / num_steps;
  int    max_thrds = omp_get_max_threads();
  int    chk_thrds = 0;

  std::vector<double> temp(max_thrds, 0.0);

  #pragma omp parallel             \
    default      (none)            \
    shared       (chk_thrds, temp) \
    firstprivate (num_steps, step_size)
  {
    int id        = omp_get_thread_num();
    int num_thrds = omp_get_num_threads();

    if (id == 0)
      chk_thrds = num_thrds;

    for (int i = id; i < num_steps; i += num_thrds) {
      double x = (i + 0.5) * step_size;
      temp[id] = temp[id] + 4.0 / (1.0 + x * x);
    }
  } // end omp parallel

  double sum = 0.0;

  for (int i = 0; i < chk_thrds; ++i)
    sum = sum + temp[i];

  return { sum * step_size, chk_thrds };
}


int main()
{
  int num_steps = 1024*1024*1024;

  double  start_time      = omp_get_wtime();
  auto    [pi, num_thrds] = pi_03_par_spmd(num_steps);
  double  stop_time       = omp_get_wtime();

  std::cout << "\nthreads:  " << num_thrds
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
