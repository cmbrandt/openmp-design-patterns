// pi_08_par_omp_task.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_08_par_omp_task.cxx -o pi_08.exe

// Usage:
//    ./pi_08.exe


#include <iostream>
#include <omp.h>


double pi_08_impl(int iter_begin, int iter_end, double step_size)
{
  const int block_size = 1024*1024*512;

  double sum   = 0.0;
  double temp1 = 0.0;
  double temp2 = 0.0;

    if (iter_end - iter_begin < block_size) {

      for (int i = iter_begin; i < iter_end; ++i) {
        double x = (i + 0.5) * step_size;
        sum      = sum + 4.0 / (1.0 + x * x); 
      }
    }
    else {

      int block = iter_end - iter_begin;

      #pragma omp task  \
        default (none)  \
        shared  (temp1) \
        firstprivate (block, iter_begin, iter_end, step_size)
        temp1 = pi_08_impl(iter_begin, iter_end - block / 2, step_size);

      #pragma omp task  \
        default (none)  \
        shared  (temp2) \
        firstprivate (block, iter_begin, iter_end, step_size)
        temp2 = pi_08_impl(iter_end - block / 2,   iter_end, step_size);

      #pragma omp taskwait
        sum = temp1 + temp2;
    }

  return sum;
}


double pi_08_par_omp_task(int num_steps)
{
  double step_size = 1.0 / num_steps;
  double sum       = 0.0;

  #pragma omp parallel \
    default (none)     \
    shared  (sum)      \
    firstprivate (num_steps, step_size)
  {
    #pragma omp single
      sum = pi_08_impl(0, num_steps, step_size);
  }
  
  return sum * step_size;
}


int main ()
{
  int num_steps = 1024*1024*1024;
  int max_thrds = omp_get_max_threads();

  double start_time = omp_get_wtime();
  double pi         = pi_08_par_omp_task(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << max_thrds
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
