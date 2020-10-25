// pi_02_seq_recursive.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -O3 pi_02_seq_recursive.cxx -o pi_02.exe

// Usage:
//    ./pi_02.exe


#include <iostream>
#include <omp.h>


double pi_02_impl(int iter_begin, int iter_end, double step_size)
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

      temp1 = pi_02_impl(iter_begin, iter_end - block / 2, step_size);
      temp2 = pi_02_impl(iter_end - block / 2,   iter_end, step_size);

      sum = temp1 + temp2;
    }

  return sum;
}


double pi_02_seq_recursive(int num_steps)
{
  double step_size = 1.0 / num_steps;
  return pi_02_impl(0, num_steps, step_size) * step_size;
}


int main ()
{
  int num_steps = 1024*1024*1024;

  double start_time = omp_get_wtime();
  double pi         = pi_02_seq_recursive(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << 1
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
