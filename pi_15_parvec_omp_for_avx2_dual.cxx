// pi_15_parvec_omp_for_avx2_dual.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -mavx2 -mfma -O3 pi_15_parvec_omp_for_avx2_dual.cxx -o pi_15.exe

// Usage:
//    ./pi_15.exe


#include <iostream>
#include <vector>
#include <immintrin.h>
#include <omp.h>


double pi_15_parvec_omp_for_avx2_dual_accum(int num_steps)
{
  int num_thrds = omp_get_max_threads();

  std::vector<double> sum(num_thrds, 0.0);

  #pragma omp parallel \
    default (none)     \
    shared  (sum)      \
    firstprivate (num_steps, num_thrds)
  {
    int id = omp_get_thread_num();

    double step_size = 1.0 / num_steps;

    double one  = 1.0;
    double four = 4.0;

    __m256d vcoef = _mm256_set_pd(0.5, 1.5, 2.5, 3.5);
    __m256d vone  = _mm256_broadcast_sd(&one);
    __m256d vfour = _mm256_broadcast_sd(&four);
    __m256d vstep = _mm256_broadcast_sd(&step_size);
    __m256d vsum1 = _mm256_setzero_pd();
    __m256d vsum2 = _mm256_setzero_pd();

    #pragma omp for
      for (int i = 0; i < num_steps; i += 8) {
        double  id1   = static_cast<double>(i);
        double  id2   = static_cast<double>(i+4);
        __m256d vid1  = _mm256_broadcast_sd(&id1);
        __m256d vid2  = _mm256_broadcast_sd(&id2);
        __m256d vx    = _mm256_mul_pd(_mm256_add_pd(vid1, vcoef), vstep);
        __m256d vden  = _mm256_fmadd_pd(vx, vx, vone);
                vsum1 = _mm256_add_pd(_mm256_div_pd(vfour, vden), vsum1);
                vx    = _mm256_mul_pd(_mm256_add_pd(vid2, vcoef), vstep);
                vden  = _mm256_fmadd_pd(vx, vx, vone);
                vsum2 = _mm256_add_pd(_mm256_div_pd(vfour, vden), vsum2);
    }

    double temp1[4];
    double temp2[4];
    _mm256_store_pd(&temp1[0], vsum1);
    _mm256_store_pd(&temp2[0], vsum2);

    sum[id] = ( temp1[0] + temp1[1] + temp1[2] + temp1[3]
              + temp2[0] + temp2[1] + temp2[2] + temp2[3] ) * step_size;
  }

  double pi = 0.0;

  for (int i = 0; i < num_thrds; ++i)
    pi = pi + sum[i];
  
  return pi;
}


int main()
{
  int num_steps = 1024*1024*1024;
  int max_thrds = omp_get_max_threads();

  double start_time = omp_get_wtime();
  double pi         = pi_15_parvec_omp_for_avx2_dual_accum(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << max_thrds
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
