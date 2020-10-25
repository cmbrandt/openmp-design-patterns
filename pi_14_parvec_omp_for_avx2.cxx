// pi_14_parvec_omp_for_avx2.cxx


// Compile:
//    g++ -Wall -pedantic -std=c++17 -fopenmp -mavx2 -mfma -O3 pi_14_parvec_omp_for_avx2.cxx -o pi_14.exe

// Usage:
//    ./pi_14.exe


#include <iostream>
#include <vector>
#include <immintrin.h>
#include <omp.h>


double pi_14_parvec_omp_for_avx2(int num_steps)
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
    __m256d vsum  = _mm256_setzero_pd();

    #pragma omp for
      for (int i = 0; i < num_steps; i += 4) {
        double  idx = static_cast<double>(i);
        __m256d vidx = _mm256_broadcast_sd(&idx);
        __m256d vx   = _mm256_mul_pd(_mm256_add_pd(vidx, vcoef), vstep);
        __m256d vden = _mm256_fmadd_pd(vx, vx, vone);
                vsum = _mm256_add_pd(_mm256_div_pd(vfour, vden), vsum);
      }

    __m128d vl  = _mm256_castpd256_pd128(vsum);
    __m128d vh  = _mm256_extractf128_pd(vsum, 1);
            vl  = _mm_add_pd(vl, vh);
    __m128d h64 = _mm_unpackhi_pd(vl, vl);
    sum[id]     = _mm_cvtsd_f64(_mm_add_sd(vl, h64)) * step_size; 
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
  double pi         = pi_14_parvec_omp_for_avx2(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << max_thrds
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
