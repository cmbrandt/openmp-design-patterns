// pi_12_vec_avx2_dual.cxx


// Compile:
//    g++ -Wall -pedantic -mavx2 -mfma -std=c++17 -fopenmp -O3 pi_12_vec_avx2_dual.cxx -o pi_12.exe

// Usage:
//    ./pi_12.exe


#include <iostream>
#include <immintrin.h>
#include <omp.h>


double pi_12_vec_avx2_dual_accum(int num_steps)
{
  double step_size = 1.0 / num_steps;

  double one  = 1.0;
  double four = 4.0;

  __m256d vcoef = _mm256_set_pd(0.5, 1.5, 2.5, 3.5);
  __m256d vone  = _mm256_broadcast_sd(&one);
  __m256d vfour = _mm256_broadcast_sd(&four);
  __m256d vstep = _mm256_broadcast_sd(&step_size);
  __m256d vsum1 = _mm256_setzero_pd();
  __m256d vsum2 = _mm256_setzero_pd();

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

          vsum1 = _mm256_add_pd(vsum1, vsum2);
  __m128d vl    = _mm256_castpd256_pd128(vsum1);
  __m128d vh    = _mm256_extractf128_pd(vsum1, 1);
          vl    = _mm_add_pd(vl, vh);
  __m128d h64   = _mm_unpackhi_pd(vl, vl);
  return _mm_cvtsd_f64(_mm_add_sd(vl, h64)) * step_size; 
}


int main()
{
  int num_steps = 1024*1024*1024;

  double start_time = omp_get_wtime();
  double pi         = pi_12_vec_avx2_dual_accum(num_steps);
  double stop_time  = omp_get_wtime();

  std::cout << "\nthreads:  " << 1
            << "\nsteps:    " << num_steps
            << "\npi:       " << pi
            << "\ntime:     " << stop_time - start_time << std::endl;
}
