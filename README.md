# OpenMP Design Patterns

A worked example demonstrating common OpenMP design patterns to compute the definite integral

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\pi&space;=\int_{0}^{1}&space;\frac{4}{&space;1&plus;x^{2}&space;}\&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\pi&space;=\int_{0}^{1}&space;\frac{4}{&space;1&plus;x^{2}&space;}\&space;dx" title="\large \pi =\int_{0}^{1} \frac{4}{ 1+x^{2} }\ dx" /></a>
</p>

using the midpoint rule with <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;2^{20}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;2^{20}" title="\small 2^{20}" /></a> subintervals.  Each example program is listed below, with a brief description of the algorithm and its implementation details underneath.  For simplicity, each algorithm assumes a number of subintervals evenly divisible by eight.

##### <code> pi_01_seq_loop.cxx </code>
Serial algorithm to estimate the definite integral by sequentially iterating over the subintervals to compute each Riemann sum.

##### <code> pi_02_seq_recursive.cxx </code>
Serial divide-and-conquer algorithm that uses an implementation function to perform tail recursive calls for a chosen block size.  Each recursive call computes a partial sum of Riemann sums over a distinct range of subintervals before being combined and returned by the algorithm.

##### <code> pi_03_par_spmd.cxx </code>
Parallel algorithm that uses a cyclic distribution of threads. The accumulator variable is promoted to an array with length equal to the number of threads employed, and each element of the array stores the partial sum computed by each thread.  Inside the parallel region, each thread ID is used to initialize a loop counter variable for each thread, which is then incremented by the total number of threads following each iteration.  Also demonstrated is how to check the number of threads employed at run-time within an OpenMP Parallel Section and return that number to the user.  Note that the cyclic distribution of threads explicitly disables vectorization.

##### <code> pi_04_par_spmd_padded.cxx </code>
Parallel algorithm that extends the SPMD implementation by promoting the array for storing partial sums to two dimensions.  The size of the second dimension is chosen to match the size of the L1 cache line, eliminating false sharing between CPU cores.  Only the first element of the second dimension is accessed by the routine.

##### <code> pi_05_par_spmd_critical.cxx </code>
Parallel algorithm that modifies the SPMD implementation by creating a private accumulator variable for each thread to store a partial sum.  Following the loop, an OpenMP Critical Section is established to combine each partial sum to a variable shared among all threads.

##### <code> pi_06_par_omp_for.cxx </code>
Parallel algorithm that uses the OpenMP Parallel For construct to define a work sharing loop, where the iterations of the loop are executed by the team of threads within the OpenMP Parallel Section. 

##### <code> pi_07_par_omp_for_reduction.cxx </code>
Parallel algorithm that uses the OpenMP Parallel For construct with a reduction clause on the accumulator variable.

##### <code> pi_08_par_omp_task.cxx </code>
Parallel divide-and-conquer algorithm that uses OpenMP Tasks to perform tail recursive calls for a chosen block size.  A single thread calls the implementation function, and each recursive call creates two new tasks, all working independently but concurrently over a distinct range of subintervals.

##### <code> pi_09_vec_unroll_4.cxx </code>
Serial algorithm that unrolls four iterations of the loop body, reducing the computational overhead by performing only one quarter of the instructions that control the loop.

##### <code> pi_10_vec_unroll_8.cxx </code>
Serial algorithm that unrolls eight iterations of the loop body.

##### <code> pi_11_vec_avx2.cxx </code>
Serial algorithm that uses explicit AVX2 SIMD instructions to perform vector operations on four subintervals per loop iteration.

##### <code> pi_12_vec_avx2_dual.cxx </code>
Serial algorithm that uses explicit AVX2 SIMD instructions with dual accumulator variables to perform vector operations on eight subintervals per loop iteration.

##### <code> pi_13_vec_omp_simd.cxx </code>
Serial algorithm that uses the OpenMP SIMD construct to direct the compiler to generate vector instructions for the loop.

##### <code> pi_14_parvec_omp_for_avx2.cxx </code>
Parallel algorithm that uses the OpenMP Parallel construct for multithreading and explicit AVX2 SIMD instructions for vector operations.

##### <code> pi_15_parvec_omp_for_avx2_dual.cxx </code>
Parallel algorithm that uses the OpenMP Parallel construct for multithreading and explicit AVX2 SIMD instructions with dual accumulator variables.

##### <code> pi_16_parvec_omp_for_simd.cxx </code>
Parallel algorithm that uses the OpenMP Parallel For SIMD construct for both multithreading and vector operations.
