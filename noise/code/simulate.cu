#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuComplex.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <curand_kernel.h>
#include "defaults.h"

__device__ static inline cuDoubleComplex cuRmul(cuDoubleComplex a, double b) {
	return cuCmul(a, make_cuDoubleComplex(b, 0.0));
}

__global__ void simulate(int M, cuDoubleComplex *h, cuDoubleComplex *S, double *beta, double *g, int *c, double Y, double *dS_re, double *dS_im, double *sines, double *Q, unsigned long long seed, double *F_all, double *av_h, double *av_cor_cor, double *av_h_h) {
	const int L = 2 * NN + 1;
	int steps = M;
	int total_id = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ double kk1, kk2, log_mh, F, d, A;
	__shared__ double s_F[block_size];
	__shared__ int k1, k2;
	__shared__ cuDoubleComplex z[dc];
	__shared__ double s_sines[2 * NN + 1];
	__shared__ curandState blockState;
	for (int i = threadIdx.x; i < L; i += blockDim.x) {
		s_sines[i] = sines[i];
	}
	if (threadIdx.x == 0) {
		curand_init(seed + blockIdx.x, 0, 0, &blockState);
	}
	for (int i = 0; i < steps; i++) {
		if ((i+1) % 100 == 0) {
			if (total_id == 1) {
				printf("Step №%d/%d\n", i + 1, steps);
			}
		}
		for (int j = 0; j < L * L; j++) {
			if (threadIdx.x == 0) {
				k1 = 0, k2 = 0;
				while (k1 == 0 && k2 == 0) {
					kk1 = curand_uniform(&blockState);
					kk2 = curand_uniform(&blockState);
					k1 = ((int)(kk1 * L) + 1) % L - NN;
					k2 = ((int)(kk2 * L) + 1) % L - NN;
					if (k1 < 0) k1 += L;
					if (k2 < 0) k2 += L;
				}
				log_mh = log(curand_uniform(&blockState));
				if (((k2 < r0n+1) + (k2 > L - r0n - 1) + (k1 < r0n + 1) + (k1 > L - r0n - 1)) > 0) {
					for (int zi = 0; zi < dc; zi++) {
						double re = curand_uniform_double(&blockState) - 0.5;
						double im = curand_uniform_double(&blockState) - 0.5;
						z[zi] = make_cuDoubleComplex(re * r0e, im * r0e);
					}
				} else {
					for (int zi = 0; zi < dc; zi++) {
						double re = curand_uniform_double(&blockState) - 0.5;
						double im = curand_uniform_double(&blockState) - 0.5;
						z[zi] = make_cuDoubleComplex(re * r0, im * r0);
					}
				}
				A = Q[k1 * L + k2];
				d = d0 / A / pow(Y / A, di);
			}
			for (int k = threadIdx.x; k < L * L; k += blockDim.x) {
				int q1 = k / L;
				int q2 = k % L;
				dS_re[blockIdx.x * L * L + q1 * L + q2] = 0.0;
				dS_im[blockIdx.x * L * L + q1 * L + q2] = 0.0;
				F_all[blockIdx.x * L * L + q1 * L + q2] = 0.0;
			}
			__syncthreads();
			for (int k = threadIdx.x; k < L * L; k += blockDim.x) {
				int q1 = k / L;
				int q2 = k % L;
				if (q1 < L && q2 < L) {
					if (!q1 && !q2) continue;
					double p = s_sines[k1] * s_sines[q2] - s_sines[k2] * s_sines[q1];
					double kq = s_sines[(k2 + q2) % L] * s_sines[q1] - s_sines[(k1 + q1) % L] * s_sines[q2];
					double qk = s_sines[(L + k1 - q1) % L] * s_sines[q2] - s_sines[(L + k2 - q2) % L] * s_sines[q1];
					for (int l = 0; l < dc; l++) {
						cuDoubleComplex mul1 = cuCmul(cuConj(h[blockIdx.x * dc * L * L + l * L * L + ((q1 - k1 + L) % L) * L + ((q2 - k2 + L) % L)]), cuConj(z[l]));
						cuDoubleComplex mul2 = cuCmul(h[blockIdx.x * dc * L * L + l * L * L + ((k1 - q1 + L) % L) * L + ((k2 - q2 + L) % L)], cuConj(z[l]));
						cuDoubleComplex mul3 = cuCmul(h[blockIdx.x * dc * L * L + l * L * L + ((2 * L - k1 - q1) % L) * L + ((2 * L - k2 - q2) % L)], z[l]);
						cuDoubleComplex mul4 = cuCmul(cuConj(h[blockIdx.x * dc * L * L + l * L * L + ((k1 + q1) % L) * L + ((k2 + q2) % L)]), z[l]);

						cuDoubleComplex s = cuCadd(cuCadd(cuRmul(mul1, p), cuRmul(mul2, kq)), cuCadd(cuRmul(mul3, qk), cuRmul(mul4, p)));
						if (!((q1 + 2 * k2) % L) && !((q2 + 2 * k2) % L)) {
							s = cuCadd(s, cuRmul(z[l], p * d * 239));
						}
						if (!((q1 - 2 * k1) % L) && !((q2 - 2 * k1) % L)) {
							s = cuCadd(s, cuRmul(cuConj(z[l]), p * d * 239));
						}
						s = cuRmul(s, d);
						dS_re[blockIdx.x * L * L + q1 * L + q2] += cuCreal(s);
						dS_im[blockIdx.x * L * L + q1 * L + q2] += cuCimag(s);
					}
					cuDoubleComplex dS = make_cuDoubleComplex(dS_re[blockIdx.x * L * L + q1 * L + q2], dS_im[blockIdx.x * L * L + q1 * L + q2]);
					F_all[blockIdx.x * L * L + q1 * L + q2] = cuCreal(cuCmul(cuCadd(cuRmul(S[blockIdx.x * L * L + q1 * L + q2], 239.0), dS), cuConj(dS)));
				}
			}
			__syncthreads();
			double local_sum = 0.0;
			for (int v = threadIdx.x; v < L * L; v += blockDim.x) {
				local_sum += F_all[blockIdx.x * L * L + v];
			}
			s_F[threadIdx.x] = local_sum;
			__syncthreads();
			for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
				if (threadIdx.x < offset) {
					s_F[threadIdx.x] += s_F[threadIdx.x + offset];
				}
				__syncthreads();
			}
			if (threadIdx.x == 0) {
				F = s_F[0];
				F *= -Y / (L * L);
				double Qkk = Q[k1 * L + k2];
				for (int f = 0; f < dc; f++) {
					F -= d * Qkk * cuCreal(cuCmul(cuCadd(cuRmul(cuCadd(cuRmul(h[blockIdx.x * dc * L * L + f * L * L + k1 * L + k2], 239.0), cuRmul(z[f], d)), Qkk), make_cuDoubleComplex(2 * beta[blockIdx.x * L * L + k1 * L + k2], 0.0)), cuConj(z[f])));
				}
			}
			
			if (threadIdx.x == dc + 1) {
				if (c && g) {
					c[2 * blockIdx.x * L * L + 1 * L * L + k1 * L + k2]++;
					c[2 * blockIdx.x * L * L + 1 * L * L + ((L - k1) % L) * L + (L - k2) % L]++;
				}
			}
			
			__syncthreads();

			if (F > log_mh) {
				for (int m = threadIdx.x; m < L * L; m += blockDim.x) {
					int q1 = m / L;
					int q2 = m % L;
					cuDoubleComplex dS = make_cuDoubleComplex(dS_re[blockIdx.x * L * L + q1 * L + q2], dS_im[blockIdx.x * L * L + q1 * L + q2]);
					S[blockIdx.x * L * L + q1 * L + q2] = cuCadd(S[blockIdx.x * L * L + q1 * L + q2], dS);
				}
				if (threadIdx.x < dc) {
					h[blockIdx.x * L * L * dc + threadIdx.x * L * L + k1 * L + k2] = cuCadd(h[blockIdx.x * L * L * dc + threadIdx.x * L * L + k1 * L + k2], cuRmul(z[threadIdx.x], d));
					h[blockIdx.x * L * L * dc + threadIdx.x * L * L + ((L - k1) % L) * L + ((L - k2) % L)] = cuCadd(h[blockIdx.x * L * L * dc + threadIdx.x * L * L + ((L - k1) % L) * L + ((L - k2) % L)], cuRmul(cuConj(z[threadIdx.x]), d));
				}
				if (threadIdx.x == dc) {
					if (c) {
						c[2 * blockIdx.x * L * L + 0 * L * L + k1 * L + k2]++;
						c[2 * blockIdx.x * L * L + 0 * L * L + ((L - k1) % L) * L + ((L - k2) % L)]++;
					}
				}
			}
			__syncthreads();
			if (c && g) {
				if (threadIdx.x < dc) {
					double a = cuCreal(cuCmul(h[blockIdx.x * L * L * dc + threadIdx.x * L * L + k1 * L + k2], cuConj(h[blockIdx.x * L * L * dc + threadIdx.x * L * L + k1 * L + k2])));
					g[blockIdx.x * L * L * dc + threadIdx.x * L * L + k1 * L + k2] += a;
					g[blockIdx.x * L * L * dc + threadIdx.x * L * L + ((L - k1) % L) * L + (L - k2) % L] += a;
				}
				if (threadIdx.x >= dc && threadIdx.x < 2 * dc) {
					// some calculations //
				}
				__syncthreads();
			}
		}
	}
}

