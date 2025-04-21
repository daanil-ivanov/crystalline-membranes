#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuComplex.h>
#include <omp.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include "defaults.h"
#include <curand_kernel.h>

int main(){
	int q1, q2, M = M0, MTH = MTH0, L = 2 * NN + 1, blocks = len_beta;
	double p8 = p80, Y = (2 * pi) / 3 * p8 * p8, t0, t = omp_get_wtime(), a = 2.0 * pi / L;
	printf("L=%d, N=%d, block size=%d, number of beta=%d, sigma=%.7lf\n", NN, L-1, block_size, len_beta, sigma0);
	printf("dc=%d, M=%d, MTH=%d, threads=%d\n", dc, M, MTH, omp_get_num_procs());
	setbuf(stdout, NULL);

	unsigned long long seed = time(NULL);

	double *device_beta;
	cudaMalloc(&device_beta, L * L * len_beta * sizeof(double));
	cuDoubleComplex *host_h = (cuDoubleComplex *)malloc(L * L * dc * len_beta * sizeof(cuDoubleComplex));
	cuDoubleComplex *device_h;
	cudaMalloc(&device_h, L * L * dc * len_beta * sizeof(cuDoubleComplex));

	cuDoubleComplex *device_S;
	cudaMalloc(&device_S, L * L * len_beta * sizeof(cuDoubleComplex));
	
	cuDoubleComplex *dump_h = (cuDoubleComplex*)malloc(L * L * sizeof(cuDoubleComplex));
	double *dump_g = (double*)malloc(L * L * sizeof(double));
	int *dump_c = (int*)malloc(2 * L * L * sizeof(int));
	
	double *device_dS_re;
	cudaMalloc(&device_dS_re, L * L * len_beta * sizeof(double));

	double *device_dS_im;
	cudaMalloc(&device_dS_im, L * L * len_beta * sizeof(double));

	double *host_g = (double *)malloc(L * L * dc * len_beta * sizeof(double));
	double *device_g;
	cudaMalloc(&device_g, L * L * dc * len_beta * sizeof(double));

	int *host_c = (int *)malloc(2 * L * L * len_beta * sizeof(int));
	int *device_c;
	cudaMalloc(&device_c, 2 * L * L * len_beta * sizeof(int));

	double *host_sines = (double *)malloc(L * sizeof(double));
	double *device_sines;
	cudaMalloc(&device_sines, L * sizeof(double));

	double *host_Q = (double *)malloc(L * L * sizeof(double));
	double *device_Q;
	cudaMalloc(&device_Q, L * L * sizeof(double));

	double *device_F_all;
	cudaMalloc(&device_F_all, L * L * len_beta * sizeof(double));

	double *device_av_h;
	cudaMalloc(&device_av_h, L * L * dc * len_beta * sizeof(double));
	cudaMemset(device_av_h, 0, L * L * dc * len_beta * sizeof(double));

	double *host_av_cor_cor = (double *)malloc(L * L * dc * len_beta * sizeof(double));
	double *device_av_cor_cor;
	cudaMalloc(&device_av_cor_cor, L * L * dc * len_beta * sizeof(double));
	cudaMemset(device_av_cor_cor, 0, L * L * dc * len_beta * sizeof(double));

	double *host_av_h_h = (double *)malloc(L * L * dc * len_beta * sizeof(double));
	double *device_av_h_h;
	cudaMalloc(&device_av_h_h, L * L * dc * len_beta * sizeof(double));
	cudaMemset(device_av_h_h, 0, L * L * dc * len_beta * sizeof(double));

	double *host_av_av_cor_cor = (double *)calloc(L * L * dc, sizeof(double));
	double *host_av_av_h_h = (double *)calloc(L * L * dc, sizeof(double));

	#pragma omp parallel for
	for (q1 = -NN; q1 < NN+1; q1++) {
		host_sines[(q1 + L) % L] = sin(a * q1);
	}
	#pragma omp parallel for collapse(2)
	for (q1 = -NN; q1 < NN + 1; q1++) {
		for (q2 = -NN; q2 < NN + 1; q2++) {
			host_Q[(((q1 + L) % L) * L)+((q2 + L) % L)] = 4 * (sin(a * q1 / 2) * sin(a * q1 / 2) + sin(a * q2 / 2) * sin(a * q2 / 2));
		}
	}
	cudaMemcpy(device_sines, host_sines, L * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_Q, host_Q, L * L * sizeof(double), cudaMemcpyHostToDevice);
	printf("initialization\n");
	
	printf("generating noise\n");
	cudaDeviceSynchronize();
	generate_noise<<<blocks, block_size>>>(device_beta, seed);
	printf("noise generated\n");
	cudaDeviceSynchronize();
	
	
	if (!init(host_h, host_g, host_c)) {
		printf("no previous data\n");
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < dc; i++) {
			for (int j = 0; j < len_beta; j++) {
				host_h[j * L * L * dc + i * L * L] = make_cuDoubleComplex(1 / (a * a), 0.0);
			}
		}
		#pragma omp parallel for collapse(2)
		for (int q1 = 0; q1 < L; q1++) {
			for(int q2 = 0; q2 < L; q2++) {
				if (!q1 && !q2) continue;
				for (int j = 0; j < len_beta; j++) {
					host_c[2 * j * L * L + 0 * L * L + q1 * L + q2] = 0;
					host_c[2 * j * L * L + 1 * L * L + q1 * L + q2] = 1;
					for (int i = 0; i < dc; i++) {
						host_h[j * L * L * dc + i * L * L + q1 * L + q2] = make_cuDoubleComplex(1.0 / host_Q[q1 * L + q2], 0);
						host_g[j * L * L * dc + i * L * L + q1 * L + q2] = cuCreal(cuCmul(host_h[j * L * L * dc + i * L * L + q1 * L + q2], host_h[j * L * L * dc + i * L * L + q1 * L + q2]));
					}
				}
			}
		}
		cudaMemcpy(device_h, host_h, L * L * dc * len_beta * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cudaMemcpy(device_g, host_g, L * L * dc * len_beta * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(device_c, host_c, 2 * L * L * len_beta * sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		printf("initial S calculating\n");
		calcS<<<blocks, block_size>>>(device_h, device_S, device_sines, device_Q);
		cudaDeviceSynchronize();

		t0 = omp_get_wtime();
		printf("thermalization stage\n");
		cudaDeviceSynchronize();
		simulate<<<blocks, block_size>>>(MTH, device_h, device_S, device_beta, NULL, NULL, Y, device_dS_re, device_dS_im, device_sines, device_Q, seed, device_F_all, device_av_h, device_av_cor_cor, device_av_h_h);
		cudaDeviceSynchronize();
		printf("thermalization time: \t%.2lf min\n", (omp_get_wtime()-t0)/60);
	}
	
	else {
		printf("previous avereged data is loaded\n");
		printf("initial S calculating\n");
		cudaMemcpy(device_h, host_h, L * L * dc * len_beta * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
		cudaMemcpy(device_g, host_g, L * L * dc * len_beta * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(device_c, host_c, 2 * L * L * len_beta * sizeof(int), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		calcS<<<blocks, block_size>>>(device_h, device_S, device_sines, device_Q);
		cudaDeviceSynchronize();
	}

	t0 = omp_get_wtime();
	printf("simulation stage\n");
	cudaDeviceSynchronize();
	simulate<<<blocks, block_size>>>(M, device_h, device_S, device_beta, device_g, device_c, Y, device_dS_re, device_dS_im, device_sines, device_Q, seed, device_F_all, device_av_h, device_av_cor_cor, device_av_h_h);
	cudaDeviceSynchronize();

	cudaFree(device_S);
	cudaFree(device_dS_re);
	cudaFree(device_dS_im);
	free(host_sines);
	cudaFree(device_sines);
	free(host_Q);
	cudaFree(device_Q);


	cudaMemcpy(host_h, device_h, L * L * dc * len_beta * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_g, device_g, L * L * dc * len_beta * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c, device_c, 2 * L * L * len_beta * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_av_cor_cor, device_av_cor_cor, L * L * dc * len_beta * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_av_h_h, device_av_h_h, L * L * dc * len_beta * sizeof(double), cudaMemcpyDeviceToHost);

	printf("simulation time: \t%.2lf min\n", (omp_get_wtime()-t0)/60);

	printf("saving data\n");
	cudaDeviceSynchronize();
	dump(host_g, host_c, host_av_av_cor_cor, host_av_av_h_h, host_av_cor_cor, host_av_h_h, host_h, dump_h, dump_g, dump_c);
	cudaFree(host_g);
	cudaFree(device_g);
	cudaFree(host_c);
	cudaFree(device_c);
	free(host_h);
	cudaFree(device_h);
	cudaDeviceReset();
	printf("total elapsed time: \t%.2lf min\n", (omp_get_wtime()-t)/60);
}
