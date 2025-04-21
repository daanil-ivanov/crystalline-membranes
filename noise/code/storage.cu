#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include "defaults.h"
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuComplex.h>


__global__ void generate_noise(double *noise, unsigned long long seed) {
	const int L = 2 * NN + 1;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= len_beta * blockDim.x) return;

	curandState localState;
	curand_init(seed + tid, 0, 0, &localState);

	for (int idx = threadIdx.x; idx < L * L; idx += blockDim.x) {
		noise[blockIdx.x * L * L + idx] = sigma0 * curand_normal_double(&localState);
	}
}


int init(cuDoubleComplex *h, double *g, int *c) {
	const int L = 2 * NN + 1;
	int q1, q2, i, d;
	char sigma_str[64];
	sprintf(sigma_str, "%.14lf", sigma0);
	for (char *p = sigma_str; *p; p++) {
	    if (*p == '.')
		*p = '_';
	}
	char name[128];
	sprintf(name, "data/N=%d_dc=%d_sigma=%s_dat", NN, dc, sigma_str);
	FILE *file = fopen(name, "r");
	if (file){
		double re, im;
		for (i = 0; i < len_beta; i++) {
			rewind(file);
			for (q1 = 0; q1 < L; q1++){
				for (q2 = 0; q2 < L; q2++){
					fscanf(file,"%d\t%d\n", &c[2 * i * L * L + 0 * L * L + q1 * L + q2], &c[2 * i * L * L + 1 * L * L + q1 * L + q2]);
					for (d = 0; d < dc; d++) {
						fscanf(file,"%lf\t%lf\n", &re, &im);
						h[i * L * L * dc + d * L * L + q1 * L + q2] = make_cuDoubleComplex(re, im);
						fscanf(file, "%lf\n", &g[i * L * L * dc + d * L * L + q1 * L + q2]);
					}
				}
			}
		}
		fclose(file);
		return 1;
	}
	return 0;
}


int dump(double *g, int *c, double *av_av_cor_cor, double *av_av_h_h, double *av_cor_cor, double *av_h_h, cuDoubleComplex *h, cuDoubleComplex *dump_h, double *dump_g, int *dump_c) {
	const int L = 2 * NN + 1;
	double a = 2.0 * pi / L;

	#pragma omp parallel for collapse(4) reduction(+:av_av_cor_cor[:dc*L*L], av_av_h_h[:dc*L*L])
	for (int beta = 0; beta < len_beta; beta++) {
		for (int d = 0; d < dc; d++) {
			for (int q1 = 0; q1 < L; q1++) {
				for (int q2 = 0; q2 < L; q2++) {
					av_av_cor_cor[d * L * L + q1 * L + q2] += av_cor_cor[beta * dc * L * L + d * L * L + q1 * L + q2] / len_beta;
					av_av_h_h[d * L * L + q1 * L + q2] += av_h_h[beta * dc * L * L + d * L * L + q1 * L + q2] / len_beta;
				}
			}
		}
	}
	#pragma omp parallel for collapse(3)
	for (int d = 0; d < dc; d++) {
		for (int q1 = 0; q1 < L; q1++) {
			for (int q2 = 0; q2 < L; q2++) {
				av_av_cor_cor[d * L * L + q1 * L + q2] = abs(av_av_cor_cor[d * L * L + q1 * L + q2]);
				av_av_h_h[d * L * L + q1 * L + q2] = abs(av_av_h_h[d * L * L + q1 * L + q2]);
			}
		}
	}
	
	struct stat st = {0};
	if (stat("eta", &st) == -1) {
		if (mkdir("eta", 0777) != 0) {
			perror("error creating directory eta");
			return -1;
		}
	}
	char name[128];
	snprintf(name, sizeof(name), "eta/gr_N=%u_dc=%u_sigma=%.14lf_blocks=%u", NN, dc, sigma0, len_beta);
	for (char *p = name; *p; ++p)
		if (*p == '.')
			*p = '_';

	FILE *fit_file = fopen(name, "w");
	if (!fit_file) {
			perror("error creating directory data");
		return -1;
	}

	fprintf(fit_file, "%.14lf\n", sigma0);
	fprintf(fit_file, "%d\n", len_beta);

	for (int i = 1; i <= NN; i++) {
		double x = i * a;
		for (int j = 0; j < dc; j++) {
			double sum_gy = 0.0, sum_gx = 0.0, sum_gr = 0.0, sum_gR = 0.0;
			#pragma omp parallel for reduction(+:sum_gy,sum_gx,sum_gr,sum_gR)
			for (int k = 0; k < len_beta; k++) {
				double gy = c[2 * k * L * L + 1 * L * L + 0 * L + i] /
				g[k * L * L * dc + j * L * L + 0 * L + i];
				double gx = c[2 * k * L * L + 1 * L * L + i * L + 0] /
				g[k * L * L * dc + j * L * L + i * L + 0];
				double gr = c[2 * k * L * L + 1 * L * L + i * L + i] /
				g[k * L * L * dc + j * L * L + i * L + i];
				double gR = c[2 * k * L * L + 1 * L * L + i * L + ((L - i) % L)] /
				g[k * L * L * dc + j * L * L + i * L + ((L - i) % L)];
				sum_gy += gy;
				sum_gx += gx;
				sum_gr += gr;
				sum_gR += gR;
			}

			double avg_gy = sum_gy / len_beta;
			double avg_gx = sum_gx / len_beta;
			double avg_gr = sum_gr / len_beta;
			double avg_gR = sum_gR / len_beta;

			fprintf(fit_file, "%.14lf\t%.14lf\n", x, avg_gy);
			fprintf(fit_file, "%.14lf\t%.14lf\n", x, avg_gx);
			fprintf(fit_file, "%.14lf\t%.14lf\n", x * sqrt(2), avg_gr);
			fprintf(fit_file, "%.14lf\t%.14lf\n", x * sqrt(2), avg_gR);

			if (i >= 3) {
				fprintf(fit_file, "%.14lf\t%.14lf\n", x, avg_gy);
				fprintf(fit_file, "%.14lf\t%.14lf\n", x, avg_gx);
				fprintf(fit_file, "%.14lf\t%.14lf\n", x * sqrt(2), avg_gr);
				fprintf(fit_file, "%.14lf\t%.14lf\n", x * sqrt(2), avg_gR);
			}
		}
	}
	fclose(fit_file);

	for (int i = 0; i < L * L; i++) {
		dump_h[i] = make_cuDoubleComplex(0.0, 0.0);
		dump_g[i] = 0.0;
	}

	for (int i = 0; i < 2 * L * L; i++) {
		dump_c[i] = 0;
	}	
	
	for (int beta = 0; beta < len_beta; beta++) {
		for (int d = 0; d < dc; d++) {
			for (int q1 = 0; q1 < L; q1++) {
				for (int q2 = 0; q2 < L; q2++) {
					dump_h[q1 * L + q2] = cuCadd(dump_h[q1 * L + q2], make_cuDoubleComplex(cuCimag(h[beta * dc * L * L + d * L * L + q1 * L + q2]) / len_beta, cuCreal(h[beta * dc * L * L + d * L * L + q1 * L + q2]) / len_beta));
					dump_g[q1 * L + q2] += g[beta * L * L * dc + d * L * L + q1 * L + q2] / len_beta;
					if (d == 0) {
						dump_c[0 * L * L + q1 * L + q2] += c[2 * beta * L * L + 0 * L * L + q1 * L + q2];
						dump_c[1 * L * L + q1 * L + q2] += c[2 * beta * L * L + 1 * L * L + q1 * L + q2];
					}					
				}
			}
		}
	}
	for (int q1 = 0; q1 < L; q1++) {
		for (int q2 = 0; q2 < L; q2++) {
			dump_c[0 * L * L + q1 * L + q2] = dump_c[0 * L * L + q1 * L + q2] / len_beta;
			dump_c[1 * L * L + q1 * L + q2] = dump_c[1 * L * L + q1 * L + q2] / len_beta;
		}
	}

	st = {0};
	if (stat("data", &st) == -1) {
		if (mkdir("data", 0777) != 0) {
			perror("error creating directory data");
			return -1;
		}
	}
	
	snprintf(name, sizeof(name), "data/N=%d_dc=%d_sigma=%.14lf_dat", NN, dc, sigma0);
	for (char *p = name; *p; ++p)
		if (*p == '.')
			*p = '_';

	FILE *data_file = fopen(name, "w");
	if (!data_file) {
		perror("error opening file to write data");
		return -1;
	}
	
	for (int q1 = 0; q1 < L; q1++){
		for (int q2 = 0; q2 < L; q2++){
			fprintf(data_file,"%d\t%d\n", dump_c[0 * L * L + q1 * L + q2], dump_c[1 * L * L + q1 * L + q2]);
			for (int d = 0; d < dc; d++) {
				fprintf(data_file,"%lf\t%lf\n", cuCreal(dump_h[q1 * L + q2]), cuCimag(dump_h[q1 * L + q2]));
				fprintf(data_file, "%lf\n", dump_g[q1 * L + q2]);
			}
		}
	}
	fclose(data_file);
	return 1;
}
