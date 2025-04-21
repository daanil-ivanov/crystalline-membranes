#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include "defaults.h"

int main(){
	int n=n0, N=N0, M=M0, MTH=MTH0;
	int i, j, q1, q2, C=0, L=2*N+1, N8;
	printf("\nN=%d, n=%d, di=%.2lf, M=%d, MTH=%d, rad=%.2lf, edge rad=%.2lf, threads=%d\n", N, n, di, M, MTH, r0, r0e, omp_get_num_procs());
	double p8=p80, Y=(2*pi)/3*p8*p8, t0, t, a=2.*pi/L;
	srand(time(NULL));
	setbuf(stdout, NULL);
	N8=(int)(p8/pi*N);
	double complex ***h = (double complex ***)malloc(n*sizeof(double complex **));
	double ***g = (double ***)malloc(n*sizeof(double **));
	for (i = 0; i < n; i++) {
		h[i] = (double complex **)malloc(L*sizeof(double complex *));
		g[i] = (double **)malloc(L*sizeof(double *));
		for (j = 0; j < L; j++) {
			h[i][j] = (double complex *)malloc(L*sizeof(double complex));
			g[i][j] = (double *)malloc(L*sizeof(double));
		}	
	}
	double complex **S = (double complex **)malloc(L*sizeof(double complex*));
	double complex **dS = (double complex **)malloc(L*sizeof(double complex*));
	int ***c = malloc(2*sizeof(int**));
	*(c+0) = (int **)malloc(L*sizeof(int*));
	*(c+1) = (int **)malloc(L*sizeof(int*));
	double *sins  = (double *)malloc(L*sizeof(double));
	double **Q  = (double **)malloc(L*sizeof(double *));
	for (i = 0; i < L; i++) {
		*(S+i) = (double complex *)malloc(L*sizeof(double complex));
		*(dS+i) = (double complex *)malloc(L*sizeof(double complex));
		*(*(c+0)+i) = (int *)malloc(L*sizeof(int));
		*(*(c+1)+i) = (int *)malloc(L*sizeof(int));
		*(Q+i) = (double *)malloc(L*sizeof(double));
	}
	dS[0][0] = 0;
	for (q1 = -N; q1 < N+1; q1++) {
		sins[(q1+L)%L] = sin(a*q1);
		for (q2 = -N; q2 < N+1; q2++){
			Q[(q1+L)%L][(q2+L)%L] = 4*(sin(a*q1/2)*sin(a*q1/2)+sin(a*q2/2)*sin(a*q2/2));
		}
	}
	if (!init(N, n, h, g, c, &C, sins, Q)){
		calcS(N, n, h, S, sins, Q);
		t0 = omp_get_wtime();
		for (i=0; i<MTH; i++){
			t = omp_get_wtime();
			for (j=0; j<L*L; j++){
				simulate(N, Y, n, h, S, dS, NULL, NULL, sins, Q);
			}
			printf("\nMTH step №%d/%d\tstep time: %.2lf min", i+1, MTH, (omp_get_wtime()-t)/60);
		}
		printf("\nMTH time: \t%.2lf min\n", (omp_get_wtime()-t0)/60);
	}
	if (M) {
		calcS(N, n, h, S, sins, Q);
		t0 = omp_get_wtime();
		for (i=0; i<M; i++){
			t = omp_get_wtime();
			for (j=0; j<L*L; j++){
				simulate(N, Y, n, h, S, dS, g, c, sins, Q);
			}
			printf("\nM step №%d/%d\tstep time: %.2lf min", i+1, M, (omp_get_wtime()-t)/60);
		}
		C += M;
		printf("\nM time: \t%.2lf min\n", (omp_get_wtime()-t0)/60);
	}
	dump(N, n, N8, h, S, g, c, C, sins);
	for (i = 0; i < n; i++) {for (j = 0; j < L; j++) {free(h[i][j]);} free(h[i]);} free(h);
	for (i = 0; i < L; i++){{free(*(S+i)); free(*(dS+i)); free(*(g+i)); free(*(*(c+0)+i)); free(*(*(c+1)+i)); free(*(Q+i));}}
	free(S); free(dS); free(g); free(*(c+0)); free(*(c+1)); free(c); free(sins); free(Q);
	return 0;
}
