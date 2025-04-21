#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/wait.h>
#include "defaults.h"

int init(int N, int n, double complex ***h, double ***g, int ***c, int *C, double *sins, double **Q){
	int L = 2*N+1, q1, q2, i;
	char name[32];
	sprintf(name, "data/N=%d_dc=%d.dat", N, n);
	FILE *file = fopen(name, "r");
	if (file){
		double re, im;
		for (q1=0;q1<L;q1++){
			for (q2=0;q2<L;q2++){
				fscanf(file,"%d\t%d\n", &c[0][q1][q2], &c[1][q1][q2]);
				for (i=0; i<n; i++){
					fscanf(file,"%lf\t%lf\n", &re, &im );
					h[i][q1][q2] = re + im*I;
					fscanf(file,"%lf\n", &g[i][q1][q2]);
				}
			}
		}
		fscanf(file,"%d\n", C);
		fclose(file);
		return 1;
	}
	double a = 2.*pi/L;
	for (i=0;i<n;i++){
		h[i][0][0] = 1/a/a;
	}
	#pragma omp parallel for private(q1, q2, i) collapse(2)
	for (q1=0;q1<L;q1++){
		for(q2=0;q2<L;q2++){
			c[0][q1][q2] = 0;
			c[1][q1][q2] = 1;
			if (!q1&&!q2) continue;
			for (i=0;i<n;i++){
				h[i][q1][q2] = 1./Q[q1][q2];
				g[i][q1][q2] = h[i][q1][q2]*h[i][q1][q2];
			}
		}
	}
	return 0;
}

int calcS(int N, int n, double complex ***h, double complex **S, double *sins, double **Q){
	int L=2*N+1, q1, q2;
	#pragma omp parallel for collapse(2)
	for (q1=0;q1<L;q1++){
		for (q2=0;q2<L;q2++){
			S[q1][q2]=0;
			if (!q1&&!q2) continue;
			int i, k1, k2;
			for (k1=0;k1<L;k1++){
				for (k2=0;k2<L;k2++){
					double p = sins[k1]*sins[q2] - sins[k2]*sins[q1];
					p *= p;
					p /= Q[q1][q2];
					for (i=0; i<n; i++){
						S[q1][q2] += p * h[i][k1][k2] * conj(h[i][(k1+q1)%L][(k2+q2)%L]);
					}
				}
			}
		}
	}
	return 0;
}

int dump(int N, int n, int N8, double complex ***h, double complex **S, double ***g, int ***c, int C, double *sins){
	int L = 2*N + 1, q1, q2, i;
	double a = 2*pi/L;
	char name[32];
	mkdir("data", 0777);
	sprintf(name, "data/N=%d_dc=%d.dat", N, n);
	FILE *file = fopen(name, "w");
	if (!file){
		printf("data error\n");
		return -1;
	}
	for (q1=0;q1<L;q1++){
		for (q2=0;q2<L;q2++){
			fprintf(file,"%d\t%d\n", c[0][q1][q2], c[1][q1][q2]);
			for (i=0; i<n; i++){
				fprintf(file,"%.14lf\t%.14lf\n", creal(h[i][q1][q2]), cimag(h[i][q1][q2]));
        			fprintf(file,"%.14lf\n", g[i][q1][q2]);
			}
		}
	}
	fprintf(file,"%d\n", C);
	fclose(file);
	
	mkdir("AR", 0777);
	sprintf(name, "AR/AR_N=%d_dc=%d", N, n);
	FILE *filear = fopen(name, "w");
	if (!filear){
		printf("data error\n");
		return -1;
	}
        for (int k1 = 0; k1 < L; k1++){
	  for (int k2=0;k2<L;k2++){
	    fprintf(filear, "AR(h[%d][%d])=%.3lf\n", k1, k2, (double)c[0][k1][k2]/c[1][k1][k2]);
	  }
	}
	fclose(filear);

        mkdir("eta", 0777);
	sprintf(name, "eta/N=%u_dc=%u", N, n);
	file = fopen(name, "w");
	sprintf(name, "eta/fit");
	FILE *fite = fopen(name, "a");
	if (!file || !fite){
		printf("data error\n");
		return -1;
	}
	for (int i=1; i < N+1; i++) {
	        for (int j=0; j<n; j++){
	                double x = i*a, gy = c[1][0][i]/g[j][0][i], gx = c[1][i][0]/g[j][i][0], gr = c[1][i][i]/g[j][i][i], gR = c[1][i][(L-i)%L]/g[j][i][(L-i)%L];
		        fprintf(file,"%.14lf\t%.14lf\n", x, gy);
		        fprintf(file,"%.14lf\t%.14lf\n", x, gx);
		        fprintf(file,"%.14lf\t%.14lf\n", x*sqrt(2), gr);
		        fprintf(file,"%.14lf\t%.14lf\n", x*sqrt(2), gR);
		        if (i < 3) continue;
		        fprintf(fite,"%.14lf\t%.14lf\n", x, gy);
		        fprintf(fite,"%.14lf\t%.14lf\n", x, gx);
		        fprintf(fite,"%.14lf\t%.14lf\n", x*sqrt(2), gr);
		        fprintf(fite,"%.14lf\t%.14lf\n", x*sqrt(2), gR);
	        }
	}
	fclose(file);
	fclose(fite);
	file = fopen(name, "a");
	if (!file){
		printf("Cannot save data\n");
		return -1;
	}

	pid_t pid = fork();
	if (pid == -1) {
		perror("fork");
		exit(EXIT_FAILURE);
	}
	else{
		if (pid == 0) {
			char *zip_args[] = {"zip", "-r", "archive.zip", "data", "eta", "AR", NULL};
			execvp("zip", zip_args);
			perror("execvp");
			exit(EXIT_FAILURE);
		}
	}
	exit(EXIT_SUCCESS);
	return 0;
}
