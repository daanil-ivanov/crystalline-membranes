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

int simulate(int N, double Y, int n, double complex ***h, double complex **S, double complex **dS, double ***g, int ***c, double *sins, double **Q) {
	int L = 2*N+1;
	int k1 = rand()%L-N, k2 = rand()%L-N, q1, q2, i;
	if (!k1 && !k2) {
		simulate(N, Y, n, h, S, dS, g, c, sins, Q);
		return 0;
	}
	if (k1 < 0) k1 += L;
	if (k2 < 0) k2 += L;
	double F = 0, A = Q[k1][k2], d = d0/A/pow(1+Y/A,di); A *= A;
	double complex *z = (double complex *)malloc(n*sizeof(double complex));
	for (i=0;i<n;i++){
	        if (((k2 < r0n+1) + (k2 > L-r0n-1) + (k1 < r0n+1) + (k1 > L-r0n-1)) > 0){
	          z[i] = r0e*(1.*rand()/RAND_MAX-.5) + r0e*(1.*rand()/RAND_MAX-.5)*I;
	        }
	        else {
		  z[i] = r0*(1.*rand()/RAND_MAX-.5) + r0*(1.*rand()/RAND_MAX-.5)*I;
		}
	}
	#pragma omp parallel for collapse(2)
	for (q1=0; q1<L; q1++){
		for (q2=0; q2<L; q2++){
			dS[q1][q2] = 0+0*I;
		}
	}
	
	#pragma omp parallel for collapse(2) reduction(+:F)
	for (q1 = 0; q1 < L; q1++){
		for (q2 = 0; q2 < L; q2++){
			if (!q1 && !q2) continue;
			double p = sins[k1]*sins[q2]-sins[k2]*sins[q1]; p *= p;
			double kq = sins[(k2+q2)%L]*sins[q1]-sins[(k1+q1)%L]*sins[q2]; kq *= kq;
			double qk = sins[(L+k1-q1)%L]*sins[q2]-sins[(L+k2-q2)%L]*sins[q1]; qk *= qk;
			for (int j=0;j<n;j++){
				double complex s = p*conj(h[j][(k1+q1)%L][(k2+q2)%L])*z[j];
				s += kq*h[j][(2*L-k1-q1)%L][(2*L-k2-q2)%L]*z[j];
				s += qk*h[j][(k1-q1+L)%L][(k2-q2+L)%L]*conj(z[j]);
				s +=  p*conj(h[j][(q1-k1+L)%L][(q2-k2+L)%L])*conj(z[j]);
				if (!((q1+2*k1)%L) && !((q2+2*k2)%L)) s += p*z[j]*z[j]*d;
				if (!((q1-2*k1+2*L)%L) && !((q2-2*k2+2*L)%L)) s += p*conj(z[j])*conj(z[j])*d;
				s *= d/Q[q1][q2];
				dS[q1][q2] += s;
			}
			F += creal((2*S[q1][q2]+dS[q1][q2])*conj(dS[q1][q2]));
		}
	}
	F *= -Y/L/L;
	for (i=0; i<n; i++){
		F -= A*creal((2*h[i][k1][k2] + d*z[i]) * conj(z[i]))*d;
	}
	//printf("\nF = %lf", F);
	if (F > log(1.*rand()/RAND_MAX)){
		for (i=0; i<n; i++){
			h[i][k1][k2] += d*z[i];
			h[i][(L-k1)%L][(L-k2)%L] += d*conj(z[i]);
		}
		#pragma omp parallel for collapse(2)
		for (q1 = 0; q1 < L; q1++){
			for (q2 = 0; q2 < L; q2++){
				S[q1][q2] += dS[q1][q2];
			}
		}
		if (c){
			c[0][k1][k2]++;
			c[0][(L-k1)%L][(L-k2)%L]++;
		}
	}
	if (c && g){
		for (int i=0; i<n; i++){
			double a = creal(h[i][k1][k2] * conj(h[i][k1][k2]));
			g[i][k1][k2] += a;
			g[i][(L-k1)%L][(L-k2)%L] += a;
		}
                c[1][k1][k2]++;
                c[1][(L-k1)%L][(L-k2)%L]++;
	}
	return 0;
}

