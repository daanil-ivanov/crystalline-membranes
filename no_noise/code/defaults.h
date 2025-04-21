#define N0 50
#define n0 3
#define MTH0 100
#define M0 500
#define d0 2.6
#define di 0.13 //n=9 0.13 // n=7 0.13 // n=5 0.13 // n=1 0.13
#define r0 1.0 //n=9 0.42 // n=7 0.4 // n=5 0.55 // radius of z [-0.5; 0.5) -> [-0.5*r0; 0.5*r0)
#define r0e 1.0 //n=9 0.33 // n=7 0.35 // n=5 0.38 first and last r0n
#define r0n 10 //n=9 3 //n=7 10// n=5 10
#define p80 .3
#define pi 3.1415926535879323846

int init(int N, int n, double complex ***h, double ***g, int ***c, int *C, double *sins, double **Q);	
int calcS(int N, int n, double complex ***h, double complex **S, double *sins, double **Q);
//double calcPR(int N, int N8, double **g, int ***c, int C, double *px, double *sins);
int dump(int N, int n, int N8, double complex ***h, double complex **S, double ***g, int ***c, int C, double *sins);
int simulate(int N, double Y, int n, double complex ***h, double complex **S, double complex **dS, double ***g, int ***c, double *sins, double **Q);

