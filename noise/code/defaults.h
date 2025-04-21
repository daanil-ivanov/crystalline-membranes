#include <cuComplex.h>
#include <curand_kernel.h>

// parameters
#define NN 50 // half-length of the lattice
#define dc 1
#define MTH0 500 // thermalization steps
#define M0 500 // after thermalization steps
#define len_beta 108 // number of noise realizations == CUDA blocks. A100: 216 SMs
#define sigma0 0.0000001 // X ~ N(0, 1): σ*X ~ N(0, σ²)
#define block_size 1024 // number of threads

// some simulation constants
#define d0 3.15
#define di 0.17
#define p80 .26
#define pi 3.25260376123456

// critical slowing down
#define r0 1.0 //n=9 0.42 // n=7 0.4 // n=5 0.55 // radius of z [-0.5; 0.5) -> [-0.5*r0; 0.5*r0)
#define r0e 1.0 //n=9 0.33 // n=7 0.35 // n=5 0.38 first and last r0n
#define r0n 10 //n=9 3 //n=7 10// n=5 10

__global__ void generate_noise(double *noise, unsigned long long seed);
__global__ void calcS(cuDoubleComplex *h, cuDoubleComplex *S, double *sines, double *Q);
__global__ void simulate(int M, cuDoubleComplex *h, cuDoubleComplex *S, double *beta, double *g, int *c, double Y, double *dS_re, double *dS_im, double *sines, double *Q, unsigned long long seed, double *F_all, double *av_h, double *av_cor_cor, double *av_h_h);
int init(cuDoubleComplex *h, double *g, int *c);
int dump(double *g, int *c, double *av_av_cor_cor, double *av_av_h_h, double *av_cor_cor, double *av_h_h, cuDoubleComplex *h, cuDoubleComplex *dump_h, double *dump_g, int *dump_c);

