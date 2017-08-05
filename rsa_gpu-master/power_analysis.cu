#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <openssl/rsa.h>
#include <openssl/err.h>

#include <cuda.h>
#include <curand.h>
#include <helper_cuda.h>

#include "common.hh"
#include "rsa_cpu.hh"
#include "rsa_gpu.hh"
#include "tanc.hh"

#define checkCurandErrors(x) do { curandStatus_t status= (x);\
	if (status != CURAND_STATUS_SUCCESS) { \
	printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); \
	exit(EXIT_FAILURE);}} while(0)

/* msg copy: every trace will have one msg but msg_num replicas*/
__global__ void gpu_memcpy(int msg_num, int msg_size, int trace_num, WORD *msges) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= trace_num) return;
	int idx = tid * msg_num * msg_size;
	for (int i = 1; i < msg_num; i++) {
		for (int j = 0; j < msg_size; j++) {
			msges[idx + i * msg_size + j] = msges[idx + j];
		}
	}
}

/* squeeze out the unused or redundant msges*/
__global__ void gpu_memdecimate(int msg_num, int msg_size, int trace_num, WORD *msges) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= trace_num) return;
	int idx = tid * msg_num * msg_size;
	for (int i = 0; i < msg_size; i++) {
		msges[tid * msg_size + i] = msges[idx + i];
	}
}

int main(int argc, char *argv[]) {
	if (argc < 4) {
		printf("Usage: %s msg_num trace_num power_date_file\n", argv[0]);
		exit(1);
	}
	/* disable in/output buffering */
	setvbuf(stdin, NULL, _IONBF, 0);
	setvbuf(stdout, NULL, _IONBF, 0);

	srand(time(NULL));
	int key_bits = 512;
	const char *key_file = "private_key.pem";
	/* Generate key file if not exits */
	FILE *file =fopen(key_file, "r");
	if(file!=NULL) {
		fclose(file);
		printf("Key file %s exists.\n", key_file);
	}
	else {
		Gen_RSA_key(key_bits, key_file);
		printf("Key file %s generated.\n", key_file);
	}
	/* Read key file into memory */
	RSA *rsa = Read_key_file(key_file);
	assert(rsa != NULL);
	key_bits = RSA_size(rsa)*8;
	int key_bytes = key_bits/8;
	int msg_bytes = key_bytes;
	int msg_size = key_bytes/sizeof(WORD);
	printf("key bits: %d\n", key_bits);
	/* store key in to WORD array */
	WORD e[msg_size];
	WORD d[msg_size];
	WORD n[msg_size];
	WORD np[msg_size];
	WORD r_sqr[msg_size];

	WORD pd[msg_size/2];
	WORD qd[msg_size/2];
	WORD p[msg_size/2];
	WORD q[msg_size/2];
	WORD pp[msg_size/2];
	WORD qp[msg_size/2];
	WORD pr_sqr[msg_size/2];
	WORD qr_sqr[msg_size/2];
	WORD iqmp[msg_size/2];

	key_setup(rsa, e, d, n ,np, r_sqr,
		pd, qd, p, q, pp, qp, pr_sqr, qr_sqr, iqmp);
	int mem_bytes = key_bytes * 56 * 1024;
	gpu_setup(msg_size, d, n, np, r_sqr, mem_bytes,
		pd, qd, p, q, pp, qp, pr_sqr, qr_sqr, iqmp, rsa);

	int msg_num = 56;
	int trace_num = 1;
	if (argc >= 2) msg_num = atoi(argv[1]);
	if (argc >= 3) trace_num = atoi(argv[2]);
	printf("msg_num: %d, trace_num: %d, file: %s.\n", msg_num, trace_num, argv[3]);
	mem_bytes = msg_bytes*msg_num*trace_num;

	/* Random number Gen setup */
	WORD *msges;
	checkCudaErrors(cudaMalloc(&msges, mem_bytes));
	curandGenerator_t gen;
	checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, 0));
	checkCurandErrors(curandGenerate(gen, (unsigned int *)msges, 
		mem_bytes / sizeof(unsigned int)));
	checkCurandErrors(curandDestroyGenerator(gen));
	checkCudaErrors(cudaDeviceSynchronize());
	int grid_size = (trace_num + 256 -1) / 256;
	gpu_memdecimate<<<grid_size, 256>>>(msg_num, msg_size, trace_num, msges);

	/* load power data */
	char *file_name = argv[3];
	FILE *dfile = fopen(file_name, "rb");
	double *power_h = (double *) malloc(sizeof(double) * trace_num);
	fread(power_h, sizeof(double), trace_num, dfile);

	/* attack */
	WORD *msges0, *msges1, *msges_mont;
	checkCudaErrors(cudaMalloc(&msges0, msg_bytes*trace_num));
	checkCudaErrors(cudaMalloc(&msges1, msg_bytes*trace_num));
	checkCudaErrors(cudaMalloc(&msges_mont, msg_bytes*trace_num));
	/* pre-processing */
	grid_size = (msg_size * trace_num + BLK_SIZE -1) / BLK_SIZE;
	gpu_preprocessing<<<grid_size, BLK_SIZE>>>
		(msg_num * trace_num, msg_size, msges, msges0, msges_mont);
	/* print msges */
	/*WORD *msg_h = (WORD *)malloc(10 * msg_bytes);
	checkCudaErrors(cudaMemcpy(msg_h, msges_mont, 10*msg_bytes, cudaMemcpyDeviceToHost));
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < msg_size; j++) {
			printf("%016lx", msg_h[i*msg_size+j]);
		}
		printf("\n");
	}
	free(msg_h);*/
	dt_attack(msg_size, 1, trace_num, 3, 510/3, power_h, d, msges0, msges1, msges_mont);

	/* Clear memory */
	free(power_h);
	checkCudaErrors(cudaFree(msges0));
	checkCudaErrors(cudaFree(msges1));
	checkCudaErrors(cudaFree(msges_mont));
	RSA_free(rsa);
	checkCudaErrors(cudaFree(msges));
	gpu_reset();
	return 0;
}
