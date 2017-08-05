#ifndef RSA_GPU_H
#define RSA_GPU_H

#define BLK_SIZE (64)

#include <openssl/rsa.h>
#include <cuda_runtime.h>
#include "common.hh"

void gpu_setup(
	int msg_size, WORD *d, WORD *n, WORD *np, WORD *r_sqr, int mem_bytes,
	WORD *pd, WORD *qd, WORD *p, WORD *q, WORD *pp, WORD *qp, WORD *pr_sqr, 
	WORD *qr_sqr, WORD *iqmp, RSA *rsa, struct SW *sw = nullptr);
void gpu_reset();
void gpu_private_decrypt(
	int msg_num, const unsigned char *input, unsigned char *output, SW_Type swType = SW_Type::none);
void gpu_private_decrypt_crt(
	int msg_num, unsigned char *input, unsigned char *output);
void gpu_private_decrypt_crt2(
	int msg_num, unsigned char *input, unsigned char *output);

__global__ void gpu_modexp(int msg_num, int msg_size, WORD *input, WORD *output);
__global__ void gpu_modexp_sw(int msg_num, int msg_size, WORD *input, WORD *output);
__global__ void gpu_demont(int msg_num, int msg_size, WORD *msges_mont, WORD *msges);
__global__ void gpu_modexp_timing( int msg_num, int msg_size, WORD *input, uint64_t *output);
__global__ void gpu_modexp_reduction( int msg_num, int msg_size, WORD *input, uint64_t *output);
__global__ void gpu_modexp_timing_sw( int msg_num, int msg_size, WORD *input, uint64_t *output);
__global__ void gpu_preprocessing( int msg_num, int msg_size, WORD *input, WORD *output0, WORD *mont);

__global__ void gpu_bit0( int msg_num, int msg_size, WORD *msges_mont, WORD *msges, uint64_t *t);
__global__ void gpu_bit1( int msg_num, int msg_size, WORD *msges_mont, WORD *msges, uint64_t *t);
__global__ void gpu_bit0_reduction( int msg_num, int msg_size, WORD *msges_mont, WORD *msges, uint16_t *reduction);
__global__ void gpu_bit1_reduction( int msg_num, int msg_size, WORD *msges_mont, WORD *msges, uint16_t *reduction);

__global__ void gpu_mbit_reduction( int msg_num, int msg_size, WORD *msges_mont, WORD *input_msges,
	int bit_size, int bit_value, uint16_t *reduction, bool update);

__global__ void gpu_preprocessing_sw( int msg_num, int msg_size, WORD *input, WORD *output);
__global__ void gpu_reduction_sw
	(int msg_num, int msg_size, WORD *msges0, uint16_t *reduction, uint16_t frag, char update);
__global__ void gpu_demont_sw( int msg_num, int msg_size, WORD *msges, uint16_t frag);

__global__ void gpu_reduction_length
	(int msg_num, int msg_size, WORD *msges0, uint16_t *reduction, uint16_t length, int minSquare, char update);
__global__ void gpu_reduction_value
	(int msg_num, int msg_size, WORD *msges0, uint16_t *reduction, uint16_t value, int minSquare, char update);

#endif
