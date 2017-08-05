/* RSA GPU implementation GPU code
 * 10/29/2016 Chao Luo
 */

#include <cuda_runtime.h>
#include <assert.h>
#include <helper_cuda.h>
#include <stdint.h>
#include <stdio.h>

#include <openssl/bn.h>
#include <openssl/rsa.h>

#include "common.hh"
#include "rsa_gpu.hh"
#include "rsa_cpu.hh"

//#define BLK_SIZE (32)  /* 1 WORD per thread */

/* c: carry (may increment by 1)
   s: partial sum
   x, y: operands */
#define ADD_CARRY(c, s, x, y) \
		do { \
			WORD _t = (x) + (y); \
			(c) += (_t < (x)); \
			(s) = _t; \
		} while (0)

/* Same with ADD_CARRY, but sets y to 0 */
#define ADD_CARRY_CLEAR(c, s, x, y) \
		do { \
			WORD _t = (x) + (y); \
			(y) = 0; \
			(c) += (_t < (x)); \
			(s) = _t; \
		} while (0)

/* b: borrow (may increment by 1)
   d: partial difference
   x, y: operands (a - b) */
#define SUB_BORROW(b, d, x, y) \
		do { \
			WORD _t = (x) - (y); \
			(b) += (_t > (x)); \
			(d) = _t; \
		} while (0)

/* Same with SUB_BORROW, but sets y to 0 */
#define SUB_BORROW_CLEAR(b, d, x, y) \
		do { \
			WORD _t = (x) - (y); \
			(y) = 0; \
			(b) += (_t > (x)); \
			(d) = _t; \
		} while (0)

// Return most significant WORD
__device__ WORD mp_umul_hi(WORD a, WORD b) {
#ifdef USE_64BIT
	return __umul64hi(a, b);
#else
	return __umulhi(a, b);
#endif
}
// Return least significant WORD
__device__ WORD mp_umul_lo(WORD a, WORD b) {
	return a * b;
}

// RSA key info in constant memory
__constant__ WORD d_d[MAX_MSG_SIZE]; 
__constant__ WORD n_d[MAX_MSG_SIZE]; 
__constant__ WORD np_d[MAX_MSG_SIZE]; 
__constant__ WORD r_sqr_d[MAX_MSG_SIZE]; 

__constant__ WORD pqd_d[MAX_MSG_SIZE]; 
__constant__ WORD pq_d[MAX_MSG_SIZE]; 
__constant__ WORD pqp_d[MAX_MSG_SIZE]; 
__constant__ WORD pqr_sqr_d[MAX_MSG_SIZE]; 
__constant__ WORD iqmp_d[MAX_MSG_SIZE/2]; 
__constant__ struct SW sw_d;
// Device Memory
static WORD *input_d;
static WORD *output_d;

static int MSG_SIZE; 		/* msg in terms of WORD */
static RSA* rsa_gpu;

// Setup RSA key in device constant memory
/*void gpu_setup(
	int msg_size, WORD *d, WORD *n, WORD *np, WORD *r_sqr, int mem_bytes,
	WORD *pd, WORD *qd, WORD *p, WORD *q, WORD *pp, WORD *qp, WORD *pr_sqr, WORD *qr_sqr,
	WORD *iqmp, RSA *rsa) {
    printf("gpu_setup\n");
	rsa_gpu = rsa;
	MSG_SIZE = msg_size;
	int msg_bytes = msg_size * sizeof(WORD);
	checkCudaErrors(cudaMemcpyToSymbol(d_d, d, msg_bytes));
	checkCudaErrors(cudaMemcpyToSymbol(n_d, n, msg_bytes));
	checkCudaErrors(cudaMemcpyToSymbol(np_d, np, msg_bytes));
	checkCudaErrors(cudaMemcpyToSymbol(r_sqr_d, r_sqr, msg_bytes));
	
	WORD tmp[MAX_MSG_SIZE];
	memcpy(tmp, pd, msg_bytes/2);
	memcpy(tmp + msg_size/2, qd, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pqd_d, tmp, msg_bytes));

	memcpy(tmp, p, msg_bytes/2);
	memcpy(tmp + msg_size/2, q, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pq_d, tmp, msg_bytes));

	memcpy(tmp, pp, msg_bytes/2);
	memcpy(tmp + msg_size/2, qp, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pqp_d, tmp, msg_bytes));

	memcpy(tmp, pr_sqr, msg_bytes/2);
	memcpy(tmp + msg_size/2, qr_sqr, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pqr_sqr_d, tmp, msg_bytes));

	checkCudaErrors(cudaMemcpyToSymbol(iqmp_d, iqmp, msg_bytes /2));

	checkCudaErrors(cudaMalloc((void **)&input_d, mem_bytes));
	checkCudaErrors(cudaMalloc((void **)&output_d, mem_bytes));
}*/


// Overloaded function with CLSW
void gpu_setup(
	int msg_size, WORD *d, WORD *n, WORD *np, WORD *r_sqr, int mem_bytes,
	WORD *pd, WORD *qd, WORD *p, WORD *q, WORD *pp, WORD *qp, WORD *pr_sqr, WORD *qr_sqr,
	WORD *iqmp, RSA *rsa, struct SW *sw) {
    //printf("gpu_setup with sliding window\n");
	//printf("sw first window length =  %d, value = %d\n", sw->length[0], sw->fragment[0]);
	rsa_gpu = rsa;
	MSG_SIZE = msg_size;
	int msg_bytes = msg_size * sizeof(WORD);
	checkCudaErrors(cudaMemcpyToSymbol(d_d, d, msg_bytes));
	checkCudaErrors(cudaMemcpyToSymbol(n_d, n, msg_bytes));
	checkCudaErrors(cudaMemcpyToSymbol(np_d, np, msg_bytes));
	checkCudaErrors(cudaMemcpyToSymbol(r_sqr_d, r_sqr, msg_bytes));
	if (sw)
		checkCudaErrors(cudaMemcpyToSymbol(sw_d, sw, sizeof(struct SW)));
	
	WORD tmp[MAX_MSG_SIZE];
	memcpy(tmp, pd, msg_bytes/2);
	memcpy(tmp + msg_size/2, qd, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pqd_d, tmp, msg_bytes));

	memcpy(tmp, p, msg_bytes/2);
	memcpy(tmp + msg_size/2, q, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pq_d, tmp, msg_bytes));

	memcpy(tmp, pp, msg_bytes/2);
	memcpy(tmp + msg_size/2, qp, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pqp_d, tmp, msg_bytes));

	memcpy(tmp, pr_sqr, msg_bytes/2);
	memcpy(tmp + msg_size/2, qr_sqr, msg_bytes/2);
	checkCudaErrors(cudaMemcpyToSymbol(pqr_sqr_d, tmp, msg_bytes));

	checkCudaErrors(cudaMemcpyToSymbol(iqmp_d, iqmp, msg_bytes /2));

	checkCudaErrors(cudaMalloc((void **)&input_d, mem_bytes));
	checkCudaErrors(cudaMalloc((void **)&output_d, mem_bytes));
}

// Free device memory & reset GPU
void gpu_reset() {
	checkCudaErrors(cudaFree(input_d));
	checkCudaErrors(cudaFree(output_d));
	checkCudaErrors(cudaDeviceReset());
}

// Montgomery multiplication
__device__ void mp_montmul_dev(WORD *ret, WORD *ar, WORD *br,
	int limb_idx, int idx, int msg_size, WORD *n, WORD np) {
	__shared__ WORD _t[2*BLK_SIZE];
	__shared__ WORD _c[2*BLK_SIZE];

	volatile WORD *t = _t + 2 * msg_size * limb_idx;
	volatile WORD *c = _c + 2 * msg_size * limb_idx;

	c[idx] = 0;
	c[idx + msg_size] = 0;
	t[idx] = 0;
	t[idx + msg_size] = 0;

	for (int i=0; i<msg_size; i++) {
		WORD hi = mp_umul_hi(ar[i], br[idx]);
		WORD lo = mp_umul_lo(ar[i], br[idx]);

		ADD_CARRY(c[i+idx+1], t[i+idx+1], t[i+idx+1], hi);
		ADD_CARRY(c[i+idx], t[i+idx], t[i+idx], lo);

		WORD m = t[i] * np;
		hi = mp_umul_hi(m, n[idx]);
		lo = mp_umul_lo(m, n[idx]);

		ADD_CARRY(c[idx+i+1], t[idx+i+1], t[idx+i+1], hi);
		ADD_CARRY(c[idx+i], t[idx+i], t[idx+i], lo);
		ADD_CARRY_CLEAR(c[idx+i+1], t[idx+i+1], t[idx+i+1], c[idx+i]);
	}

	/* here all t[0] ~ t[msg_size - 1] should be zero. c too */
	while (__any(c[idx + msg_size - 1] != 0))
		ADD_CARRY_CLEAR(c[idx + msg_size], t[idx + msg_size], 
			t[idx + msg_size], c[idx + msg_size - 1]);

	/* step 2: return t or t - n */
	if (c[msg_size * 2 - 1])		// c may be 0 or 1, but not 2
		goto u_is_bigger;

	/* Ugly, but practical.
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = msg_size - 1; i >= 0; i--) {
		if (t[i + msg_size] > n[i])
			goto u_is_bigger;
		if (t[i + msg_size] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	/* return t - n. Here, c is used for borrow */
	SUB_BORROW(c[idx], ret[idx], t[idx + msg_size], n[idx]);

	if (idx < msg_size - 1) {
		while (__any(c[idx] != 0)) {
			SUB_BORROW_CLEAR(c[idx + 1], ret[idx + 1],
					ret[idx + 1], c[idx]);
		}
	}
	return;

n_is_bigger:
	/* return t */
	ret[idx] = t[idx + msg_size];
	return;
}


/* Record the extra reduction in Montgomery multiplication
 * Reduction is recorded in bits
 */
template<class UINT>
__device__ void mp_montmul_dev2(WORD *ret, WORD *ar, WORD *br,
	int limb_idx, int idx, int msg_size, WORD *n, WORD np, UINT *reduction, int rdct_idx) {
	__shared__ WORD _t[2*BLK_SIZE];
	__shared__ WORD _c[2*BLK_SIZE];

	volatile WORD *t = _t + 2 * msg_size * limb_idx;
	volatile WORD *c = _c + 2 * msg_size * limb_idx;

	c[idx] = 0;
	c[idx + msg_size] = 0;
	t[idx] = 0;
	t[idx + msg_size] = 0;

	for (int i=0; i<msg_size; i++) {
		WORD hi = mp_umul_hi(ar[i], br[idx]);
		WORD lo = mp_umul_lo(ar[i], br[idx]);

		ADD_CARRY(c[i+idx+1], t[i+idx+1], t[i+idx+1], hi);
		ADD_CARRY(c[i+idx], t[i+idx], t[i+idx], lo);

		WORD m = t[i] * np;
		hi = mp_umul_hi(m, n[idx]);
		lo = mp_umul_lo(m, n[idx]);

		ADD_CARRY(c[idx+i+1], t[idx+i+1], t[idx+i+1], hi);
		ADD_CARRY(c[idx+i], t[idx+i], t[idx+i], lo);
		ADD_CARRY_CLEAR(c[idx+i+1], t[idx+i+1], t[idx+i+1], c[idx+i]);
	}

	/* here all t[0] ~ t[msg_size - 1] should be zero. c too */
	while (__any(c[idx + msg_size - 1] != 0))
		ADD_CARRY_CLEAR(c[idx + msg_size], t[idx + msg_size], 
			t[idx + msg_size], c[idx + msg_size - 1]);

	/* step 2: return t or t - n */
	if (c[msg_size * 2 - 1])		// c may be 0 or 1, but not 2
		goto u_is_bigger;

	/* Ugly, but practical.
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = msg_size - 1; i >= 0; i--) {
		if (t[i + msg_size] > n[i])
			goto u_is_bigger;
		if (t[i + msg_size] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	if (idx == 0)
		*reduction |= ((UINT)1 << rdct_idx);
	/* return t - n. Here, c is used for borrow */
	SUB_BORROW(c[idx], ret[idx], t[idx + msg_size], n[idx]);

	if (idx < msg_size - 1) {
		while (__any(c[idx] != 0)) {
			SUB_BORROW_CLEAR(c[idx + 1], ret[idx + 1],
					ret[idx + 1], c[idx]);
		}
	}
	return;

n_is_bigger:
	/* return t */
	if (idx == 0)
		*reduction &= ~((UINT)1 << rdct_idx);
	ret[idx] = t[idx + msg_size];
	return;
}


/* Record the extra reduction in Montgomery multiplication
 * Reduction are summed.
 */
template<class UINT>
__device__ void mp_montmul_dev3(WORD *ret, WORD *ar, WORD *br,
	int limb_idx, int idx, int msg_size, WORD *n, WORD np, UINT *reduction) {
	__shared__ WORD _t[2*BLK_SIZE];
	__shared__ WORD _c[2*BLK_SIZE];

	volatile WORD *t = _t + 2 * msg_size * limb_idx;
	volatile WORD *c = _c + 2 * msg_size * limb_idx;

	c[idx] = 0;
	c[idx + msg_size] = 0;
	t[idx] = 0;
	t[idx + msg_size] = 0;

	for (int i=0; i<msg_size; i++) {
		WORD hi = mp_umul_hi(ar[i], br[idx]);
		WORD lo = mp_umul_lo(ar[i], br[idx]);

		ADD_CARRY(c[i+idx+1], t[i+idx+1], t[i+idx+1], hi);
		ADD_CARRY(c[i+idx], t[i+idx], t[i+idx], lo);

		WORD m = t[i] * np;
		hi = mp_umul_hi(m, n[idx]);
		lo = mp_umul_lo(m, n[idx]);

		ADD_CARRY(c[idx+i+1], t[idx+i+1], t[idx+i+1], hi);
		ADD_CARRY(c[idx+i], t[idx+i], t[idx+i], lo);
		ADD_CARRY_CLEAR(c[idx+i+1], t[idx+i+1], t[idx+i+1], c[idx+i]);
	}

	/* here all t[0] ~ t[msg_size - 1] should be zero. c too */
	while (__any(c[idx + msg_size - 1] != 0))
		ADD_CARRY_CLEAR(c[idx + msg_size], t[idx + msg_size], 
			t[idx + msg_size], c[idx + msg_size - 1]);

	/* step 2: return t or t - n */
	if (c[msg_size * 2 - 1])		// c may be 0 or 1, but not 2
		goto u_is_bigger;

	/* Ugly, but practical.
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = msg_size - 1; i >= 0; i--) {
		if (t[i + msg_size] > n[i])
			goto u_is_bigger;
		if (t[i + msg_size] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	if (idx == 0)
		*reduction += 1;
	/* return t - n. Here, c is used for borrow */
	SUB_BORROW(c[idx], ret[idx], t[idx + msg_size], n[idx]);

	if (idx < msg_size - 1) {
		while (__any(c[idx] != 0)) {
			SUB_BORROW_CLEAR(c[idx + 1], ret[idx + 1],
					ret[idx + 1], c[idx]);
		}
	}
	return;

n_is_bigger:
	/* return t */
	ret[idx] = t[idx + msg_size];
	return;
}

// Exponentiate msg with private key, including Montgomerization and deMontgomerization
__global__ void gpu_modexp(
	int msg_num, int msg_size, WORD *input, WORD *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];

	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];

	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = ar */
	tmp[idx] = ret[idx];

	int t = msg_size * BITS_PER_WORD - 1; /* bit index of d_d */
	while (((d_d[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 0 && t>0)
		t--;
	t--;

	while (t >= 0) {
		mp_montmul_dev(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
	
		if (((d_d[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 1) {
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
		}
		t--;
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);

	output[msg_idx*msg_size + idx] = ret[idx];
}

// msg odd power memory
__device__ WORD ar_pow[SW_MAX_FRAGMENT / 2][MAX_MSG_NUM][MAX_MSG_SIZE];

// Using CLSW in exponentiating
__global__ void gpu_modexp_sw(
	int msg_num, int msg_size, WORD *input, WORD *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];

	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;
	int num_frags = sw_d.num_fragments;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];

	mp_montmul_dev(tmp, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* tmp = ar */
	ar_pow[0][msg_idx][idx] = tmp[idx];
	
	mp_montmul_dev(ret, tmp, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = a*a*r */

	for (int i = 3; i <= sw_d.max_fragment; i += 2) {
		mp_montmul_dev(tmp, tmp, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
		ar_pow[i >> 1][msg_idx][idx] = tmp[idx];
	}

	ret[idx] = ar_pow[sw_d.fragment[num_frags - 1] >> 1][msg_idx][idx];
	
	for (int i = num_frags - 2; i >= 0; i--) {
		for (int k = sw_d.length[i]; k >= 1; k--)
			mp_montmul_dev(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
		if (sw_d.fragment[i]) {
			tmp[idx] = ar_pow[sw_d.fragment[i] >> 1][msg_idx][idx];
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
		}
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);

	output[msg_idx*msg_size + idx] = ret[idx];
}

// Private decryption of msg
void gpu_private_decrypt(
	int msg_num, const unsigned char* input, unsigned char * output, SW_Type swType) {
	int mem_bytes = msg_num * MSG_SIZE * sizeof(WORD);
	int msg_bytes = MSG_SIZE *	sizeof(WORD);
	/* Big endian to little endian */
	unsigned char *tmp = (unsigned char *)malloc(mem_bytes);
	unsigned char *tmp_;
	for (int j=0; j<msg_num; j++) {
        const unsigned char *inout_;
		tmp_ = tmp + j * msg_bytes;
		inout_ = input + j * msg_bytes;
		for(int i=0; i<msg_bytes; i++) {
			tmp_[i] = inout_[msg_bytes-1-i];
		}
	}
	/* Copy input into device memory */
	checkCudaErrors(cudaMemcpy(
		input_d, tmp, mem_bytes, cudaMemcpyHostToDevice));

	int grid_size = (msg_num * MSG_SIZE + BLK_SIZE -1) / BLK_SIZE;
	if (swType == SW_Type::none)
		gpu_modexp<<<grid_size, BLK_SIZE>>>(msg_num, MSG_SIZE, input_d, output_d);
	else
		gpu_modexp_sw<<<grid_size, BLK_SIZE>>>(msg_num, MSG_SIZE, input_d, output_d);

	/* Copy output into host memory */
	checkCudaErrors(cudaMemcpy(
		tmp, output_d, mem_bytes, cudaMemcpyDeviceToHost));
	/* Little endian to big endian */
	for(int j=0; j<msg_num; j++) {
        unsigned char *inout_;
		tmp_ = tmp + j * msg_bytes;
		inout_ = output + j * msg_bytes;
		for(int i=0; i<msg_bytes; i++) {
			inout_[i] = tmp_[msg_bytes-1-i];
		}
	}
	/* Release memory */
	free(tmp);
}

/* Exponentiation with CRT, Put p q in the same block */
__global__ void gpu_modexp_crt(int msg_num, int msg_size, WORD *input, WORD *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];

	/* Timing */
	/*long long int tick;
	if(threadIdx.x == 0 && blockIdx.x == 0)
		tick = clock64(); */
	
	/* msg_idx is interm of sub_msg */
	const int sub_msg_size = msg_size / 2;
	const int limb_idx = threadIdx.x / sub_msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / sub_msg_size + limb_idx;
	const int idx = threadIdx.x % sub_msg_size;
	const int pair_idx = msg_idx % 2;

	if(msg_idx >= msg_num*2) return;

	WORD *ret = _ret + limb_idx * sub_msg_size;
	WORD *tmp = _tmp + limb_idx * sub_msg_size;
	const WORD *exp = pqd_d + pair_idx * sub_msg_size;
	WORD *n = pq_d + pair_idx * sub_msg_size;
	WORD np = *(pqp_d + pair_idx * sub_msg_size);

	tmp[idx] = input[msg_idx * sub_msg_size + idx];
	ret[idx] = pqr_sqr_d[idx + pair_idx * sub_msg_size];

	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, sub_msg_size, n, np); /* ret = ar */
	tmp[idx] = ret[idx];

	int t = sub_msg_size * BITS_PER_WORD - 1; /* bit index of n_d */
	while (((exp[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 0 && t>0)
		t--;
	t--;

	while (t >= 0) {
		mp_montmul_dev(ret, ret, ret, limb_idx, idx, sub_msg_size, n, np);
	
		if (((exp[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 1) {
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, sub_msg_size, n, np);
		}
		t--;
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, sub_msg_size, n, np);

	output[msg_idx*sub_msg_size + idx] = ret[idx];

	/* Store timing in output[0] */
	/*if(threadIdx.x == 0 && blockIdx.x == 0)
		*(long long int*)output = clock64() - tick;*/
}

// Private decryption with CRT
void gpu_private_decrypt_crt(
	int msg_num, unsigned char* input, unsigned char * output) {
	int mem_bytes = msg_num * MSG_SIZE * sizeof(WORD);
	int msg_bytes = MSG_SIZE *	sizeof(WORD);
	int sub_msg_bytes = msg_bytes / 2;
	unsigned char *tmp = (unsigned char *)malloc(mem_bytes);
	unsigned char *tmp_, *inout_;
	BIGNUM *bn_input, *bn_pq;
	BN_CTX *bn_ctx;
	bn_input = BN_new();
	bn_pq = BN_new();
	bn_ctx = BN_CTX_new();

	for(int i=0; i<msg_num; i++) {
		BN_bin2bn(input+msg_bytes*i, msg_bytes, bn_input);
		assert(bn_input != NULL);
		BN_nnmod(bn_pq, bn_input, rsa_gpu->p, bn_ctx);
		mp_bn2mp((WORD *)(tmp+msg_bytes*i), bn_pq, MSG_SIZE/2);
		BN_nnmod(bn_pq, bn_input, rsa_gpu->q, bn_ctx);
		mp_bn2mp((WORD *)(tmp+msg_bytes*i+msg_bytes/2), bn_pq, MSG_SIZE/2);
	}

	/* Copy input into device memory */
	checkCudaErrors(cudaMemcpy(
		input_d, tmp, mem_bytes, cudaMemcpyHostToDevice));

	int grid_size = (msg_num * MSG_SIZE + BLK_SIZE -1) / BLK_SIZE;
	for (int i=0; i<1; i++) { /* repeat decryption */
		gpu_modexp_crt<<<grid_size, BLK_SIZE>>>(msg_num, MSG_SIZE, input_d, output_d);
	}

	/* Copy output into host memory */
	checkCudaErrors(cudaMemcpy(
		tmp, output_d, mem_bytes, cudaMemcpyDeviceToHost));

	/* Save timing info */
	long long int time_info = *(long long int *)tmp;

	/* Little endian to big endian */
	for(int j=0; j<msg_num*2; j++) {
		tmp_ = tmp + j * sub_msg_bytes;
		inout_ = output + j * sub_msg_bytes;
		for(int i=0; i<sub_msg_bytes; i++) {
			inout_[i] = tmp_[sub_msg_bytes-1-i];
		}
	}
	/* CRT combine */
	BIGNUM *m1, *m2;
	m1 = BN_new();
	m2 = BN_new();
	for(int i=0; i<msg_num; i++) {
		inout_ = output + i * msg_bytes;
		BN_bin2bn(inout_, sub_msg_bytes, m1);
		BN_bin2bn(inout_+sub_msg_bytes, sub_msg_bytes, m2);
		BN_mod_sub(m1, m1, m2, rsa_gpu->p, bn_ctx);
		BN_mod_mul(m1, m1, rsa_gpu->iqmp, rsa_gpu->p, bn_ctx);
		BN_mul(m1, m1, rsa_gpu->q, bn_ctx);
		BN_add(m1, m1, m2);
		memset(inout_, 0, msg_bytes);
		/* Skip the zeros at the beginning of M1*/
		int zero_num = msg_bytes - BN_num_bytes(m1);
		BN_bn2bin(m1, inout_+zero_num);
	}

	/* Put timing info into output */
	*(long long int *)output = time_info;

	/* Release memory */
	free(tmp);
	BN_free(bn_input);
	BN_free(bn_pq);
	BN_CTX_free(bn_ctx);
	BN_free(m1);
	BN_free(m2);
}

/* Exponentiation with CRT, put p q sub_msg into different blocks */
__global__ void gpu_modexp_crt2(int msg_num, int msg_size, WORD *input, WORD *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];

	/* Timing */
	//long long int tick;
	//if(threadIdx.x == 0 && blockIdx.x == 0)
	//	tick = clock64();
	
	/* msg_idx is interm of sub_msg */
	const int sub_msg_size = msg_size / 2;
	const int limb_idx = threadIdx.x / sub_msg_size;
	const int idx = threadIdx.x % sub_msg_size;
	const int pair_idx = blockIdx.x % 2;
	const int msg_idx = (blockIdx.x / 2 * blockDim.x / sub_msg_size + limb_idx)*2 + pair_idx;

	if(msg_idx >= msg_num*2) return;

	WORD *ret = _ret + limb_idx * sub_msg_size;
	WORD *tmp = _tmp + limb_idx * sub_msg_size;
	const WORD *exp = pqd_d + pair_idx * sub_msg_size;
	WORD *n = pq_d + pair_idx * sub_msg_size;
	WORD np = *(pqp_d + pair_idx * sub_msg_size);

	tmp[idx] = input[msg_idx * sub_msg_size + idx];
	ret[idx] = pqr_sqr_d[idx + pair_idx * sub_msg_size];

	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, sub_msg_size, n, np); /* ret = ar */
	tmp[idx] = ret[idx];

	int t = sub_msg_size * BITS_PER_WORD - 1; /* bit index of n_d */
	while (((exp[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 0 && t>0)
		t--;
	t--;

	while (t >= 0) {
		mp_montmul_dev(ret, ret, ret, limb_idx, idx, sub_msg_size, n, np);
	
		if (((exp[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 1) {
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, sub_msg_size, n, np);
		}
		t--;
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, sub_msg_size, n, np);

	output[msg_idx*sub_msg_size + idx] = ret[idx];

	/* Store timing in output[0] */
//	if(threadIdx.x == 0 && blockIdx.x == 0)
//		*(long long int*)output = clock64() - tick;
}


void gpu_private_decrypt_crt2(
	int msg_num, unsigned char* input, unsigned char * output) {
	int mem_bytes = msg_num * MSG_SIZE * sizeof(WORD);
	int msg_bytes = MSG_SIZE *	sizeof(WORD);
	int sub_msg_bytes = msg_bytes / 2;
	unsigned char *tmp = (unsigned char *)malloc(mem_bytes);
	unsigned char *tmp_, *inout_;
	BIGNUM *bn_input, *bn_pq;
	BN_CTX *bn_ctx;
	bn_input = BN_new();
	bn_pq = BN_new();
	bn_ctx = BN_CTX_new();

	for(int i=0; i<msg_num; i++) {
		BN_bin2bn(input+msg_bytes*i, msg_bytes, bn_input);
		assert(bn_input != NULL);
		BN_nnmod(bn_pq, bn_input, rsa_gpu->p, bn_ctx);
		mp_bn2mp((WORD *)(tmp+msg_bytes*i), bn_pq, MSG_SIZE/2);
		BN_nnmod(bn_pq, bn_input, rsa_gpu->q, bn_ctx);
		mp_bn2mp((WORD *)(tmp+msg_bytes*i+msg_bytes/2), bn_pq, MSG_SIZE/2);
	}

	/* Copy input into device memory */
	checkCudaErrors(cudaMemcpy(
		input_d, tmp, mem_bytes, cudaMemcpyHostToDevice));

	int grid_size = (msg_num * MSG_SIZE / 2 + BLK_SIZE -1) / BLK_SIZE * 2;
	for (int i=0; i<1; i++) { /* repeat decryption */
		gpu_modexp_crt2<<<grid_size, BLK_SIZE>>>(msg_num, MSG_SIZE, input_d, output_d);
	}

	/* Copy output into host memory */
	checkCudaErrors(cudaMemcpy(
		tmp, output_d, mem_bytes, cudaMemcpyDeviceToHost));

	/* Save timing info */
	//long long int time_info = *(long long int *)tmp;

	/* Little endian to big endian */
	for(int j=0; j<msg_num*2; j++) {
		tmp_ = tmp + j * sub_msg_bytes;
		inout_ = output + j * sub_msg_bytes;
		for(int i=0; i<sub_msg_bytes; i++) {
			inout_[i] = tmp_[sub_msg_bytes-1-i];
		}
	}
	/* CRT combine */
	BIGNUM *m1, *m2;
	m1 = BN_new();
	m2 = BN_new();
	for(int i=0; i<msg_num; i++) {
		inout_ = output + i * msg_bytes;
		BN_bin2bn(inout_, sub_msg_bytes, m1);
		BN_bin2bn(inout_+sub_msg_bytes, sub_msg_bytes, m2);
		BN_mod_sub(m1, m1, m2, rsa_gpu->p, bn_ctx);
		BN_mod_mul(m1, m1, rsa_gpu->iqmp, rsa_gpu->p, bn_ctx);
		BN_mul(m1, m1, rsa_gpu->q, bn_ctx);
		BN_add(m1, m1, m2);
		memset(inout_, 0, msg_bytes);
		/* Skip the zeros at the beginning of M1*/
		int zero_num = msg_bytes - BN_num_bytes(m1);
		BN_bn2bin(m1, inout_+zero_num);
	}

	/* Put timing info into output */
	//*(long long int *)output = time_info;

	/* Release memory */
	free(tmp);
	BN_free(bn_input);
	BN_free(bn_pq);
	BN_CTX_free(bn_ctx);
	BN_free(m1);
	BN_free(m2);
}

/* Timing attack without CRT */
__global__ void gpu_modexp_timing(
	int msg_num, int msg_size, WORD *input, uint64_t *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	/* Print smid */
	/*unsigned int smid;
	if(threadIdx.x == 0) {
		asm("mov.u32 %0, %%smid;" : "=r"(smid));
		printf("block id: %d, sm id: %d\n", blockIdx.x, smid);
	}*/
	long long int tick;
	if(threadIdx.x==0 && blockIdx.x==0)
		tick = clock64();
	__syncthreads();

	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];

	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = ar */
	tmp[idx] = ret[idx];

	int t = msg_size * BITS_PER_WORD - 1; /* bit index of d_d */
	while (((d_d[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 0 && t>0)
		t--;
	t--;

	while (t >= 0) {
		mp_montmul_dev(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
	
		if (((d_d[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 1) {
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
		}
		t--;
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);

	//output[msg_idx*msg_size + idx] = ret[idx];
	__syncthreads();
	if(threadIdx.x==0 && blockIdx.x==0)
		*output += (uint64_t)(clock64() - tick);
}


// Record reduction for every bit of key operation
// The reduction for first bit is 0
__global__ void gpu_modexp_reduction(
	int msg_num, int msg_size, WORD *input, uint64_t *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];

	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = ar */
	tmp[idx] = ret[idx];

	int t = msg_size * BITS_PER_WORD - 1; /* bit index of d_d */
	while (((d_d[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 0 && t>0)
		t--;
	t--;
    // reduction is stored as little endian: msb in the highest address, same as msg
    uint64_t *reductionS = output + msg_idx * msg_size;
    uint64_t *reductionM = output + msg_idx * msg_size + msg_num * msg_size;
	while (t >= 0) {
        int word = t / BITS_PER_WORD; 
        int bit = t % BITS_PER_WORD;
		mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0], reductionS + word, bit);
	
		if (((d_d[t/BITS_PER_WORD] >> (t%BITS_PER_WORD)) & 1) == 1) {
			mp_montmul_dev2(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0], reductionM + word, bit);
		}
		t--;
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
}

// Timing with CLSW
__global__ void gpu_modexp_timing_sw(
	int msg_num, int msg_size, WORD *input, uint64_t *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];

	long long int tick;
    __syncthreads();
	if (threadIdx.x == 0 && blockIdx.x == 0)
		tick = clock64();

	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;
	int num_frags = sw_d.num_fragments;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];

	mp_montmul_dev(tmp, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* tmp = ar */
	ar_pow[0][msg_idx][idx] = tmp[idx];
	
	mp_montmul_dev(ret, tmp, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = a*a*r */

	for (int i = 3; i <= sw_d.max_fragment; i += 2) {
		mp_montmul_dev(tmp, tmp, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
		ar_pow[i >> 1][msg_idx][idx] = tmp[idx];
	}

	ret[idx] = ar_pow[sw_d.fragment[num_frags - 1] >> 1][msg_idx][idx];
	
	for (int i = num_frags - 2; i >= 0; i--) {
		for (int k = sw_d.length[i]; k >= 1; k--)
			mp_montmul_dev(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
		if (sw_d.fragment[i]) {
			tmp[idx] = ar_pow[sw_d.fragment[i] >> 1][msg_idx][idx];
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
		}
	}

	/* ret = (a^e)*r; calculate a^e = montmul(ret, 1) */
	tmp[idx] = (idx==0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);

	//output[msg_idx*msg_size + idx] = ret[idx];
	__syncthreads();
	if (threadIdx.x == 0 && blockIdx.x == 0)
		*output = (uint64_t)(clock64() - tick);
}


/* Pre-processing input msg:
 * montgomeritize, and square 
 * assume the first bit of d is always 1.
 */
__global__ void gpu_preprocessing(
	int msg_num, int msg_size, WORD *input, WORD *output0, WORD *mont) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];

	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = ar */
	tmp[idx] = ret[idx];

	mp_montmul_dev(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0]); /* square */

	output0[msg_idx*msg_size + idx] = ret[idx];
	mont[msg_idx*msg_size + idx] = tmp[idx];
}

/* Generate ar_pow
 * Set output0 msges to be 1*r
 */
__global__ void gpu_preprocessing_sw(
	int msg_num, int msg_size, WORD *input, WORD *output) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = input[msg_idx * msg_size + idx];
	ret[idx] = r_sqr_d[idx];
	
	/* calculate ar_pow */
	mp_montmul_dev(tmp, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* tmp = ar */
	ar_pow[0][msg_idx][idx] = tmp[idx];
	
	mp_montmul_dev(ret, tmp, tmp, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = a*a*r */

	for (int i = 3; i <= sw_d.max_fragment; i += 2) {
		mp_montmul_dev(tmp, tmp, ret, limb_idx, idx, msg_size, n_d, np_d[0]);
		ar_pow[i >> 1][msg_idx][idx] = tmp[idx];
	}

	/* set msges0 to 1*r */
	tmp[idx] = (idx == 0);
	ret[idx] = r_sqr_d[idx];
	mp_montmul_dev(ret, tmp, ret, limb_idx, idx, msg_size, n_d, np_d[0]); /* ret = 1*r */
	output[msg_idx * msg_size + idx] = ret[idx];
}

/* one segment operation with reduction info
 * frag: odd value as it is, even value as 0 with length = value >> 1;
 */
__global__ void gpu_reduction_sw
	(int msg_num, int msg_size, WORD *msges0, uint16_t *reduction, uint16_t frag, char update) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	ret[idx] = msges0[msg_idx * msg_size + idx];
	
	int length = 5;
	if ((frag & 1)== 0) {
		length = frag >> 1;
	}
	else {
		tmp[idx] = ar_pow[frag >> 1][msg_idx][idx];
		mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	}

	for (int i = length; i >= 1; i--)
		mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0],
			reduction + msg_idx, i);
	
	if (update)
		msges0[msg_idx * msg_size + idx] = ret[idx];

}

/* Reduction calculation for vlnw
 * first guess length: if 16+minSquare >= length >= minSquare, if =, value is 1;
 * otherwise do length - minSquare # of squaring
 * second guess value:  multiply value, do minSquare square.
 */
__global__ void gpu_reduction_length
	(int msg_num, int msg_size, WORD *msges0, uint16_t *reduction, uint16_t length, int minSquare, char update) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	ret[idx] = msges0[msg_idx * msg_size + idx];

	if (length < minSquare) // not allowed
		return;
	if (length == minSquare) {
		// multiply by ar
		tmp[idx] = ar_pow[1 >> 1][msg_idx][idx];
		mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
		// square minSqaure times
		for (int i = 0; i < minSquare; i++)
			mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0],
				reduction + msg_idx, i);
	}
	else {
		// square length - minSqaure times
		for (int i = 0; i < length - minSquare; i++)
			mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0],
				reduction + msg_idx, i);
	}
	if (update)
		msges0[msg_idx * msg_size + idx] = ret[idx];

}

__global__ void gpu_reduction_value
	(int msg_num, int msg_size, WORD *msges0, uint16_t *reduction, uint16_t value, int minSquare, char update) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	ret[idx] = msges0[msg_idx * msg_size + idx];

	// multiply by ar^value
	tmp[idx] = ar_pow[value >> 1][msg_idx][idx];
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	// square minSqaure times
	for (int i = 0; i < minSquare; i++)
		mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0],
			reduction + msg_idx, i);

	if (update)
		msges0[msg_idx * msg_size + idx] = ret[idx];

}


/* one bit operation of d with reduction info */
__global__ void gpu_bit0_reduction(
	int msg_num, int msg_size, WORD *msges_mont, WORD *msges, uint16_t *reduction) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = msges_mont[msg_idx * msg_size + idx];
	ret[idx] = msges[msg_idx * msg_size + idx];
	mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0], 
		reduction + msg_idx, 0);
	msges[msg_idx*msg_size + idx] = ret[idx];
}

__global__ void gpu_bit1_reduction(
	int msg_num, int msg_size, WORD *msges_mont, WORD *msges, uint16_t *reduction) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = msges_mont[msg_idx * msg_size + idx];
	ret[idx] = msges[msg_idx * msg_size + idx];
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0], 
		reduction + msg_idx, 0);
	msges[msg_idx*msg_size + idx] = ret[idx];
}

__global__ void gpu_mbit_reduction(
	int msg_num, int msg_size, WORD *msges_mont, WORD *input_msges,
	int bit_size, int bit_value, uint16_t *reduction, bool update) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = msges_mont[msg_idx * msg_size + idx];
	ret[idx] = input_msges[msg_idx * msg_size + idx];
	for (int i = 0; i < bit_size; i++) {
		if ((bit_value >> (bit_size - i - 1)) & 0x01 )
			mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
		mp_montmul_dev2(ret, ret, ret, limb_idx, idx, msg_size, n_d, np_d[0], 
			reduction + msg_idx, i);
	}
	if (update)
		input_msges[msg_idx*msg_size + idx] = ret[idx];
}

/* De-mont: montmul(msg, msg_mont), montmul(msg, 1) */
__global__ void gpu_demont(
	int msg_num, int msg_size, WORD *msges_mont, WORD *msges) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = msges_mont[msg_idx * msg_size + idx];
	ret[idx] = msges[msg_idx * msg_size + idx];
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	tmp[idx] = (idx == 0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	msges[msg_idx*msg_size + idx] = ret[idx];
}

__global__ void gpu_demont_sw(
	int msg_num, int msg_size, WORD *msges, uint16_t frag) {
	__shared__ WORD _ret[BLK_SIZE];
	__shared__ WORD _tmp[BLK_SIZE];
	
	const int limb_idx = threadIdx.x / msg_size;
	const int msg_idx = blockIdx.x * blockDim.x / msg_size + limb_idx;
	const int idx = threadIdx.x % msg_size;

	if(msg_idx >= msg_num) return;

	WORD *ret = _ret + limb_idx * msg_size;
	WORD *tmp = _tmp + limb_idx * msg_size;

	tmp[idx] = ar_pow[frag >> 1][msg_idx][idx];
	ret[idx] = msges[msg_idx * msg_size + idx];
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	tmp[idx] = (idx == 0);
	mp_montmul_dev(ret, ret, tmp, limb_idx, idx, msg_size, n_d, np_d[0]);
	msges[msg_idx*msg_size + idx] = ret[idx];
}

