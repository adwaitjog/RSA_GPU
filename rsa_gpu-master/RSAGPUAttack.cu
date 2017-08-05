/* class RSAGPUAttack implementation in CUDA
 * 3/16/2017
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdint.h>
#include <curand.h>
#include <fstream>
#include <stdexcept>
#include <iostream>
using namespace std;

#include "RSAGPUAttack.h"
#include "rsa_gpu.hh"
#include "tanc.hh"

#define checkCurandErrors(x) do { curandStatus_t status= (x);\
	if (status != CURAND_STATUS_SUCCESS) { \
	printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); \
	exit(EXIT_FAILURE);}} while(0)

RSAGPUAttack::RSAGPUAttack(
	int traceNum_, 
	int traceSize_, 
	int seed_, 
	SW_Type swType,
	const char *fileName, 
	int keyBits_)
	:RSAGPU(swType, KEY_BITS / 8, fileName, keyBits_), traceNum(traceNum_), traceSize(traceSize_), seed(seed_) {
	// Allocated device memory for traceNum * traceSize of msg
	checkCudaErrors(cudaMalloc(&deviceMsg, traceNum * traceSize * keyBytes));
	// Device memory for time info and reset to 0
	checkCudaErrors(cudaMalloc(&deviceTime, traceNum * sizeof(uint64_t)));
	checkCudaErrors(cudaMemset(deviceTime, 0, traceNum * sizeof(int64_t)));
	// Host memory for time info and reduction
	hostTime = new uint64_t[traceNum];
}

RSAGPUAttack::~RSAGPUAttack() {
	// Free device and host memory
	checkCudaErrors(cudaFree(deviceMsg));
	checkCudaErrors(cudaFree(deviceTime));
	delete [] hostTime;
}

void RSAGPUAttack::genRandMsg() {
	// Generate rand msg
	curandGenerator_t gen;
	checkCurandErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, seed));
	checkCurandErrors(curandGenerate(
		gen, (unsigned int *)deviceMsg, traceNum * traceSize * keyBytes / sizeof(unsigned int)));
	checkCurandErrors(curandDestroyGenerator(gen));
	checkCudaErrors(cudaDeviceSynchronize());
}

// Record the running time of decryption of msges 
void RSAGPUAttack::recordTime(const char* timeFileName) {
	// Time the decryption
	int gridSize = (traceSize * keySize + BLK_SIZE -1) / BLK_SIZE;
	cout << "record time grid size " << gridSize << endl; 
	for (int i = 0; i < traceNum; i++) {
		switch (swType) {
		case (SW_Type::none):
			gpu_modexp_timing<<<gridSize, BLK_SIZE>>>
				(traceSize, keySize, deviceMsg + i * keySize * traceSize, deviceTime + i);
			break;
		case (SW_Type::clnw):
		case (SW_Type::vlnw):
			gpu_modexp_timing_sw<<<gridSize, BLK_SIZE>>>
				(traceSize, keySize, deviceMsg + i * keySize * traceSize, deviceTime + i);
			break;
		}
		// single time copy to avoid kernel overlapping
		checkCudaErrors(cudaMemcpy(hostTime + i, deviceTime + i, sizeof(uint64_t), cudaMemcpyDeviceToHost));
	}
	// Save timing into file
	ofstream timeFile(timeFileName, ios::binary | ios::out);
	if (!timeFile) throw runtime_error("Could not open time file");
	timeFile.write((const char*)hostTime, traceNum * sizeof(uint64_t));
	timeFile.close();
}


// Record the reductions of decryption of msges
// The size of reduction is the same as msg size and key size for binary MP
void RSAGPUAttack::recordReduction(const char * reductionFileName) {
	// Device memory for reduction, for each msg, there will be KEY_BITS reductions, needing KEY_BITS bits memory
	// Double the size of reduction by including both Square and Multiplication
	uint64_t* deviceReduction;
	checkCudaErrors(cudaMalloc(&deviceReduction, traceNum * traceSize * keyBytes * 2));
	checkCudaErrors(cudaMemset(deviceReduction, 0, traceNum * traceSize * keyBytes * 2));
	uint64_t* hostReduction = new uint64_t[traceNum * traceSize * keySize * 2];

	int gridSize = (BLK_SIZE - 1 + traceNum * traceSize * keySize) / BLK_SIZE;
	gpu_modexp_reduction<<<gridSize, BLK_SIZE>>>
		(traceNum * traceSize, keySize, deviceMsg, deviceReduction);
	checkCudaErrors(cudaMemcpy
		(hostReduction, deviceReduction, traceNum * traceSize * keyBytes * 2, cudaMemcpyDeviceToHost));
	// Save timing into file
	ofstream reductionFile(reductionFileName, ios::binary | ios::out);
	if (!reductionFile) throw runtime_error("Could not open time file");
	reductionFile.write((const char*)hostReduction, traceNum * traceSize * keyBytes * 2);
	reductionFile.close();

	checkCudaErrors(cudaFree(deviceReduction));
	delete [] hostReduction;
}

void RSAGPUAttack::timingAttack(const char *fileName) {
    // Load time info from fileName
    uint64_t *timeInfo = new uint64_t[traceNum];
    ifstream dataFile(fileName, ios::binary | ios::in);
    if (!dataFile) throw runtime_error("Could not open file for timming Attack.");
    dataFile.read((char *)timeInfo, traceNum * sizeof(uint64_t));

	WORD *msges0, *msges1, *msges_mont; // device memory for intermediate result
	checkCudaErrors(cudaMalloc(&msges0, traceNum * traceSize * keyBytes));
	int gridSize = (traceNum * traceSize * keySize + BLK_SIZE - 1) / BLK_SIZE;
	gpu_preprocessing_sw<<<gridSize, BLK_SIZE>>>
		(traceNum * traceSize, keySize, deviceMsg, msges0);
	// Attack
	switch (swType) {
	case (SW_Type::clnw):
		gpu_preprocessing_sw<<<gridSize, BLK_SIZE>>>
			(traceNum * traceSize, keySize, deviceMsg, msges0);
		dt_attack_sw(keySize, traceSize, traceNum, timeInfo, key.clnw, msges0, 128);
		break;
	case (SW_Type::vlnw):
		gpu_preprocessing_sw<<<gridSize, BLK_SIZE>>>
			(traceNum * traceSize, keySize, deviceMsg, msges0);
		dt_attack_vlnw(keySize, traceSize, traceNum, timeInfo, key.vlnw, msges0, 158);
		break;
	case (SW_Type::none):
		checkCudaErrors(cudaMalloc(&msges1, traceNum * traceSize * keyBytes));
		checkCudaErrors(cudaMalloc(&msges_mont, traceNum * traceSize * keyBytes));
		gpu_preprocessing<<<gridSize, BLK_SIZE>>>
			(traceSize * traceNum, keySize, deviceMsg, msges0, msges_mont);
		int bit_size = 1;
		int bit_num = 508;
		dt_attack(keySize, traceSize, traceNum, bit_size, bit_num, timeInfo,
			key.d, msges0, msges1, msges_mont);
		checkCudaErrors(cudaFree(msges1));
		checkCudaErrors(cudaFree(msges_mont));
		break;
	}
		
    // Free memory
    delete [] timeInfo;
	checkCudaErrors(cudaFree(msges0));
}
