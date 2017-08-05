/* Class RSAGPUAttack Header file
 * Derived from RSAGPU and RSAAttack
 * Both RSAGPU and RSAAttack are derived from RSABase, they should be virtually derived from RSABase
 * 3/16/2017
 */

#ifndef RSAGPUATTACK_H
#define RSAGPUATTACK_H

#include "common.hh"
#include "RSAGPU.h"
#include <stdint.h>

class RSAGPUAttack : public RSAGPU{
public:
	RSAGPUAttack(int traceNum_ = 1000,
				 int traceSize_ = 1,
				 int seed_ = 0,
				 SW_Type swType = SW_Type::clnw,
				 const char *fileName = "private_key.pem",
				 int keyBits_ = KEY_BITS);
	~RSAGPUAttack();
	void recordTime(const char *timeFileName = "data_gpu.bin");
	void recordReduction(const char *reductionFileName = "reduction_gpu.bin");
	void timingAttack(const char *fileName = "data_gpu.bin");
protected:
	// Generate traceNum msg on device memory deviceMsg
	void genRandMsg();

	int traceNum;
	int traceSize; // msg num for each trace
	int seed;
	WORD *deviceMsg;
	uint64_t *deviceTime; 
	uint64_t *hostTime; 
};

#endif
