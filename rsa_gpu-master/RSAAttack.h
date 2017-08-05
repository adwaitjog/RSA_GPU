/* Class RSAAttack headfile
 * Derived from class RSABase
 * 3/10/2017
 */

#ifndef RSAATTACK_H
#define RSAATTACK_H

#include <stdint.h>

#include "common.hh"
#include "RSAGPUAttack.h"


class RSAAttack : public RSAGPUAttack {
public:
	RSAAttack(int traceNum_ = 1000, int seed_ = 0, SW_Type swType = SW_Type::clnw, 
		const char *fileName = "private_key.pem", int keyBits_ = KEY_BITS);
	~RSAAttack();
	// Record private decryption time
	void timeDecrypt(const char *fileName = "data_cpu.bin");
	// Attack RSA on CPU using GPU
	//void timingAttack(const char *fileName = "data_cpu.bin", SW_Type swType = SW_Type::clnw);
	// Timing attack CRT with chosen input 
	void attackCRT(int windowSize = 32);
	// Timing CRT with random input
	void timeCRT(const char *fileName = "data_crt.bin");
	// Timing attack CRT with random input
	void attackCRTRandom(const char *fileName = "data_crt.bin");
protected:
	WORD *hostMsg; // host msg
};

#endif
