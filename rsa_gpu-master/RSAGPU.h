/* Class RSAGPU header file
 * RSA operation on GPU
 * 3/15/2017
 */
#ifndef RSAGPU_H
#define RSAGPU_H

#include "RSABase.h"

class RSAGPU : public RSABase {
public:
	RSAGPU(SW_Type swType = SW_Type::clnw,
		int devMemBytes_ = KEY_BITS / 8, const char *fileName = "private_key.pem", int keyBits_ = KEY_BITS);
    void privateDecryptBatch(const unsigned char *ctext, unsigned char *ptext, int msgNum);
	~RSAGPU();
protected:
	// device memory managed by gpu_setup and gpu_reset in rsa_gpu.cu
    int devMemBytes;
	SW_Type swType;
};

#endif
