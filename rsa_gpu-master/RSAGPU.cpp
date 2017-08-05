/* Class RSAGPU implementaion
 * 3/15/2017
 */

#include "common.hh"
#include "RSAGPU.h"
#include "rsa_gpu.hh"
#include <stdexcept>
using namespace std;

RSAGPU::RSAGPU(SW_Type swType_, int devMemBytes_, const char *fileName, int keyBits_)
	:RSABase(fileName, keyBits_), devMemBytes(devMemBytes_), swType(swType_) {
	switch (swType) {
	case (SW_Type::none):
		gpu_setup(keySize, (WORD *)key.d, (WORD *)key.n, (WORD *)key.np, (WORD *)key.r_sqr, devMemBytes,
		(WORD *)key.pd, (WORD *)key.qd, (WORD *)key.p, (WORD *)key.q, (WORD *)key.pp, (WORD *)key.qp, 
		(WORD *)key.pr_sqr, (WORD *)key.qr_sqr, (WORD *)key.iqmp, &key.rsa);
		break;
	case (SW_Type::clnw):
		gpu_setup(keySize, (WORD *)key.d, (WORD *)key.n, (WORD *)key.np, (WORD *)key.r_sqr,	devMemBytes, 
		(WORD *)key.pd, (WORD *)key.qd, (WORD *)key.p, (WORD *)key.q, (WORD *)key.pp, (WORD *)key.qp, 
		(WORD *)key.pr_sqr, (WORD *)key.qr_sqr, (WORD *)key.iqmp, &key.rsa, &key.clnw);
		break;
	case (SW_Type::vlnw):
		gpu_setup(keySize, (WORD *)key.d, (WORD *)key.n, (WORD *)key.np, (WORD *)key.r_sqr,	devMemBytes, 
		(WORD *)key.pd, (WORD *)key.qd, (WORD *)key.p, (WORD *)key.q, (WORD *)key.pp, (WORD *)key.qp, 
		(WORD *)key.pr_sqr, (WORD *)key.qr_sqr, (WORD *)key.iqmp, &key.rsa, &key.vlnw);
		break;
	}
}

RSAGPU::~RSAGPU() {
	gpu_reset();
}

void RSAGPU::privateDecryptBatch(const unsigned char *ctext, unsigned char *ptext, int msgNum) {
	// the memory size must be smaller than devMemBytes
	if (msgNum * keyBytes > devMemBytes)
		throw runtime_error("Not enough device memory");
	gpu_private_decrypt(msgNum, ctext, ptext, swType);
}
