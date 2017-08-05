/* Class Key head file
 * 3/6/2017
 */
#ifndef KEY_H
#define KEY_H

#include <openssl/rsa.h>

#include "common.hh"

class Key {
	friend class RSABase;
    friend class RSAGPU;
	friend class RSAAttack;
	friend class RSAGPUAttack;
public:
	// Static functions for generating and reading key file
	static void genRSAKey(const char *fileName, int keyBits_);
	static RSA *readKeyFile(const char *fileName);

	Key(const char *fileName = "private_key.pem", int keyBits_ = KEY_BITS);
	void saveSW(const char *fileName = "SW.txt");
	int getKeyBits();
	RSA *getRSA();
private:
	void keySetup(RSA *rsa);

	int keyBits;
	int keyBytes;
	int keySize;
	RSA rsa;
	WORD e[KEY_SIZE];
	WORD d[KEY_SIZE];
	WORD n[KEY_SIZE];
	WORD np[KEY_SIZE];
	WORD r_sqr[KEY_SIZE];

	WORD pd[KEY_SIZE/2];
	WORD qd[KEY_SIZE/2];
	WORD p[KEY_SIZE/2];
	WORD q[KEY_SIZE/2];
	WORD pp[KEY_SIZE/2];
	WORD qp[KEY_SIZE/2];
	WORD pr_sqr[KEY_SIZE/2];
	WORD qr_sqr[KEY_SIZE/2];
	WORD iqmp[KEY_SIZE/2];
	struct SW clnw; // constant length non-zero window
	struct SW vlnw; // variable length non-zero window
};

#endif
