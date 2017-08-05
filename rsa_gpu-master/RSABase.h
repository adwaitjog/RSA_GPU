/* Class RSABase head file
 * 3/6/2017
 */

#ifndef RSABASE_H
#define RSABASE_H
#include <iostream>
#include "common.hh"
#include "Key.h"


class RSABase {
public:
	RSABase(const char *fileName = "private_key.pem", int keyBits_ = KEY_BITS);
	// CPU encrypt using openssl
	int publicEncryptCPU_SSL(const unsigned char *ptext, unsigned char *ctext);
	// CPU decrypt using openssl
	int privateDecryptCPU_SSL(const unsigned char *ctext, unsigned char *ptext);
	// CPU decrypt using own code
	void privateDecryptCPU(const unsigned char *ctext, unsigned char *ptext, SW_Type swType = SW_Type::clnw);
	// Using CRT
	void privateDecryptCRT(const unsigned char *ctext, unsigned char *ptext);

	static void msgEndiannessSwitch(unsigned char *text, int msgNum, int msgBytes);
	int getKeyBytes();
	// CPU montgomery exponentiation of key.d
	void msgMontExpCPU(WORD *ret, const WORD *ar, SW_Type swType = SW_Type::clnw);
	void msgMontExpCRT(WORD *ret, WORD *ar, bool pq);
protected:
	Key key;
	int keyBits;
	int keyBytes; // keyBits_ / 8
	int keySize; // keyBytes / sizeof(WORD);
};

#endif
