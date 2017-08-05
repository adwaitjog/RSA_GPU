/* Class RSABase CPU code
 * 3/6/2017
 */
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace std;
#include <openssl/rsa.h>
#include <openssl/bn.h>
#include <assert.h>

#include "common.hh"
#include "Key.h"
#include "RSABase.h"
#include "rsa_cpu.hh"

// Default constructor
RSABase::RSABase(const char *fileName, int keyBits_) 
	: key(fileName, keyBits_), keyBits(keyBits_){
	keyBytes = keyBits_ / 8;
	keySize = keyBytes / sizeof(WORD);
}

// Switch endianness of msges
void RSABase::msgEndiannessSwitch(unsigned char *text, int msgNum, int msgBytes) {
	unsigned char *msg;
	unsigned char tmp;
	for (int i = 0; i < msgNum; i++) {
		msg = text + i * msgBytes;
		for (int j = 0; j < msgBytes/2; j++) {
			tmp = msg[j];
			msg[j] = msg[msgBytes - 1 -j];
			msg[msgBytes - 1 - j] = tmp;
		}
	}
}

int RSABase::publicEncryptCPU_SSL(const unsigned char *ptext, unsigned char *ctext) {
	return RSA_public_encrypt(keyBytes, ptext, ctext, &key.rsa, RSA_NO_PADDING);
};

int RSABase::privateDecryptCPU_SSL(const unsigned char *ctext, unsigned char *ptext) {
	return RSA_private_decrypt(keyBytes, ctext, ptext, &key.rsa, RSA_NO_PADDING);
};

void RSABase::privateDecryptCPU(const unsigned char *ctext, unsigned char *ptext, SW_Type swType) {
	unsigned char *temp = new unsigned char[keyBytes];
	// Copy ctext to temp
	for (int i = 0; i < keyBytes; i++) {
		temp[i] = ctext[i];
	}
	// Switch endianness
	msgEndiannessSwitch(temp, 1, keyBytes);
	// Typecast
	WORD *tempWord = (WORD *)temp;
	// Exponentiataion
	msgMontExpCPU(tempWord, tempWord, swType);
	// Switch endianness
	msgEndiannessSwitch(temp, 1, keyBytes);
	// Copy temp to ptext
	for (int i = 0; i < keyBytes; i++) {
		ptext[i] = temp[i];
	}
	delete [] temp;
}

int RSABase::getKeyBytes() {
	return keyBytes;
}


// Return most significant 64 bit 
static WORD umulhi(WORD a, WORD b) {
#define UMUL_HI_ASM(a,b)   ({      \
	register BN_ULONG ret,discard;  \
	asm ("mulq      %3"             \
		      : "=a"(discard),"=d"(ret)  \
		      : "a"(a), "g"(b)           \
		      : "cc");                   \
	ret;})

	return UMUL_HI_ASM(a, b);
}
// Return least significant 64 bit
static WORD umullo(WORD a, WORD b)
{
	return a * b;
}
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

// CPU multiplication of two message
static void msgMulCPU(WORD *ret, const WORD *a, const WORD *b, int msgSize) {
	WORD t[KEY_SIZE * 2];
	WORD c[KEY_SIZE * 2];	// carry

	for (int i = 0; i < msgSize; i++) {
		c[i] = 0;
		c[i + msgSize] = 0;
		t[i] = 0;
		t[i + msgSize] = 0;
	}

	for (int j = 0; j < msgSize; j++) {
		for (int i = 0; i < msgSize; i++) {
			WORD hi = umulhi(a[i], b[j]);
			WORD lo = umullo(a[i], b[j]);
		
			ADD_CARRY(c[i + j + 2], t[i + j + 1], t[i + j + 1], hi);
			ADD_CARRY(c[i + j + 1], t[i + j], t[i + j], lo);
		}
	}

	while (1) {
		bool all_zero = true;
		for (int j = 0; j < msgSize; j++) {
			if (c[j])
				all_zero = false;
			if (c[j + msgSize])
				all_zero = false;
		}
		if (all_zero)
			break;

		for (int j = 2 * msgSize - 1; j >= 1; j--)
			ADD_CARRY_CLEAR(c[j + 1], t[j], t[j], c[j]);
	}

	for (int i = 0; i < msgSize; i++) {
		ret[i] = t[i];
		ret[i + msgSize] = t[i + msgSize];
	}
}

// CPU subtraction of two message, , return 1 for the most significant borrow
static int msgSubCPU(WORD *ret, const WORD *a, const WORD *b, int msgSize) {
	WORD brw[KEY_SIZE]; // borrow

	for (int i = 0; i < msgSize; i++) {
		brw[i] = 0;
		SUB_BORROW(brw[i], ret[i], a[i], b[i]);
	}

	while (1) {
		bool all_zero = true;
		/* NOTE msgSize - 1, not just msgSize */ 
		for (int j = 0; j < msgSize - 1; j++) { 
			if (brw[j])
				all_zero = false;
		}

		if (all_zero)
			break;

		for (int j = msgSize - 2; j >= 0; j--)
			SUB_BORROW_CLEAR(brw[j + 1], ret[j + 1], ret[j + 1], brw[j]);
	}

	return brw[msgSize - 1];
}


// CPU montgomery multiplication of two message
static void msgMontMulCPU(WORD *ret, const WORD *a, const WORD *b, 
    const WORD *n, const WORD *np, int msgSize) {
	WORD t[KEY_SIZE * 2];
	WORD c[KEY_SIZE * 2];
	WORD u[KEY_SIZE];
	int carry = 0;

	//WORD *n = key.n;
	//WORD *np = key.np;
	
	for (int i = 0; i < msgSize; i++) {
		c[i] = 0;
		c[i + msgSize] = 0;
	}

	msgMulCPU(t, a, b, msgSize);
		
	for (int j = 0; j < msgSize; j++) {
		WORD m = t[j] * np[0];
		for (int i = 0; i < msgSize; i++) {
			WORD hi = umulhi(m, n[i]);
			WORD lo = umullo(m, n[i]);

			ADD_CARRY(c[i + j + 1], t[i + j + 1], t[i + j + 1], hi);
			ADD_CARRY(c[i + j], t[i + j], t[i + j], lo);
		}
		
		while (1) {
			bool all_zero = true;
			for (int j = 0; j < msgSize; j++) {
				if (c[j])
					all_zero = false;
				if (j < msgSize - 1 && c[j + msgSize])
					all_zero = false;
			}
			if (all_zero)
				break;

			for (int j = 2 * msgSize - 1; j >= 1; j--)
				ADD_CARRY_CLEAR(c[j], t[j], t[j], c[j - 1]);
		}
	}

	for (int i = 0; i < msgSize; i++)
		u[i] = t[i + msgSize];

	//carry = mp_add_cpu(u, t + msgSize, mn + msgSize);
	carry = c[2 * msgSize - 1];

	// c may be 0 or 1, but not 2
	if (carry)	
		goto u_is_bigger;

	/* Ugly, but practical. 
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = msgSize - 1; i >= 0; i--) {
		if (u[i] > n[i])
			goto u_is_bigger;
		if (u[i] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	msgSubCPU(ret, u, n, msgSize);
	return;

n_is_bigger:
	for (int i = 0; i < msgSize; i++)
		ret[i] = u[i];
	return;
}

// CPU montgomery exponentiation
void RSABase::msgMontExpCPU(WORD *ret, const WORD *ar, SW_Type swType) {
	//struct SW &sw = key.sw;
	//struct SW &sw = key.vlnw;
    struct SW sw;
    if (swType == SW_Type::clnw)
        sw = key.clnw;
    else if (swType == SW_Type::vlnw)
        sw = key.vlnw;
	else
		throw runtime_error("SW_Type not supported");
	WORD ar_sqr[KEY_SIZE];

	/* odd powers of ar (ar, (ar)^3, (ar)^5, ... ) */
	WORD ar_pow[SW_MAX_FRAGMENT / 2][KEY_SIZE];

	for (int i = 0; i < KEY_SIZE; i++)
		ar_pow[0][i] = ar[i];
	
	// Montgomerization of ar_pow[0]
	msgMontMulCPU(ar_pow[0], ar_pow[0], key.r_sqr, key.n, key.np, keySize);

	msgMontMulCPU(ar_sqr, ar_pow[0], ar_pow[0], key.n, key.np, keySize);

	for (int i = 3; i <= sw.max_fragment; i += 2)
		msgMontMulCPU(ar_pow[i >> 1], ar_pow[(i >> 1) - 1], ar_sqr, key.n, key.np, keySize);

	for (int i = 0; i < KEY_SIZE; i++)
		ret[i] = ar_pow[sw.fragment[sw.num_fragments - 1] >> 1][i];

	for (int k = sw.num_fragments - 2; k >= 0; k--) {
		for (int i = 0; i < sw.length[k]; i++)
			msgMontMulCPU(ret, ret, ret, key.n, key.np, keySize);

		if (sw.fragment[k])
			msgMontMulCPU(ret, ret, ar_pow[sw.fragment[k] >> 1], key.n, key.np, keySize);
	}

	// Demontgomerization of ret
	WORD one[KEY_SIZE];
	memset(one, 0, KEY_SIZE * sizeof(WORD));
	one[0] = 1;
	msgMontMulCPU(ret, ret, one, key.n, key.np, keySize);
}

// Exp using CRT without sliding window, ar will be montgomerized.
void RSABase::msgMontExpCRT(WORD *ret, WORD *ar, bool pq) {
    const WORD *e;
	const WORD *n;
	const WORD *np;
	const WORD *r_sqr;

    if (pq) { 
    	e = key.pd;
		n = key.p;
		np = key.pp;
		r_sqr = key.pr_sqr;
	}
	else {
		e = key.qd;
		n = key.q;
		np = key.qp;
		r_sqr = key.qr_sqr;
	}

    int t = keyBytes / 2 * 8 - 1; // MSB index of exponent
	int msgSize = keySize / 2;

	// Montgomerization
	msgMontMulCPU(ar, ar, r_sqr, n, np, msgSize);

	while (((e[t / BITS_PER_WORD] >> (t % BITS_PER_WORD)) & 1) == 0 && t > 0)
		t--;

	for (int i = 0; i < msgSize; i++)
		ret[i] = ar[i];

	t--;

	while (t >= 0) {
		msgMontMulCPU(ret, ret, ret, n, np, msgSize);
		if (((e[t / BITS_PER_WORD] >> (t % BITS_PER_WORD)) & 1) == 1)
			msgMontMulCPU(ret, ret, ar, n, np, msgSize);

		t--;
	}

	// Demontgomeriztion
	WORD one[keySize];
	memset(one, 0, sizeof(WORD) * keySize);
	one[0] = 1;
	msgMontMulCPU(ret, ret, one, n, np, msgSize);
}

void RSABase::privateDecryptCRT(const unsigned char *ctext, unsigned char *ptext) {
	int msgBytes = keyBytes / 2;
	int msgSize = keySize / 2;
	WORD *msgp = new WORD[msgSize];
	WORD *msgq = new WORD[msgSize];
	WORD *msgp_ret = new WORD[msgSize];
	WORD *msgq_ret = new WORD[msgSize];

	// Convert ctext into BN, split it into two shorter msg, convert it backinto char array
	BIGNUM *bn_input, *bn_pq;
	BN_CTX *bn_ctx;
	bn_input = BN_new();
	bn_pq = BN_new();
	bn_ctx = BN_CTX_new();
	
	BN_bin2bn(ctext, keyBytes, bn_input);
	assert(bn_input != NULL);
	BN_nnmod(bn_pq, bn_input, key.rsa.p, bn_ctx);
	mp_bn2mp(msgp, bn_pq, msgSize);
	BN_nnmod(bn_pq, bn_input, key.rsa.q, bn_ctx);
	mp_bn2mp(msgq, bn_pq, msgSize);
	
	// Exponentiate msgp and msgq
	msgMontExpCRT(msgp_ret, msgp, true);
	msgMontExpCRT(msgq_ret, msgq, false);

	// Combine two msges
	msgEndiannessSwitch((unsigned char *)msgp_ret, 1, msgBytes);
	msgEndiannessSwitch((unsigned char *)msgq_ret, 1, msgBytes);

	BIGNUM *m1, *m2;
	m1 = BN_new();
	m2 = BN_new();

	BN_bin2bn((unsigned char*)msgp_ret, msgBytes, m1);
	BN_bin2bn((unsigned char*)msgq_ret, msgBytes, m2);
	BN_mod_sub(m1, m1, m2, key.rsa.p, bn_ctx);
	BN_mod_mul(m1, m1, key.rsa.iqmp, key.rsa.p, bn_ctx);
	BN_mul(m1, m1, key.rsa.q, bn_ctx);
	BN_add(m1, m1, m2);
	// Skip the zeros at the beginning of m1
	int zeroNum = keyBytes - BN_num_bytes(m1);
	memset(ptext, 0, keyBytes);
	BN_bn2bin(m1, ptext + zeroNum);

	// Free memory
	BN_free(bn_input);
	BN_free(bn_pq);
	BN_CTX_free(bn_ctx);
	delete [] msgp;
	delete [] msgq;
	delete [] msgp_ret;
	delete [] msgq_ret;
	BN_free(m1);
	BN_free(m2);
}
