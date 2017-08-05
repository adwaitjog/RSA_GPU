/* RSABase test
 * Test RSABase and RSAAttack
 * 3/10/2017
 */
#include <iostream>
using namespace std;

#include <stdio.h>
#include <openssl/bn.h>
#include <openssl/rsa.h>

#include "RSAAttack.h"
#include "RSABase.h"
#include "rsa_cpu.hh"
#include "RSAGPU.h"
#include "RSAGPUAttack.h"

int main () {
    //Key key;
    //key.saveSW();
	RSABase rsaBase;
	int keyBytes = rsaBase.getKeyBytes();
	//cout << rsaBase.getKeyBytes() << endl;
	unsigned char *ptext = new unsigned char[keyBytes];
	unsigned char *ctext = new unsigned char[keyBytes];
	unsigned char *pptext = new unsigned char[keyBytes];
	
	set_random(ptext, keyBytes);
	ptext[0] = 0;
	rsaBase.publicEncryptCPU_SSL(ptext, ctext);
	// Print ptext ctext
	cout << "plaint text" << endl;
	for (int i = 0; i < keyBytes; i++) {
		printf("%02x", ptext[i]);
	}
	cout << endl;
	cout << "cipher text" << endl;
	for (int i = 0; i < keyBytes; i++) {
		printf("%02x", ctext[i]);
	}
	cout << endl;

	// CPU decrypt
	cout << "CPU decrypt" << endl;
	rsaBase.privateDecryptCPU(ctext, pptext, SW_Type::vlnw);
	for (int i = 0; i < keyBytes; i++) {
		printf("%02x", pptext[i]);
	}
	cout << endl;
	// CPU CRT
	cout << "CPU CRT decrypt" << endl;
	rsaBase.privateDecryptCRT(ctext, pptext);
	for (int i = 0; i < keyBytes; i++) {
		printf("%02x", pptext[i]);
	}
	cout << endl;

    // Test RSAGPU
    RSAGPU rsaGPU(SW_Type::vlnw);
	cout << "GPU decrypt" << endl;
    rsaGPU.privateDecryptBatch(ctext, pptext, 1);
    // Print result
	for (int i = 0; i < keyBytes; i++) {
		printf("%02x", pptext[i]);
	}
	cout << endl;

	// test c % p = (c * Rp^-1 % N ) * Rp % p
	/*RSA *rsa = key.getRSA();
	int keyBits = key.getKeyBits();
	BIGNUM *c = BN_new();
	BIGNUM *m1 = BN_new();
	BIGNUM *m2 = BN_new();
	BIGNUM *Rp = BN_new();
	BIGNUM *iRp = BN_new();
	BN_CTX *bn_ctx = BN_CTX_new();
	// Get Rp, iRp
	BN_set_bit(Rp, keyBits/2);
	BN_mod_inverse(iRp, Rp, rsa->n, bn_ctx);
	// Generate random num
	BN_rand(c, keyBits, -1, 0);
	// m1 = c % p
	BN_nnmod(m1, c, rsa->p, bn_ctx);
	// m2 = c * iRp % N * Rp % p
	BN_mod_mul(m2, c, iRp, rsa->n, bn_ctx);
	BN_mod_mul(m2, m2, Rp, rsa->p, bn_ctx);
	BN_print_fp(stdout, m1);
	printf("\n");
	BN_print_fp(stdout, m2);
	printf("\n");

	// Free memory
	BN_free(c);
	BN_free(m1);
	BN_free(m2);
	BN_free(Rp);
	BN_free(iRp);
	BN_CTX_free(bn_ctx);*/
    
    // Deallocate memory
	delete [] ptext;
	delete [] ctext;
	delete [] pptext;
}
