/* Class Key CPU code
 * 3/6/2017
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <cstring>
using namespace std;

#include <openssl/rsa.h>
#include <openssl/pem.h>

#include "Key.h"

/*Convert char to binary string
 * binstr should be at least 9 bytes long
 */
static unsigned char *char2binstr(unsigned char *binstr, unsigned char c) {
	for(int i=0; i<8; i++){
		binstr[i] = (c & (0x80>>i)) ? '1' : '0';
	}
	binstr[8]=0;
	return binstr;
}

/* Convert BIGNUM into WORD array */
static void bn2word(WORD *a, const BIGNUM *bn, int word_len) {
	assert(word_len * (int)sizeof(WORD) >= BN_num_bytes(bn));
	memset(a, 0, sizeof(WORD) * word_len);
	memcpy(a, bn->d, BN_num_bytes(bn));
}

static int getBit(const WORD *a, int i)
{
	return (a[i / BITS_PER_WORD] >> (i % BITS_PER_WORD)) & 1;
}

static void getVLNW(struct SW *ret, const WORD *a, int word_len){
    const int num_bits = word_len * BITS_PER_WORD;
	int d;
	int q;
	int r;
	int l = 0;

	int n = 0;
	int i = 0;

	if (num_bits < 512)
		d = 4;
	else if (num_bits < 1024)
		d = 5;
	else
		d = 6;

	if (512 <= num_bits && num_bits < 1024)
		q = 3;
	else
		q = 2;

	r = d - 1;
	while (r > q) {
		l++;
		r -= q;
	}

	memset(ret, 0, sizeof(*ret));

	while (i < num_bits) {
		if (getBit(a, i) == 0) {
			ret->fragment[n] = 0;
			while (i < num_bits && getBit(a, i) == 0) {
				i++;
				ret->length[n]++;
				assert(ret->length[n] > 0);
			}

			n++;
			assert(n <= SW_MAX_NUM_FRAGS);
		}

		if (i >= num_bits)
			break;

		if (getBit(a, i) == 0)
			continue;

		ret->fragment[n] = 1;
		ret->length[n] = 1;

		int j = i + 1;

		while (j < i + d && j < num_bits) {
			bool allzero = true;
			int k;

			if (j - i + 1 > r) {
				// check incoming q bits
				for (k = j; k < j + q && k < i + d; k++) {
					if (getBit(a, k)) {
						ret->fragment[n] |= (1 << (k - i));
						allzero = false;
					}
				}
			} else {
				// last step: check incoming r bits
				for (k = j; k < j + r && k < i + d; k++) {
					if (getBit(a, k)) {
						ret->fragment[n] |= (1 << (k - i));
						allzero = false;
					}
				}
			}

			if (allzero)
				break;

			ret->length[n] += (k - j);
			j = k;
		}

		if (ret->fragment[n] > ret->max_fragment)
			ret->max_fragment = ret->fragment[n];
		i = j;
		n++;
		assert(n <= SW_MAX_NUM_FRAGS);
	}

	while (n > 0 && ret->fragment[n - 1] == 0)
		n--;

	assert(n > 0);
	assert(ret->max_fragment < SW_MAX_FRAGMENT);
	assert(ret->max_fragment % 2 == 1);
	ret->num_fragments = n;
}


static void getCLNW(struct SW *ret, const WORD *a, int word_len)
{
	const int num_bits = word_len * BITS_PER_WORD;
	int d;

	int n = 0;
	int i = 0;

	if (num_bits < 256)
		d = 4;
	else if (num_bits < 768)
		d = 5;
	else if (num_bits < 1792)
		d = 6;
	else
		d = 7;

	memset(ret, 0, sizeof(*ret));

	while (i < num_bits) {
		if (getBit(a, i) == 0) {
			ret->fragment[n] = 0;
			while (i < num_bits && getBit(a, i) == 0) {
				i++;
				ret->length[n]++;
				assert(ret->length[n] > 0);
			}

			n++;
			assert(n <= SW_MAX_NUM_FRAGS);
		}

		int j;

		for (j = i; j < i + d && j < num_bits; j++) {
			ret->fragment[n] |= (getBit(a, j) << (j - i));
			ret->length[n]++;
			assert(ret->length[n] > 0);
		}

		if (ret->fragment[n] > ret->max_fragment)
			ret->max_fragment = ret->fragment[n];
		i = j;
		n++;
		//printf("n = %d, i = %d\n", n, i);
		assert(n <= SW_MAX_NUM_FRAGS);
	}

	while (n > 0 && ret->fragment[n - 1] == 0)
		n--;

	assert(n > 0);
	assert(ret->max_fragment < SW_MAX_FRAGMENT);
	assert(ret->max_fragment % 2 == 1);
	ret->num_fragments = n;
}

// Default constructor
Key::Key(const char *fileName, int keyBits_) :keyBits(keyBits_) {
	if (keyBits % 8)
		throw runtime_error("Bits of key is not multiple of 8.");
	keyBytes = keyBits / 8;
	if (keyBytes % (sizeof(WORD)))
		throw runtime_error("Bytes of key is not multiple of WORD.");
	keySize = keyBytes / (sizeof(WORD));
	/* If key file does not exist, generate it */
	if (!ifstream(fileName)) {
		genRSAKey(fileName, keyBits_);
	}
	/* Read key from key file */
	RSA *rsa_ = readKeyFile(fileName);
	memcpy(&rsa, rsa_, sizeof(RSA));
	if (RSA_size(rsa_) * 8 != keyBits_)
		throw runtime_error("Key size does match the key file.");
	keySetup(rsa_);
	free(rsa_);
}

RSA * Key::readKeyFile(const char *fileName) {
	BIO *key = BIO_new(BIO_s_file());
	assert(key != NULL);
	assert(BIO_read_filename(key, fileName)==1);
	RSA *rsa = PEM_read_bio_RSAPrivateKey(key, NULL, NULL, NULL);
	BIO_free(key);
	return rsa;
}

void Key::keySetup(RSA *rsa) {
	int key_bytes = RSA_size(rsa);
	int msg_size = key_bytes/sizeof(WORD);
	int key_bits = key_bytes * 8;
	bn2word(e, rsa->e, msg_size);
	bn2word(d, rsa->d, msg_size);
	bn2word(n, rsa->n, msg_size);
	/* Calculate np */
	BIGNUM *R = BN_new();
	BN_set_bit(R, key_bits);
	BIGNUM *NP = BN_new();
	BN_CTX *bn_ctx = BN_CTX_new();
	BIGNUM *R_inv=BN_new();

	BN_mod_inverse(R_inv, R, rsa->n, bn_ctx); /* R*R_inv = 1+N*N_inv */
	BN_mul(NP, R, R_inv, bn_ctx); /* NP = R*R_inv */
	BN_sub_word(NP, 1); /* NP = N*N_inv */
	BN_div(NP, NULL, NP, rsa->n, bn_ctx); /* NP = N_inv */
	bn2word(np, NP, msg_size);

	/* Calculate r_sqr */
	BIGNUM *R_sqr = BN_new();
	BN_mod_mul(R_sqr, R, R, rsa->n, bn_ctx);
	bn2word(r_sqr, R_sqr, msg_size);

	/* For CRT */
	bn2word(pd, rsa->dmp1, msg_size/2);
	bn2word(qd, rsa->dmq1, msg_size/2);
	bn2word(p, rsa->p, msg_size/2);
	bn2word(q, rsa->q, msg_size/2);
	bn2word(iqmp, rsa->iqmp, msg_size/2);
	/* pp */
	BN_zero(R);
	BN_set_bit(R, key_bits/2);
	BN_mod_inverse(R_inv, R, rsa->p, bn_ctx);
	BN_mul(NP, R, R_inv, bn_ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, rsa->p, bn_ctx);
	bn2word(pp, NP, msg_size/2);
	/* qp */
	BN_mod_inverse(R_inv, R, rsa->q, bn_ctx);
	BN_mul(NP, R, R_inv, bn_ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, rsa->q, bn_ctx);
	bn2word(qp, NP, msg_size/2);
	/* pr_sqr, qr_sqr */
	BN_mod_mul(R_sqr, R, R, rsa->p, bn_ctx);
	bn2word(pr_sqr, R_sqr, msg_size/2);
	BN_mod_mul(R_sqr, R, R, rsa->q, bn_ctx);
	bn2word(qr_sqr, R_sqr, msg_size/2);
	/* setup sw */
	getCLNW(&clnw, d, msg_size);
    getVLNW(&vlnw, d, msg_size);
	
	/* Free memory */
	BN_free(R);
	BN_free(NP);
	BN_CTX_free(bn_ctx);
	BN_free(R_inv);
	BN_free(R_sqr);
}

void Key::genRSAKey(const char *fileName, int keyBits) {
	assert(keyBits == 512 || keyBits == 1024 || keyBits == 2048 || keyBits == 4096);
	BIGNUM *e = BN_new();
	BN_set_word(e, RSA_F4 /*65537*/);

	RSA *rsa = RSA_new();
	RSA_generate_key_ex(rsa, keyBits, e, NULL);
	assert(RSA_check_key(rsa));

	BIO *key;
	key = BIO_new_file(fileName, "w");
	assert(key != NULL);
	PEM_write_bio_RSAPrivateKey(key, rsa, NULL, NULL, 0, NULL, NULL);

	/* write key in binary to fileName.bin*/
	char *fileName_bin = (char *)malloc(strlen(fileName)+5);
	strcpy(fileName_bin, fileName);
	strcat(fileName_bin, ".bin");
	FILE *bin_file = fopen(fileName_bin, "w");
	unsigned char *temp_num = (unsigned char *)malloc(keyBits/8);

	/*write key e*/
	unsigned char byte_str[9];
	byte_str[8] = 0;

	fprintf(bin_file, "public key e:\n");
	BN_bn2bin(rsa->e, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->e); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}
	fprintf(bin_file, "\nprivate key d:\n");
	BN_bn2bin(rsa->d, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->d); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}
	fprintf(bin_file, "\npublic key n:\n");
	BN_bn2bin(rsa->n, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->n); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}
	fprintf(bin_file, "\nprivate prime p:\n");
	BN_bn2bin(rsa->p, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->p); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}
	fprintf(bin_file, "\nprivate prime q:\n");
	BN_bn2bin(rsa->q, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->q); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}
	fprintf(bin_file, "\nd mod (p-1):\n");
	BN_bn2bin(rsa->dmp1, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->q); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}
	fprintf(bin_file, "\nd mod (q-1):\n");
	BN_bn2bin(rsa->dmq1, temp_num);
	for(int i=0; i<BN_num_bytes(rsa->q); i++){
		fputs((const char*)char2binstr(byte_str, temp_num[i]), bin_file);
		if(!((i+1)%sizeof(WORD)))
			fprintf(bin_file, "\n");
		else
			fprintf(bin_file, " ");
	}


	RSA_free(rsa);
	BIO_free(key);
	BN_free(e);
	free(fileName_bin);
	free(temp_num);
	fclose(bin_file);
}

void Key::saveSW(const char *fileName) {
	FILE *sw_file = fopen(fileName, "w");
	char s_tmp[16];

	fputs("Constant Length Non-zero Window Contents\n", sw_file);
	for (int i = clnw.num_fragments - 1; i >= 0; i--){
		for (int j = clnw.length[i] - 1; j >=0; j--) {
			if (clnw.fragment[i] >> j & 1)
				s_tmp[clnw.length[i] - 1 -j] = '1';
			else
				s_tmp[clnw.length[i] - 1 -j] = '0';
		}
		s_tmp[clnw.length[i]] = 0;
		fprintf(sw_file, "%3d: length: %2d, value: %s\n", i, clnw.length[i], s_tmp);
	}

    fputs("Variable Sliding Non-zero Window Contents\n", sw_file);
	for (int i = vlnw.num_fragments - 1; i >= 0; i--){
		for (int j = vlnw.length[i] - 1; j >=0; j--) {
			if (vlnw.fragment[i] >> j & 1)
				s_tmp[vlnw.length[i] - 1 -j] = '1';
			else
				s_tmp[vlnw.length[i] - 1 -j] = '0';
		}
		s_tmp[vlnw.length[i]] = 0;
		fprintf(sw_file, "%3d: length: %2d, value: %s\n", i, vlnw.length[i], s_tmp);
	}


	fclose(sw_file);

}

int Key::getKeyBits() {
	return keyBits;
}

RSA * Key::getRSA() {
	return &rsa;
}

