/* RSA GPU implementation CPU code
 * 10/27/2016 Chao Luo
 */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/evp.h>

#include "common.hh"
#include "rsa_cpu.hh"

std::ostream & operator << (std::ostream& out, SW_Type swType) {
	switch (swType) {
	case (SW_Type::clnw):
		out << "CLNW";
		break;
	case (SW_Type::vlnw):
		out << "VLNW";
		break;
	case (SW_Type::none):
		out << "NONE";
	}
	return out;
}

uint64_t get_usec() {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL)==0);
	return tv.tv_sec * 1e6 + tv.tv_usec;
}

/*Convert char to binary string
 * binstr should be at least 9 bytes long
 */
unsigned char *char2binstr(unsigned char *binstr, unsigned char c) {
	for(int i=0; i<8; i++){
		binstr[i] = (c & (0x80>>i)) ? '1' : '0';
	}
	binstr[8]=0;
	return binstr;
}


/* Convert BIGNUM into WORD array */
void mp_bn2mp(WORD *a, const BIGNUM *bn, int word_len) {
	assert(word_len * (int)sizeof(WORD) >= BN_num_bytes(bn));
	memset(a, 0, sizeof(WORD) * word_len);
	memcpy(a, bn->d, BN_num_bytes(bn));
}

/* Set random */
void set_random(unsigned char *buf, int len) {
	for(int i=0; i<len; i++)
		buf[i]=rand()%256;
}


template <typename T1, typename T2>
void corr(double *corr_result, T1 * d1, T2 ** d2, int corr_num, int length) {
	double *mean = (double *)malloc(sizeof(double) * (corr_num + 1));
	double *var = (double *)malloc(sizeof(double) * (corr_num + 1));
	memset(mean, 0, sizeof(double) * (corr_num + 1));
	memset(var, 0, sizeof(double) * (corr_num + 1));
	memset(corr_result, 0, sizeof(double) * (corr_num));

	/* Calculate mean */
	for (int i = 0; i < length; i++) {
		mean[0] += (double) d1[i];
		for (int j = 0; j < corr_num; j++) {
			mean[j + 1] += (double) d2[j][i];
		}
	}
	mean[0] /= (double) length;
	for (int j = 0; j < corr_num; j++) {
		mean[j + 1] /= (double) length;
	}

	/* Calculate var */
	for (int i = 0; i < length; i++) {
		var[0] += ((double) d1[i] - mean[0]) * ((double) d1[i] - mean[0]);
		for (int j = 0; j < corr_num; j++) {
			var[j + 1] += ((double) d2[j][i] - mean[j + 1]) * 
				((double) d2[j][i] - mean[j + 1]);
		}
	}

	/* Calculate corr */
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < corr_num; j++) {
			corr_result[j] += ((double) d1[i] - mean[0]) * 
				((double) d2[j][i] - mean[j + 1]);
		}
	}

	for (int j = 0; j < corr_num; j++) {
		corr_result[j] /= sqrt(var[0] * var[j + 1]);
	}

	free(mean);
	free(var);
}

template void corr<int ,int> (double *corr_result, int * d1, int ** d2, int corr_num, int length); 
template void corr<uint64_t ,uint64_t> 
	(double *corr_result, uint64_t * d1, uint64_t ** d2, int corr_num, int length); 
template void corr<double ,uint64_t> (double *corr_result, double * d1, uint64_t ** d2, int corr_num, int length); 
