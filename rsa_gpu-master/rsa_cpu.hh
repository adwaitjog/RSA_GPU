#ifndef RSA_CPU_H
#define RSA_CPU_H

#include "common.hh"

uint64_t get_usec();
unsigned char *char2binstr(unsigned char *binstr, unsigned char c);
void mp_bn2mp(WORD *a, const BIGNUM *bn, int word_len);
void set_random(unsigned char *buf, int len);

template <typename T1, typename T2>
void corr(double *corr_result, T1 * d1, T2 ** d2, int corr_num, int length);
#endif
