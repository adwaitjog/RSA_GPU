#ifndef TANC_HH
#define TANC_HH
#include "common.hh"

//void attack_nocrt(int msg_bytes, int msg_num, int trace_num, WORD *d_host, int timing, int bit_size, int bit_num);
template <typename T>
void dt_attack(int msg_size, int msg_num, int trace_num, int bit_size, int bit_num, T *t_h, WORD *d_host,
	WORD *msges0, WORD *msges1, WORD *msges_mont);
template <typename T>
void dt_attack_sw(int msg_size, int msg_num, int trace_num, T *t_h, struct SW sw,
	WORD *msges0, int seg_num);
void attack_sw(int msg_bytes, int msg_num, int trace_num, int timing, struct SW sw, int seg_num);
template <typename T>
void dt_attack_vlnw(int msg_size, int msg_num, int trace_num, T *t_h, struct SW sw,
	WORD *msges0, int seg_num);
#endif
