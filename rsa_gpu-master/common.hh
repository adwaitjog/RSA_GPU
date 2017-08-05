#ifndef COMMON_H
#define COMMON_H
#include <stdint.h>
#include <iostream>

#define USE_64BIT

#ifdef USE_64BIT
typedef uint64_t WORD;
#define BITS_PER_WORD (64)
#define BYTES_PER_WORD (8)
#else
typedef uint32_t WORD;
#define BITS_PER_WORD (32)
#define BYTES_PER_WORD (4)
#endif

#define MAX_KEY_BITS (512)
#define MAX_MSG_SIZE (8)	/* in terms of WORD */
#define KEY_SIZE (8)
#define KEY_BITS (512)

#define MAX_MSG_NUM 1600000

#ifdef __i386__
#  define RDTSC_DIRTY "%eax", "%ebx", "%ecx", "%edx"
#elif __x86_64__
#  define RDTSC_DIRTY "%rax", "%rbx", "%rcx", "%rdx"
#else
# error unknown platform
#endif

#define RDTSC_START(cycles)         \
	do {                            \
		register unsigned cyc_low;  \
		register unsigned cyc_high; \
		asm volatile("CPUID\n\t"    \
				"RDTSC\n\t"         \
				"mov %%eax, %0\n\t" \
				"mov %%edx, %1\n\t" \
				: "=r" (cyc_low),   \
				  "=r" (cyc_high)   \
				:: RDTSC_DIRTY);    \
		(cycles) = cyc_high;   		\
		(cycles) = (cycles << 32) + cyc_low; \
	} while (0)

#define RDTSC_STOP(cycles)          \
	do {                            \
		register unsigned cyc_low;  \
		register unsigned cyc_high; \
		asm volatile("RDTSCP\n\t"   \
				"mov %%eax, %0\n\t" \
				"mov %%edx, %1\n\t" \
				"CPUID\n\t"         \
				: "=r" (cyc_low),   \
				  "=r" (cyc_high)   \
				:: RDTSC_DIRTY);    \
		(cycles) = cyc_high;		\
		(cycles) = (cycles << 32) + cyc_low;  \
	} while(0)

#define SW_MAX_NUM_FRAGS 256 
#define SW_MAX_FRAGMENT 32
/* For sliding algorithms (both CLNW and VLNW) */
struct SW {
	uint16_t fragment[SW_MAX_NUM_FRAGS];
	uint16_t length[SW_MAX_NUM_FRAGS];
	int num_fragments;
	int max_fragment;
};

enum class SW_Type {none, clnw, vlnw};

std::ostream & operator << (std::ostream& out, SW_Type swType);

#endif
