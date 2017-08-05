HOME ?= /home/chao

# CPP config
CC = g++
CFLAG = -O2 -Wall -std=c++11
#CFLAG += -mcmodel=medium

# Openssl config
OPENSSL_LIB = -lcrypto

# Boost config
BOOST_LIB = -lboost_program_options

ifeq ($(HOME), /home/luo.cha)
	OPENSSL_INC = -I$(HOME)/app/include
	OPENSSL_LIB_DIR = -L$(HOME)/app/lib
endif

# CUDA config
ifeq ($(HOME), /home/luo.cha)
	CUDA_DIR = /shared/apps/cuda7.5
else
	CUDA_DIR = /usr/local/cuda
endif

CUDA_SAMPLE = $(CUDA_DIR)/samples
NVCC = nvcc
NVINC = -I$(CUDA_DIR)/include -I$(CUDA_SAMPLE)/common/inc
CUDA_LIB = -lcudart
CUDA_LIB_DIR = -L$(CUDA_DIR)/lib64
ifeq ($(HOME), /home/luo.cha)
	NVCFLAG = -gencode arch=compute_35,code=sm_35
else
	NVCFLAG = -gencode arch=compute_61,code=sm_61
endif
#NVCFLAG = -gencode arch=compute_20,code=sm_20

NVCFLAG += -Wno-deprecated-gpu-targets -std=c++11  
#NVCFLAG += -mcmodel=medium
CURAND_LIB = -lcurand

LIB = $(OPENSSL_LIB) $(CUDA_LIB) $(CURAND_LIB) 
LIB_DIR = $(CUDA_LIB_DIR) $(OPENSSL_LIB_DIR)
INC = $(OPENSSL_INC) $(NVINC)

ifeq ($(dbg), 1)
	CFLAG += -g
	NVCFLAG += -g -G
endif

DEPS = Makefile.dep
SRC_FILES = $(wildcard *.cpp) $(wildcard *.cc) $(wildcard *.cu)
HEADER_FILES = $(wildcard *.hh) $(wildcard *.h)


TARGETS = RSABaseTest RSAGPUAttackTest RSAAttackTest


all: $(TARGETS) $(DEPS)

# Pattern rules
%.o: %.cpp
	$(CC) $(INC) $(CFLAG) -c $< -o $@
%.o: %.cc
	$(CC) $(INC) $(CFLAG) -c $< -o $@
%.o: %.cu
	$(NVCC) $(INC) $(NVCFLAG) -c $< -o $@

$(DEPS): $(SRC_FILES) $(HEADER_FILES)
	$(CC) -MM -x c++ $(SRC_FILES) > $(DEPS)

RSAAttackTest: RSAAttackTest.o RSAAttack.o RSAGPUAttack.o RSAGPU.o RSABase.o Key.o rsa_cpu.o rsa_gpu.o tanc.o 
	$(CC) $(LIB_DIR) $^ -o $@ $(LIB) $(BOOST_LIB)

RSAGPUAttackTest: RSAGPUAttackTest.o RSAGPUAttack.o RSABase.o Key.o rsa_cpu.o RSAGPU.o rsa_gpu.o tanc.o 
	$(CC) $(LIB_DIR) $^ -o $@ $(LIB) $(BOOST_LIB) 

RSABaseTest: RSAGPUAttack.o RSABaseTest.o RSAAttack.o RSABase.o Key.o rsa_cpu.o RSAGPU.o rsa_gpu.o tanc.o 
	$(CC) $(LIB_DIR) $^ -o $@ $(LIB) 

ifneq ($(MAKECMDGOALS), clean)
-include $(DEPS)
endif

.PHONY: clean

clean:
	rm -f *.o $(TARGETS) 
