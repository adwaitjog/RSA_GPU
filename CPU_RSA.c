#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>

#define R_size 129
#define k 1024
#define n_size 128

void square(unsigned char *a, unsigned char *c, unsigned int size);
 
void barrett_reduction(unsigned char *buf, unsigned char *r, unsigned char *n, unsigned char *reduction, unsigned char *temp, unsigned char *shifted, unsigned char *xprime, unsigned char *result, unsigned char *tmp, unsigned int exp_size);
void multiplication(unsigned char *a, unsigned char *b, unsigned char *c, unsigned int size); 
char checkbit(unsigned char *exponent, unsigned int index_of_bit); 
void subtraction(unsigned char *a, unsigned char *b, unsigned char *c, unsigned int size); 
void bit_shift(unsigned char *a, unsigned char *b, unsigned int shift, unsigned int size_of_a); 
void exponentiation(unsigned char *message, unsigned char *exponent, unsigned char *ciphertext, unsigned char *m0_copy, unsigned char *reduction, unsigned char *buf, unsigned char *temp, unsigned char *shifted, unsigned char *xprime, unsigned char *result, unsigned char *tmp, unsigned int exponent_size, unsigned char *r, unsigned char *n); 
int main(int argc, char *argv[]) {

unsigned char *n = calloc((4*n_size + n_size), sizeof(char));

    n[0] = 0xcf;
    n[1] = 0x82;
    n[2] = 0x69;
    n[3] = 0x57;
    n[4] = 0x4d;
    n[5] = 0xe7;
    n[6] = 0x82;
    n[7] = 0x1a;
    n[8] = 0xe4;
    n[9] = 0x20;
    n[10] = 0x14;
    n[11] = 0x47;
    n[12] = 0x39;
    n[13] = 0x52;
    n[14] = 0x55;
    n[15] = 0x28;
    n[16] = 0xed;
    n[17] = 0x3f;
    n[18] = 0xa4;
    n[19] = 0x61;
    n[20] = 0xd3;
    n[21] = 0xf4;
    n[22] = 0xf2;
    n[23] = 0x34;
    n[24] = 0x6a;
    n[25] = 0x54;
    n[26] = 0xd1;
    n[27] = 0x15;
    n[28] = 0x7d;
    n[29] = 0x67;
    n[30] = 0xb;
    n[31] = 0xc7;
    n[32] = 0x8c;
    n[33] = 0xfe;
    n[34] = 0x1b;
    n[35] = 0x68;
    n[36] = 0x44;
    n[37] = 0x7;
    n[38] = 0x26;
    n[39] = 0x99;
    n[40] = 0xb;
    n[41] = 0x4d;
    n[42] = 0xc7;
    n[43] = 0x3f;
    n[44] = 0x52;
    n[45] = 0x90;
    n[46] = 0x2;
    n[47] = 0x68;
    n[48] = 0x3d;
    n[49] = 0x83;
    n[50] = 0x1d;
    n[51] = 0x79;
    n[52] = 0x7a;
    n[53] = 0x3f;
    n[54] = 0x36;
    n[55] = 0xf3;
    n[56] = 0x41;
    n[57] = 0x8b;
    n[58] = 0x7c;
    n[59] = 0xdf;
    n[60] = 0x64;
    n[61] = 0xac;
    n[62] = 0x74;
    n[63] = 0x7c;
    n[64] = 0x8;
    n[65] = 0xdb;
    n[66] = 0xa0;
    n[67] = 0x6f;
    n[68] = 0x10;
    n[69] = 0x71;
    n[70] = 0x13;
    n[71] = 0x86;
    n[72] = 0xaf;
    n[73] = 0xb8;
    n[74] = 0x71;
    n[75] = 0xf8;
    n[76] = 0xf0;
    n[77] = 0x45;
    n[78] = 0xa7;
    n[79] = 0x94;
    n[80] = 0xb3;
    n[81] = 0x6b;
    n[82] = 0x1e;
    n[83] = 0xff;
    n[84] = 0x8e;
    n[85] = 0x13;
    n[86] = 0xae;
    n[87] = 0xc2;
    n[88] = 0x59;
    n[89] = 0x56;
    n[90] = 0xd3;
    n[91] = 0xd;
    n[92] = 0x20;
    n[93] = 0x62;
    n[94] = 0x21;
    n[95] = 0x30;
    n[96] = 0x1d;
    n[97] = 0x6b;
    n[98] = 0x5e;
    n[99] = 0xc;
    n[100] = 0x0;
    n[101] = 0x35;
    n[102] = 0xae;
    n[103] = 0xbd;
    n[104] = 0xa5;
    n[105] = 0xc2;
    n[106] = 0x25;
    n[107] = 0x98;
    n[108] = 0xe7;
    n[109] = 0x57;
    n[110] = 0x89;
    n[111] = 0xc;
    n[112] = 0x12;
    n[113] = 0xf9;
    n[114] = 0x33;
    n[115] = 0x3d;
    n[116] = 0xa;
    n[117] = 0xac;
    n[118] = 0x51;
    n[119] = 0xd8;
    n[120] = 0x5c;
    n[121] = 0x40;
    n[122] = 0x9b;
    n[123] = 0xfa;
    n[124] = 0xf9;
    n[125] = 0xbc;
    n[126] = 0x3;
    n[127] = 0xe6;

    
unsigned char *r = calloc(2*n_size, sizeof(char));

    r[0] = 0x7f;
    r[1] = 0x9d;
    r[2] = 0xe9;
    r[3] = 0x40;
    r[4] = 0x57;
    r[5] = 0x2;
    r[6] = 0x6e;
    r[7] = 0x93;
    r[8] = 0x2b;
    r[9] = 0xb4;
    r[10] = 0xe3;
    r[11] = 0xfd;
    r[12] = 0xba;
    r[13] = 0xc;
    r[14] = 0xcd;
    r[15] = 0x78;
    r[16] = 0x7d;
    r[17] = 0xae;
    r[18] = 0x8d;
    r[19] = 0x80;
    r[20] = 0xff;
    r[21] = 0x66;
    r[22] = 0x33;
    r[23] = 0xb;
    r[24] = 0x28;
    r[25] = 0x4c;
    r[26] = 0x93;
    r[27] = 0x30;
    r[28] = 0x2;
    r[29] = 0x92;
    r[30] = 0xa0;
    r[31] = 0x7c;
    r[32] = 0xf1;
    r[33] = 0xc;
    r[34] = 0xa;
    r[35] = 0x5e;
    r[36] = 0xf2;
    r[37] = 0x9a;
    r[38] = 0x8f;
    r[39] = 0x17;
    r[40] = 0x4c;
    r[41] = 0x82;
    r[42] = 0x25;
    r[43] = 0xe5;
    r[44] = 0x98;
    r[45] = 0x45;
    r[46] = 0x4d;
    r[47] = 0xc7;
    r[48] = 0xd9;
    r[49] = 0x53;
    r[50] = 0x5e;
    r[51] = 0x5a;
    r[52] = 0x6e;
    r[53] = 0x37;
    r[54] = 0x43;
    r[55] = 0x29;
    r[56] = 0x88;
    r[57] = 0xcb;
    r[58] = 0xe9;
    r[59] = 0x31;
    r[60] = 0x2f;
    r[61] = 0xd7;
    r[62] = 0x6;
    r[63] = 0xfb;
    r[64] = 0xf1;
    r[65] = 0x38;
    r[66] = 0xdf;
    r[67] = 0xc4;
    r[68] = 0xda;
    r[69] = 0x7c;
    r[70] = 0x9;
    r[71] = 0x5c;
    r[72] = 0xf9;
    r[73] = 0x2b;
    r[74] = 0x81;
    r[75] = 0x30;
    r[76] = 0xe9;
    r[77] = 0x29;
    r[78] = 0xcd;
    r[79] = 0x45;
    r[80] = 0xee;
    r[81] = 0xff;
    r[82] = 0x5b;
    r[83] = 0x3c;
    r[84] = 0x23;
    r[85] = 0x6d;
    r[86] = 0xb9;
    r[87] = 0xa1;
    r[88] = 0x89;
    r[89] = 0x3f;
    r[90] = 0xc3;
    r[91] = 0x9e;
    r[92] = 0xa1;
    r[93] = 0x30;
    r[94] = 0x98;
    r[95] = 0xf8;
    r[96] = 0xc8;
    r[97] = 0x4a;
    r[98] = 0xbe;
    r[99] = 0xc6;
    r[100] = 0x49;
    r[101] = 0xf7;
    r[102] = 0xb3;
    r[103] = 0xff;
    r[104] = 0x9;
    r[105] = 0x3b;
    r[106] = 0x94;
    r[107] = 0x9d;
    r[108] = 0x2f;
    r[109] = 0x5c;
    r[110] = 0x68;
    r[111] = 0xe1;
    r[112] = 0x6;
    r[113] = 0xf1;
    r[114] = 0x33;
    r[115] = 0xeb;
    r[116] = 0xc5;
    r[117] = 0x88;
    r[118] = 0xa5;
    r[119] = 0x1c;
    r[120] = 0xde;
    r[121] = 0x2c;
    r[122] = 0x64;
    r[123] = 0xad;
    r[124] = 0x5c;
    r[125] = 0xc9;
    r[126] = 0xeb;
    r[127] = 0x1c;
    r[128] = 0x1;

	unsigned char *message = calloc(n_size, sizeof(char));
	message[0] = 0x68;//h
	message[1] = 0x65;//e
	message[2] = 0x6c;//l
	message[3] = 0x6c;//l
	message[4] = 0x6f;//o


	unsigned char *exponent = malloc(3);
	exponent[0] = 0x01;
	exponent[1] = 0x00;
	exponent[2] = 0x01;
	unsigned int exponent_size = 3;

	//exponentiate m^e mod n
	//parameters: 
	//message(m)
	//exponent(e)
	//precomputation of r = floor((4^k)/n) where k is found by where (2^k) > n
	//modulus (n)
	unsigned char *ciphertext = calloc(n_size, sizeof(char));
	struct timeval cpu_start, cpu_end;
	struct timezone tzp;

	gettimeofday(&cpu_start, &tzp);	

	unsigned char *m0_copy = calloc(n_size, sizeof(char));
	unsigned char *reduction = calloc(n_size, sizeof(char));
	unsigned char *buf = calloc((n_size * 2) + 1, sizeof(char));
	unsigned char *temp = calloc(3 * n_size, sizeof(char));
	unsigned char *shifted = calloc(n_size, sizeof(char));
	unsigned char *xprime = calloc(2*n_size, sizeof(char));
	unsigned char *result = calloc(n_size + 1, sizeof(char));
	unsigned char *tmp = calloc(n_size + 1, sizeof(char));

	exponentiation(message, exponent, ciphertext, m0_copy, reduction, buf, temp, shifted, xprime, result, tmp, exponent_size, r, n);

	unsigned int d_exponent_size = 128;
	unsigned char *d_exponent = malloc(128);
    
    d_exponent[0] = 0x91;
    d_exponent[1] = 0xa;
    d_exponent[2] = 0xb3;
    d_exponent[3] = 0x66;
    d_exponent[4] = 0xbd;
    d_exponent[5] = 0x6f;
    d_exponent[6] = 0x18;
    d_exponent[7] = 0xde;
    d_exponent[8] = 0xd5;
    d_exponent[9] = 0x1;
    d_exponent[10] = 0x61;
    d_exponent[11] = 0x36;
    d_exponent[12] = 0x95;
    d_exponent[13] = 0x6d;
    d_exponent[14] = 0xdd;
    d_exponent[15] = 0x33;
    d_exponent[16] = 0xdb;
    d_exponent[17] = 0x26;
    d_exponent[18] = 0x3;
    d_exponent[19] = 0xe;
    d_exponent[20] = 0x68;
    d_exponent[21] = 0x54;
    d_exponent[22] = 0x73;
    d_exponent[23] = 0xa0;
    d_exponent[24] = 0xe0;
    d_exponent[25] = 0x6e;
    d_exponent[26] = 0x70;
    d_exponent[27] = 0x74;
    d_exponent[28] = 0x25;
    d_exponent[29] = 0x8b;
    d_exponent[30] = 0x2b;
    d_exponent[31] = 0xfb;
    d_exponent[32] = 0x9e;
    d_exponent[33] = 0x3c;
    d_exponent[34] = 0x34;
    d_exponent[35] = 0x2e;
    d_exponent[36] = 0x45;
    d_exponent[37] = 0x10;
    d_exponent[38] = 0x10;
    d_exponent[39] = 0x6c;
    d_exponent[40] = 0xfb;
    d_exponent[41] = 0xb7;
    d_exponent[42] = 0x9b;
    d_exponent[43] = 0xc8;
    d_exponent[44] = 0xcf;
    d_exponent[45] = 0x71;
    d_exponent[46] = 0xd9;
    d_exponent[47] = 0x96;
    d_exponent[48] = 0xb7;
    d_exponent[49] = 0xbb;
    d_exponent[50] = 0x5f;
    d_exponent[51] = 0x19;
    d_exponent[52] = 0x76;
    d_exponent[53] = 0x36;
    d_exponent[54] = 0x49;
    d_exponent[55] = 0x6a;
    d_exponent[56] = 0xb3;
    d_exponent[57] = 0x83;
    d_exponent[58] = 0xc3;
    d_exponent[59] = 0x59;
    d_exponent[60] = 0x2e;
    d_exponent[61] = 0x62;
    d_exponent[62] = 0x87;
    d_exponent[63] = 0xa2;
    d_exponent[64] = 0x5a;
    d_exponent[65] = 0x2f;
    d_exponent[66] = 0x60;
    d_exponent[67] = 0x75;
    d_exponent[68] = 0x1;
    d_exponent[69] = 0xf0;
    d_exponent[70] = 0x3f;
    d_exponent[71] = 0xdb;
    d_exponent[72] = 0x5a;
    d_exponent[73] = 0x70;
    d_exponent[74] = 0x1f;
    d_exponent[75] = 0x44;
    d_exponent[76] = 0x6a;
    d_exponent[77] = 0x9c;
    d_exponent[78] = 0x77;
    d_exponent[79] = 0x63;
    d_exponent[80] = 0xba;
    d_exponent[81] = 0xcb;
    d_exponent[82] = 0xcd;
    d_exponent[83] = 0x1f;
    d_exponent[84] = 0x99;
    d_exponent[85] = 0x70;
    d_exponent[86] = 0x89;
    d_exponent[87] = 0x94;
    d_exponent[88] = 0x31;
    d_exponent[89] = 0x2;
    d_exponent[90] = 0xa;
    d_exponent[91] = 0x32;
    d_exponent[92] = 0x96;
    d_exponent[93] = 0x65;
    d_exponent[94] = 0x21;
    d_exponent[95] = 0x21;
    d_exponent[96] = 0x59;
    d_exponent[97] = 0x55;
    d_exponent[98] = 0x8a;
    d_exponent[99] = 0xd0;
    d_exponent[100] = 0x7a;
    d_exponent[101] = 0x1c;
    d_exponent[102] = 0xd2;
    d_exponent[103] = 0x66;
    d_exponent[104] = 0x48;
    d_exponent[105] = 0x95;
    d_exponent[106] = 0x8;
    d_exponent[107] = 0xd3;
    d_exponent[108] = 0x6b;
    d_exponent[109] = 0xe7;
    d_exponent[110] = 0x9c;
    d_exponent[111] = 0xb9;
    d_exponent[112] = 0x96;
    d_exponent[113] = 0x20;
    d_exponent[114] = 0x20;
    d_exponent[115] = 0x8a;
    d_exponent[116] = 0xe5;
    d_exponent[117] = 0x4d;
    d_exponent[118] = 0x3e;
    d_exponent[119] = 0x53;
    d_exponent[120] = 0x4b;
    d_exponent[121] = 0xd8;
    d_exponent[122] = 0x21;
    d_exponent[123] = 0x4;
    d_exponent[124] = 0x81;
    d_exponent[125] = 0x7d;
    d_exponent[126] = 0x29;
    d_exponent[127] = 0x38;


	memset(message, 0x00, n_size);
	exponentiation(ciphertext, d_exponent, message, m0_copy, reduction, buf, temp, shifted, xprime, result, tmp, d_exponent_size, r, n);

	gettimeofday(&cpu_end, &tzp);
	printf("CPU time: %.6f\n", (cpu_end.tv_sec - cpu_start.tv_sec) + (cpu_end.tv_usec - cpu_start.tv_usec) / 1000000.0);

	int z = 0;
	while (z < n_size) {
		printf("message[%d] = %x\n", z, message[z]);
		z++;
	}

	free(r);
	free(n);
	free(exponent);
	free(d_exponent);
	free(message);
	free(ciphertext);
	free(m0_copy);
	free(reduction);
	free(buf);
	free(temp);
	free(shifted);
	free(xprime);
	free(result);
	free(tmp);

	return 0;
}

void exponentiation(unsigned char *message, unsigned char *exponent, unsigned char *ciphertext, unsigned char *m0_copy, unsigned char *reduction, unsigned char *buf, unsigned char *temp, unsigned char *shifted, unsigned char *xprime, unsigned char *result, unsigned char *tmp, unsigned int exponent_size, unsigned char *r, unsigned char *n) {

	//get the total amount of bits in strlen(exponent) zero based
	//not including the final char index msb (byte)
	unsigned int total_bits = exponent_size * 8 - 1;
	
	//find the most signinficant bit in the most significant byte (char index)
	//find most significant bit in exponent[exp_size - 1]
	unsigned char mask = 0x80; //10000000 in binary
	unsigned char msb = 0;
	int i = 0;
	while(i < 8) {
		if((exponent[exponent_size - 1] & (mask >> i)) == (mask >> i)) {
			msb = i;
			break;
		}
		i++;
	}

	//subtract most significant bit from total_bits to know total amount of significant bits
	//for loop of exponent in binary
	unsigned int exp_bits = (total_bits - msb);

	//keep copy of original message m0
	memcpy(m0_copy, message, n_size);

	//compute m^e where e is in binary 
	//RULES:
		//iterate over the values of msb to 0 bit by bit
		//msb is amount of relevent bits to check for exponentiation
		//total bits is the amount of total bits in exponent lenth
		//square m(current) for each itteration
		//check if current bit is 1
		//current bit is 1: m(current) * m0
		//curent bit is 0: return to loop

	//subtract one from total because to exponentiate in binary
	//start at the second bit after the most significant bit
	//each bit equals m^2 and when the current bit is 1 it is
	//(m^2)*m0 or if it is 0 then m^2
	int index_of_bit = exp_bits - 1; //subtraction of 1 is becuase msb is zero based
	while (index_of_bit >= 0) {

		//allocate space for reduction to hold a value strickly less than n
		//buf holds value at most m^2 which is less than n^2

		//calculate m^2
		square(message, buf, n_size);
	
		//calculate m^2 mod n
		barrett_reduction(buf, r, n, reduction, temp, shifted, xprime, result, tmp, exponent_size);

		memcpy(message, reduction, n_size);
		memset(buf, 0, 2*n_size);
		memset(reduction, 0, n_size);

		char bit;
		if ((bit = checkbit(exponent, index_of_bit)) == 1) {
			//m * m0
			multiplication(message, m0_copy, buf, n_size);

			//barrett reduction
			barrett_reduction(buf, r, n, reduction, temp, shifted, xprime, result, tmp, exponent_size);
			memcpy(message, reduction, n_size);
			
			memset(buf, 0x00, 2*n_size);
			
			memset(reduction, 0x00, n_size);
		
		}

		index_of_bit--;
	}
	
	//copy back final value of message to ciphertext for decryption
	memcpy(ciphertext, message, n_size);

	memset(buf, 0x00, 2*n_size);
	memset(reduction, 0x00, n_size);
	memset(m0_copy, 0x00, n_size);

	return;
}

void multiplication(unsigned char *a, unsigned char *b, unsigned char *c, unsigned int size) {

	unsigned int result_position = 0;
	unsigned int multiplicand_position;
	for(multiplicand_position = 0; multiplicand_position < size; multiplicand_position++) {

		register unsigned int result_position = multiplicand_position;
		unsigned char result_carry = 0;
        	register unsigned short product;
        	unsigned int multiplier_position = 0;
        	register unsigned short sum;
        
		unsigned int loop = 0;
		while(loop < size) {
            
			unsigned short sum;
			product = a[multiplier_position] * b[multiplicand_position];
            
			multiplier_position++;
			sum = (c[result_position] + (product<<8>>8) + result_carry);

			result_carry = (sum >> 8);
            
			c[result_position] = sum;
          
			result_position++;
			loop++;
		}        
       
		 
		sum = (c[result_position] + result_carry);
        	c[result_position] = sum;

		result_carry = (sum >> 8);
		c[result_position+ 1] += result_carry;
	}


	return;
}

void square(unsigned char *a, unsigned char *c, unsigned int size) {

	unsigned int result_position = 0;
	unsigned int multiplicand_position;
	for(multiplicand_position = 0; multiplicand_position < size; multiplicand_position++) {

		register unsigned int result_position = multiplicand_position;
		unsigned char result_carry = 0;
        	register unsigned short product;
        	unsigned int multiplier_position = 0;
        	register unsigned short sum;
        
		unsigned int loop = 0;
		while(loop < size) {
            
			unsigned short sum;
			product = a[multiplier_position] * a[multiplicand_position];
            
			multiplier_position++;
			sum = (c[result_position] + (product<<8>>8) + result_carry);

			result_carry = (sum >> 8);
            
			c[result_position] = sum;
          
			result_position++;
			loop++;
		}        
       
		 
		sum = (c[result_position] + result_carry);
        	c[result_position] = sum;

		result_carry = (sum >> 8);
		c[result_position+ 1] += result_carry;
	}


	return;
}


void barrett_reduction(unsigned char *buf, unsigned char *r, unsigned char *n, unsigned char *reduction, unsigned char *temp, unsigned char *shifted, unsigned char *xprime, unsigned char *result, unsigned char *tmp, unsigned int exp_size) {
	////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////

	////////////////////  calculate: t = x - ((x*r)/(4^k))*n  /////////////////////////

	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////
	//multiply: x * r = temp
	//size of x is assumed to be the largest value which is = largest value of 2*n
	//size of r is precomputed

	multiplication(r, buf, temp, 2*n_size);

	//shift bits by (4^k) or (2^(2*k))
	//shift temp by 2*k store to shifted
	//size of shifted is 2*n + sizeof(r)
	
	//find the actual amount of bits/bytes left in the value of temp
	//which is equal to x * r so that the correct size of the value
	//can be used in the bit_shift function

	//first find the amount of bytes from most significant byte
	//to least and then when one char does not equal to 0x00

	unsigned int zero_bytes = 0;
	int count = (3*n_size) - 1;
	while((count >= 0) && (temp[count] == 0x00)) {
		count--;
		zero_bytes++;
	}

	bit_shift(temp, shifted, k, (3*n_size) - zero_bytes);
	
	//multiply: shifted * n = xprime
	//xprime is the size of 2*n + R_size - (k >> 0x07) + n
	 //2*n_size + R_size - ((2*k) >> 0x03) + n_size,
	//multiplication(shifted, n, xprime, (4*n_size) - ((2*k) >> 0x03));//2*n_size + R_size - ((2*k) >> 0x03) + n_size)
	multiplication(shifted, n, xprime, n_size);

	//subtract xprime from x^2
	subtraction(buf, xprime, result, 2*n_size);

	//compare the value of t = x - xprime and see if the value is less than n, meaning it is within
	//the field of n, if the value is not within the field of n then reduce the value by subtracting
	//the value of result = t - n which is guaranteed to be in the field of n
	if ((result[n_size] == 0x00) && (result[n_size - 1] < n[n_size - 1])) {
		memcpy(reduction, result, n_size);
	}

	else {
		subtraction(result, n, tmp, n_size + 1);
		memcpy(reduction, tmp, n_size);
		memset(tmp, 0x00, n_size + 1);
	}

	memset(temp, 0x00, 3*n_size);
	memset(shifted, 0x00, n_size); 
	memset(xprime, 0x00, 2*n_size);
	memset(result, 0x00, n_size + 1);

	return;
}

void subtraction(unsigned char *a, unsigned char *b, unsigned char *c, unsigned int size) {

	//borrow represents the value 1 or 0 for the current index
	//indecating if the current index has been borrowed from by
	//the previous index, borrow = 1 true, 0 false
	unsigned char borrow = 0x00; //value is 0 or 1

	//loop through array a size and subtract a - b,
	//a is guaranted to be greater than b in 
	//barrett reduction
	unsigned int i = 0;
	while(i < size) {
		//check current value of a to make sure that it is
		//not 0 when the previous index has borrowed
		if (a[i] == 0 && borrow == 1) {
			//borrow from next sequential index with
			//0x100 and subtract 0x01 for the
			//previous borrow which is = 0xff
			c[i] = 0xff - b[i];
			//turn on borrow for next index
			borrow = 0x01;
			i++;
			continue;
		}

		//calculate current value of a along with if the 
		//previous index has borrowed
		a[i] = a[i] - borrow;
	
		//calculate the value of a - b only when a - b >= 0
		//borrow has already been accounted for
		if (a[i] >= b[i]) {
			c[i] = a[i] - b[i];
			borrow = 0x00;
		}

		//a - b !> 0, borrow from next sequential index by 
		//taking the value 0x100 and adding to a[i] and 
		//subtracting b[i] which will give a value between
		//{0x01...0xff} and turn on borrow for next index
		else {
			c[i] = 0x100 + a[i] - b[i];
			borrow = 0x01;
		}
		i++;
	}
	return;
}

char checkbit(unsigned char *exponent, unsigned int index_of_bit) {

	unsigned char bit;
	//get the characters index of which the bit is located in by 
	//taking index_of_bit which is the size of the bits left to 
	//check and divide by 8 giving the location index of the
	//current bit to be checked
	unsigned int quotient = (index_of_bit >> 0x03); // index_of_bit / 8

	//find the bit within the index previously found by finding the 
	//remainder of 8 % index of bit, this will locate the exact
	//bit to be checked
	unsigned int remainder = index_of_bit & (0x07); // index_of_bit % 8
	
	//mask is equivelent to 1 in order to compare a single bit with a
	//the current bit to be checked
	unsigned char mask = 0x01; // use single bit to mask with selected bit
	
	//use the remainder by knowing the index of the character and 
	//the remainder allows the bit to be shifted to the position of
	//the current bit to be checked
	mask = mask << remainder; // shift single bit to bit_in_index position
	
	//bit is now located at index_of_bit character index of array
	//and bit location bit_in_index in group of 8 bits at index
	bit = (exponent[quotient] & mask); // & to see if single bit is on or off

	//shift bit back to the 1 position to represent value 1 or 0
	bit = bit >> remainder; //shift bit back to value of one or zero
	
	return bit;
}

//b is expected to be completely zero before shift
void bit_shift(unsigned char *a, unsigned char *b, unsigned int k_val, unsigned int size_of_a) {

        //expected that k will be equivlent to some power of 2
        //represents the division of (4^k) which is = (2^(2*k))
        unsigned int shift = k_val * 2;

        //quotient represents groups of 8 bits that equal 0 as in >> 8 in single char
        //leaving it to be the value of 0x00
        unsigned int quotient = shift >> 0x03; // k / 8 as integer

	//in case that the shift is greater than the actual value of the
	//number being shifted
	if(quotient > size_of_a) {
		return;
	}

        //remainder will find final char index shift value = {0...7}
        //the specific bits to be shifted in the last group which is not greater than 7
        unsigned int remainder = shift & 0x07; // k % 8 
        //move a to b by shifting the characters an index of quotient amount
        //and then use the remainder to shift the final index to correct 
        //position
        unsigned int constant = (size_of_a - quotient);

        unsigned int j = 0;
        while (j < constant) {
                b[j] = a[quotient + j] >> remainder;
                
		unsigned char cpy_bits = a[quotient + j + 1] << (8 - remainder);
                
		b[j] = b[j] | cpy_bits;
                j++;
        }

        return;
}
