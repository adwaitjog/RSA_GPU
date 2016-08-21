# RSA_Research
Efficient GPU RSA implementation

Objective:
Create a CUDA GPU algorithm to implement faster large integer multiplication which can be used create faster RSA implementation as well as other applications. The RSA algorithm does not currently use Chinese remainder theorem for optimization but is just a means to show the beneficial use of the CUDA language with applications where large integer multiplication plays a big role. The reduction method used to reduce x mod n is Barrett Reduction. Barrett Reduction relies on multiplication, bit shifting, and at most two subtractions to reduce x mod n so there is no need to compute other values.

