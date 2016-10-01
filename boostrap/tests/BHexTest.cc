#include "BHex.cc"
#include "PLOG.hh"



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

     unsigned long long x0 = 0x0123456789abcdef ; 
     unsigned long long a0 = BHex<unsigned long long>::hex_lexical_cast("0123456789abcdef") ; 
     assert( x0 == a0 ); 

     unsigned long long x1 = 0xfedcba9876543210 ; 
     unsigned long long a1 = BHex<unsigned long long>::hex_lexical_cast("fedcba9876543210") ; 
     assert( x1 == a1 ); 


    return 0 ; 
}
