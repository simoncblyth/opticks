#include "Flags.hh"    
#include <iostream>
#include <iomanip>

int main()
{
    const char* path = "/tmp/GFlagIndexLocal.ini";
    Flags flags ;
    flags.read(path);
    //flags.dump();

    unsigned long long seq = 0xfedcba9876543210 ; 
    std::string sseq = flags.getSequenceString(seq);

    std::cout 
              << std::setw(18) << seq 
              << " : "
              << sseq 
              << std::endl ; 

    return 0 ; 
}



