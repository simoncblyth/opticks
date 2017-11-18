
#include "OpticksFlags.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
        
    LOG(info) << argv[0] ;

    for(unsigned i=0 ; i < 16 ; i++)
    {
        unsigned msk = 0x1 << i ;
        std::cout  
                  << " ( 0x1 << " << std::setw(2) << i << " ) "  
                  << " (i+1) " << std::setw(2) << std::hex << (i + 1) << std::dec
                  << " " << std::setw(2)  << OpticksFlags::FlagMask(msk, true) 
                  << " " << std::setw(20) << OpticksFlags::FlagMask(msk, false)
                  << " " << std::setw(6) << std::hex << msk << std::dec 
                  << " " << std::setw(6) << std::dec << msk << std::dec 
                  << std::endl 
                  ; 
 
    }

    return 0 ; 
}
