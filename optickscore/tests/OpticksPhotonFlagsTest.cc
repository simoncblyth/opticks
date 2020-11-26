#include <vector>
#include "OPTICKS_LOG.hh"
#include "NGLM.hpp"
#include "OpticksPhotonFlags.hh"

union Quad 
{
    glm::ivec4 i ;  
    glm::uvec4 u ; 
    glm::vec4  f ;
}; 


void test_OpticksPhotonFlags(int boundary, unsigned sensorIndex, unsigned nodeIndex, unsigned photonIndex, unsigned flagMask, bool dump  )
{
    OpticksPhotonFlags a(boundary, sensorIndex, nodeIndex, photonIndex, flagMask); 

    unsigned x = ( (boundary & 0xffff) << 16 ) | ( sensorIndex & 0xffff )  ; 
    unsigned y = nodeIndex ; 
    unsigned z = photonIndex ; 
    unsigned w = flagMask ; 

    Quad q ; 
    q.u = { x, y, z, w } ; 

    int      boundary_    = OpticksPhotonFlags::Boundary(   q.f);  
    unsigned sensorIndex_ = OpticksPhotonFlags::SensorIndex(q.f);  
    unsigned nodeIndex_   = OpticksPhotonFlags::NodeIndex(  q.f);  
    unsigned photonIndex_ = OpticksPhotonFlags::PhotonIndex(q.f);  
    unsigned flagMask_    = OpticksPhotonFlags::FlagMask(   q.f);  

    bool expect = 
         ( boundary_    == boundary )    && 
         ( sensorIndex_ == sensorIndex ) && 
         ( nodeIndex_   == nodeIndex )   && 
         ( photonIndex_ == photonIndex ) && 
         ( flagMask_    == flagMask ) ; 


    OpticksPhotonFlags b(q.f); 
    bool expect2 = b == a ; 

    //std::cout << a.desc() << std::endl ; 
    //std::cout << b.desc() << std::endl ; 
    std::cout << a.brief() << std::endl ; 
    std::cout << b.brief() << std::endl ; 


    if(!expect || !expect2 || dump)
        std::cout 
            << " boundary "       << std::setw(10) << boundary
            << " boundary(hex) "  << std::setw(10) << std::hex << boundary  << std::dec 
            << " boundary_ "      << std::setw(10) << boundary_ 
            << " boundary_(hex) " << std::setw(10) << std::hex << boundary_  << std::dec 
            << " sensorIndex_ "   << std::setw(10) << sensorIndex_ 
            << " nodeIndex_ "     << std::setw(10) << nodeIndex_ 
            << " photonIndex_ "   << std::setw(10) << photonIndex_ 
            << " flagMask_ "      << std::setw(10) << flagMask_ 
            << " "                << std::setw(20) << ( expect ? " expected " : " NOT-EXPECTED " )
         //   << glm::to_string(q.i) 
            << std::endl 
            ; 
    //assert(expect); 
} 

void test_OpticksPhotonFlags()
{
    LOG(info); 
   
    std::vector<int> boundary = { -0x7fff-2, -0x7fff-1, -0x7fff, -0x1000, -0x100, -0x10, -0x1, 0x1, 0x10, 0x100, 0x1000, 0x7fff, 0x7fff+1 } ;     

    unsigned sensorIndex = 0xffff ; 
    unsigned nodeIndex = 10000001 ; // s_identity_x
    unsigned photonIndex = 99999 ; 
    unsigned flagMask = 0x1 | 0x2 | 0x4 | 0x8 ;
    bool dump = true ; 

    for(unsigned i=0 ; i < boundary.size() ; i++) test_OpticksPhotonFlags(boundary[i], sensorIndex, nodeIndex, photonIndex, flagMask, dump ); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_OpticksPhotonFlags(); 

    return 0 ; 
}
// om-;TEST=OpticksPhotonFlagsTest om-t
