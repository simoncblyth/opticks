#include <iomanip>
#include "OPTICKS_LOG.hh"
#include "NGLM.hpp"
#include "NPY.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NPY<float>* a = NPY<float>::make(7, 4, 4); 
    a->zero(); 
    glm::mat4 m(1.f); 

    int ni = a->getShape(0) ; 
    for(int i=0 ; i < ni ; i++ ) 
    {
        a->setMat4(m, i, -1);  
        a->setUInt(i, -1, -1, 0, 100*i );
        a->setUSign(i, -1, -1, 0, i % 2 == 0 ); 
    }

    bool preserve_zero = true ; 
    bool preserve_signbit = true ; 
    a->addOffset( -1, -1, 10000, preserve_zero, preserve_signbit ) ; 


    for(int i=0 ; i < ni ; i++ ) 
    {
        unsigned u = a->getUInt(i, -1, -1, 0); 
        bool sign = a->getUSign(i, -1, -1, 0 ); 
        bool xsign = i % 2 == 0 ; 
        assert( sign == xsign ); 

        std::cout 
            << " i " << std::setw(2) << i  
            << " u " << std::setw(10) << u  
            << " u & NOTSIGNBIT  " << std::setw(10) << ( u & NPYBase::NOTSIGNBIT )  
            << " sign " << sign
            << std::endl 
           ; 
    }


    a->dump(); 
    NPY<unsigned>* b = (NPY<unsigned>*)a ; 
    b->dump(); 

    a->save("$TMP/NPYaddOffsetTest.npy"); 

    return 0 ; 
}
