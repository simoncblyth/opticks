#include "NPY.hpp"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
/**


See also::

    ImageNPYConcatTest.cc
    ImageNPYConcatTest.py

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* ap = "$TMP/SPPMTest_MakeTestImage_old_concat.npy" ; 
    const char* bp = "$TMP/SPPMTest_MakeTestImage_new_concat.npy" ; 

    NPY<unsigned char>* a = NPY<unsigned char>::load(ap) ; 
    NPY<unsigned char>* b = NPY<unsigned char>::load(bp) ; 

    if( a == nullptr || b == nullptr ) return 0 ; 

    unsigned a_nv = a->getNumValues(1);  
    unsigned b_nv = b->getNumValues(1);  

    LOG(info) << " a " << a->getShapeString() << " a_nv " << a_nv ; 
    LOG(info) << " b " << b->getShapeString() << " b_nv " << b_nv ; 

    assert( a_nv == b_nv ); 
    unsigned nv = a_nv ; 


    bool dump = false ;   // dumping causes funny char problems with opticks-tl
    unsigned char epsilon = 0 ; 
    unsigned dumplimit = 100 ; 
    char mode = 'I' ;     

    unsigned diffs = NPY<unsigned char>::compare(a,b,epsilon, dump, dumplimit, mode);  
    LOG(info) << " diffs " << diffs ; 

    unsigned i = 0 ; 
    const unsigned char* av = a->getValuesConst(i, 0); 
    const unsigned char* bv = b->getValuesConst(i, 0); 

    for(unsigned v=0 ; v < nv ; v++)
    {
        unsigned char df = av[v] - bv[v] ;   
        if( df < 0 ) df = -df ;   // dodgy for unsigned 

        if( v % 10000 == 0 ) std::cout 
             << " v " << std::setw(6) << v 
             << " av[v] " << std::setw(6) << (int)av[v] 
             << " bv[v] " << std::setw(6) << (int)bv[v] 
             << " df " << std::setw(6) << (int)df
             << std::endl 
             ;
    }
    return 0 ; 
}

