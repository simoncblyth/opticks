// ./CSGFoundryTest.sh

#include <iostream>
#include <cassert>

#include "scuda.h"
#include "sqat4.h"
#include "OPTICKS_LOG.hh"

#include "CSGFoundry.h"
#include "CSGMaker.h"



void test_layered()
{
    CSGFoundry fd ;  
    CSGMaker* mk = fd.maker ; 

    CSGSolid* s0 = mk->makeLayered("sphere", 100.f, 10 ); 
    CSGSolid* s1 = mk->makeLayered("sphere", 1000.f, 10 ); 
    CSGSolid* s2 = mk->makeLayered("sphere", 50.f, 5 ); 
    CSGSolid* s3 = mk->makeSphere() ; 

    fd.dump(); 

    assert( fd.getSolidIdx(s0) == 0 ); 
    assert( fd.getSolidIdx(s1) == 1 ); 
    assert( fd.getSolidIdx(s2) == 2 ); 
    assert( fd.getSolidIdx(s3) == 3 ); 

    fd.write("/tmp", "FoundryTest_" ); 
}

void test_PrimSpec()
{
    CSGFoundry fd ; 
    fd.maker->makeDemoSolids(); 
    for(unsigned i = 0 ; i < fd.solid.size() ; i++ )
    {
        unsigned solidIdx = i ; 
        std::cout << "solidIdx " << solidIdx << std::endl ; 
        CSGPrimSpec ps = fd.getPrimSpec(solidIdx);
        ps.dump(""); 
    }

    std::string fdd = fd.desc(); 
    std::cout << fdd << std::endl ; 
}

void test_addTran()
{
    CSGFoundry fd ; 
    const Tran<double>* tr = Tran<double>::make_translate( 100., 200., 300. ) ; 
    unsigned idx = fd.addTran( tr );   // this idx is 0-based 
    std::cout << "test_addTran idx " << idx << std::endl ; 
    assert( idx == 0u );   
    const qat4* t = fd.getTran(idx) ; 
    const qat4* v = fd.getItra(idx) ; 
 
    std::cout << "idx " << idx << std::endl ; 
    std::cout << "t" << *t << std::endl ; 
    std::cout << "v" << *v << std::endl ; 
}

void test_makeClustered()
{
    std::cout << "[test_makeClustered" << std::endl ; 
    CSGFoundry fd ; 
    bool inbox = false ; 
    fd.maker->makeClustered("sphe", -1,2,1, -1,2,1, -1,2,1, 1000., inbox ); 
    fd.dumpPrim(0); 
    std::cout << "]test_makeClustered" << std::endl ; 
}

void test_Load()
{
    CSGFoundry fd ; 
    fd.maker->makeDemoSolids(); 

    const char* dir = "/tmp" ; 
    const char* rel = "CSGFoundryTestLoad" ; 
    fd.write(dir, rel ); 
 
    CSGFoundry* fdl = CSGFoundry::Load(dir, rel); 
    fdl->dump(); 

    int cmp = CSGFoundry::Compare(&fd, fdl); 
    std::cout << "test_Load " << cmp << std::endl ; 
}

void test_Compare()
{
    CSGFoundry fd ; 
    fd.maker->makeDemoSolids(); 

    int cmp = CSGFoundry::Compare(&fd, &fd); 
    std::cout << "test_Compare " << cmp << std::endl ; 
}

void test_getInstanceTransformsGAS()
{
    CSGFoundry fd ; 
    fd.maker->makeDemoGrid(); 
    fd.inst_find_unique(); 
    LOG(info) << fd.descGAS() ; 

    unsigned gas_idx = fd.getNumSolid()/2 ; 

    std::vector<qat4> sel ;  
    fd.getInstanceTransformsGAS(sel, gas_idx ); 

    LOG(info) 
        << " gas_idx " << gas_idx 
        << " sel.size " << sel.size()
        ; 

    qat4::dump(sel); 

}

void test_getInstanceGAS()
{
    CSGFoundry fd ; 
    fd.maker->makeDemoGrid(); 
    LOG(info) << fd.descGAS() ; 

    unsigned gas_idx = fd.getNumSolid()/2 ; 
    unsigned ordinal = 0 ; 

    const qat4* q = fd.getInstanceGAS(gas_idx, ordinal); 

    assert(q) ; 
    LOG(info) << *q ; 
}

void test_setMeta_getMeta()
{
    LOG(info) ; 

    CSGFoundry fd ; 

    int i0 = -101 ; 
    int i1 = -1010 ; 
    unsigned u = 202 ; 
    float f = 42.f ; 
    double d = 420. ; 
    std::string s0 = "string0" ; 
    std::string s1 = "string1" ; 

    fd.setMeta("i0", i0); 
    fd.setMeta("i1", i1); 
    fd.setMeta("u", u);
    fd.setMeta("f", f );  
    fd.setMeta("d", d );
    fd.setMeta("s0", s0 );
    fd.setMeta("s1", s1 );
  
    int i0_ = fd.getMeta("i0", 0); 
    int i1_ = fd.getMeta("i1", 0); 
    unsigned u_ = fd.getMeta("u", 0u); 
    float f_ = fd.getMeta("f", 0.f); 
    double d_ = fd.getMeta("d", 0.); 
    std::string  s0_ = fd.getMeta<std::string>("s0", ""); 
    std::string  s1_ = fd.getMeta<std::string>("s1", ""); 

    assert( i0 == i0_ ); 
    assert( i1 == i1_ ); 
    assert( u == u_ ); 
    assert( f == f_ ); 
    assert( d == d_ ); 
    assert( strcmp(s0.c_str(), s0_.c_str()) == 0 ); 
    assert( strcmp(s1.c_str(), s1_.c_str()) == 0 ); 
}

void test_setPrimBoundary()
{
    CSGFoundry* fd = CSGFoundry::Load() ; 
    unsigned numPrim = fd->getNumPrim(); 
    unsigned primIdx = numPrim - 1 ; 
    unsigned b0 = fd->getPrimBoundary(primIdx); 

    LOG(info) 
        << " numPrim " << numPrim
        << " primIdx " << primIdx 
        << " b0 " << b0 
#ifdef WITH_FOREIGN
        << " bndname " << fd->getBndName(b0) 
#endif
        ; 

    std::cout << fd->detailPrim(primIdx) << std::endl ; 

    fd->setPrimBoundary( primIdx, 0u );   
    std::cout << fd->detailPrim(primIdx) << std::endl ; 

#ifdef WITH_FOREIGN
    fd->setPrimBoundary(primIdx, "Water///Acrylic" ); 
    std::cout << fd->detailPrim(primIdx) << std::endl ; 
#endif

    fd->setPrimBoundary( primIdx, b0 );   
    std::cout << fd->detailPrim(primIdx) << std::endl ; 

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    /*
    test_layered(); 
    test_PrimSpec(); 
    test_addTran(); 
    test_makeClustered(); 
    test_Compare(); 
    test_Load(); 
    test_getInstanceTransformsGAS() ;
    test_getInstanceGAS() ;
    test_setMeta_getMeta(); 
    */

    test_setPrimBoundary(); 

    return 0 ; 
}
