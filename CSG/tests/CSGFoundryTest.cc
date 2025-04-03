// ./CSGFoundryTest.sh

#include <iostream>
#include <cassert>
#include <csignal>

#include "scuda.h"
#include "sqat4.h"
#include "OPTICKS_LOG.hh"
#include "SSim.hh"

#include "ssys.h"
#include "spath.h"


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

    std::cout
       << " s0 " << ( s0 ? "Y" : "N" )
       << " s1 " << ( s1 ? "Y" : "N" )
       << " s2 " << ( s2 ? "Y" : "N" )
       << " s3 " << ( s3 ? "Y" : "N" )
       << std::endl
       ;

    fd.dump();

    assert( fd.getSolidIdx(s0) == 0 );
    assert( fd.getSolidIdx(s1) == 1 );
    assert( fd.getSolidIdx(s2) == 2 );
    assert( fd.getSolidIdx(s3) == 3 );

    fd.save("/tmp", "FoundryTest_" );
}

void test_PrimSpec()
{
    CSGFoundry fd ;
    fd.maker->makeDemoSolids();
    for(unsigned i = 0 ; i < fd.solid.size() ; i++ )
    {
        unsigned solidIdx = i ;
        std::cout << "solidIdx " << solidIdx << std::endl ;
        SCSGPrimSpec ps = fd.getPrimSpec(solidIdx);
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
    fd.addInstancePlaceholder() ;  // Loading fails when no inst


    const char* dir = spath::Resolve("$TMP/CSGFoundryTest/test_Load") ;
    const char* rel = "CSGFoundry" ;
    fd.save(dir, rel );

    CSGFoundry* fdl = CSGFoundry::Load(dir, rel);
    fdl->dump();

    int cmp = CSGFoundry::Compare(&fd, fdl);
    std::cout << "test_Load " << cmp << std::endl ;

    std::cout << "fd.desc()  " << fd.desc() << std::endl ;
    std::cout << "fdl->desc() " << fdl->desc() << std::endl ;

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

void test_getInstance_with_GAS_ordinal()
{
    CSGFoundry fd ;
    fd.maker->makeDemoGrid();
    LOG(info) << fd.descGAS() ;

    unsigned gas_idx = fd.getNumSolid()/2 ;
    unsigned ordinal = 0 ;

    const qat4* q = fd.getInstance_with_GAS_ordinal(gas_idx, ordinal);

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


    bool i0_expect = i0 == i0_ ;
    bool i1_expect = i1 == i1_ ;
    bool u_expect = u == u_ ;
    bool f_expect = f == f_ ;
    bool d_expect = d == d_ ;
    bool s0_expect = strcmp(s0.c_str(), s0_.c_str()) == 0  ;
    bool s1_expect = strcmp(s1.c_str(), s1_.c_str()) == 0  ;

    assert( i0_expect );
    assert( i1_expect );
    assert( u_expect );
    assert( f_expect );
    assert( d_expect );
    assert( s0_expect );
    assert( s1_expect );

    if(!i0_expect) std::raise(SIGINT);
    if(!i1_expect) std::raise(SIGINT);
    if(!u_expect) std::raise(SIGINT);
    if(!f_expect) std::raise(SIGINT);
    if(!d_expect) std::raise(SIGINT);
    if(!s0_expect) std::raise(SIGINT);
    if(!s1_expect) std::raise(SIGINT);

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


void test_getPrimName()
{
    SSim::Create();

    CSGFoundry* fd = CSGFoundry::Load() ;
    std::vector<std::string> pname ;
    fd->getPrimName(pname);

    LOG(info) << " pname.size " << pname.size() ;
    for(size_t i=0 ; i < pname.size() ; i++)
        std::cout << std::setw(6) << i << " : " << pname[i] << "\n" ;

    LOG(info) << " pname.size " << pname.size() ;

}


void test_Load_Save()
{
    LOG(info) << "[" ;
    SSim::Create();
    CSGFoundry* fd = CSGFoundry::Load() ;

    //const char* opt = "meshname,prim" ;
    //fd->setSaveOpt(opt);
    //LOG(info) << " opt[" << opt << "]" ;

    const char* out = "$TMP/CSG/CSGFoundryTest/test_Load_Save";
    fd->save(out);
    LOG(info) << " out[" << out << "]" ;

    LOG(info) << "]" ;
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //const char* DEF = "ALL" ;
    //const char* DEF = "Load_Save" ;
    const char* DEF = "getPrimName" ;
    const char* TEST = ssys::getenvvar("TEST", DEF) ;
    bool ALL = TEST && strcmp(TEST, "ALL") == 0 ;

    if(ALL||strcmp(TEST,"layered")==0) test_layered();
    if(ALL||strcmp(TEST,"PrimSpec")==0) test_PrimSpec();
    if(ALL||strcmp(TEST,"addTran")==0) test_addTran();
    if(ALL||strcmp(TEST,"makeClustered")==0) test_makeClustered();
    if(ALL||strcmp(TEST,"Compare")==0) test_Compare();
    if(ALL||strcmp(TEST,"Load")==0) test_Load();
    if(ALL||strcmp(TEST,"getInstanceTransformsGAS")==0) test_getInstanceTransformsGAS();
    if(ALL||strcmp(TEST,"getInstance_with_GAS_ordinal")==0) test_getInstance_with_GAS_ordinal();
    if(ALL||strcmp(TEST,"setMeta_getMeta")==0) test_setMeta_getMeta();
    if(ALL||strcmp(TEST,"getPrimName")==0) test_getPrimName();
    if(ALL||strcmp(TEST,"Load_Save")==0) test_Load_Save();


    return 0 ;
}
