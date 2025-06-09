// om-;TEST=OpticksPhotonTest om-t    ## faster building when just making changes to this test
#include <cstring>
#include <csignal>
#include "OPTICKS_LOG.hh"
#include "sstr.h"
#include "ssys.h"
#include "spath.h"
#include "NPX.h"

#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"


int test_FlagAbbrevPairs()
{
    LOG(info);
    typedef std::pair<const char*, const char*> KV ;
    std::vector<KV> pairs ;

    OpticksPhoton::FlagAbbrevPairs(pairs);

    for(unsigned i=0 ; i < pairs.size() ; i++)
    {
        const KV& kv = pairs[i] ;
        std::cout
            << std::setw(4) << kv.second
            << std::setw(20) << kv.first
            << std::endl
            ;
    }
    return 0 ;
}


int test_FlagMask_0()
{
    LOG(info);
    for(unsigned i=0 ; i < 16 ; i++)
    {
        unsigned msk = 0x1 << i ;
        std::cout
                  << " ( 0x1 << " << std::setw(2) << i << " ) "
                  << " (i+1) " << std::setw(2) << std::hex << (i + 1) << std::dec
                  << " " << std::setw(2)  << OpticksPhoton::FlagMask(msk, true)
                  << " " << std::setw(20) << OpticksPhoton::FlagMask(msk, false)
                  << " " << std::setw(6) << std::hex << msk << std::dec
                  << " " << std::setw(6) << std::dec << msk << std::dec
                  << std::endl
                  ;

    }
    return 0 ;
}

int test_FlagMask_1()
{
    LOG(info);
    std::vector<unsigned> vmsk = { 0x5840, 0x5850, 0x5c40, 0x5940, 0x5860,  } ;

    for(unsigned i=0 ; i < vmsk.size() ; i++)
    {
        unsigned msk = vmsk[i] ;
        std::cout
            << std::setw(10) << std::hex << msk << std::dec
            << " flagmask(abbrev) " << std::setw(20) << OpticksPhoton::FlagMask(msk, true)
            << " flagmask " << OpticksPhoton::FlagMask(msk, false)
            << std::endl
            ;
    }
    return 0 ;
}

int test_GetHitMask()
{
    std::cout << "[GetHitMask\n" ;

    char delim = ',' ;
    std::vector<std::string> msks = { "SD", "EC", "EX", "EC,SD", "EX,SD", "EC,SA" } ;

    int num_msk = msks.size();
    for(int i=0 ; i < num_msk ; i++)
    {
        const char* _msk = msks[i].c_str();
        unsigned msk = OpticksPhoton::GetHitMask(_msk, delim);

        std::cout
            << " _msk " << std::setw(10) << _msk
            << " msk  " << std::setw(10) << std::hex << msk << std::dec
            << " " << OpticksPhoton::FlagMask(msk, true )
            << " " << OpticksPhoton::FlagMask(msk, false )
            << "\n"
            ;
    }
    std::cout << "]GetHitMask\n" ;
    return 0 ;
}



int test_AbbrevToFlag()
{
    LOG(info);
    for(unsigned f=0 ; f < 32 ; f++ )
    {
        unsigned flag = OpticksPhoton::EnumFlag(f);
        const char* abbrev = OpticksPhoton::Abbrev(flag) ;
        unsigned flag2 = OpticksPhoton::AbbrevToFlag( abbrev );
        unsigned f2 = OpticksPhoton::BitPos(flag2) ;
        bool bad_flag = strcmp(abbrev, OpticksPhoton::_BAD_FLAG) == 0 ;

        std::cout
              << " f " << std::setw(4) << f
              << " flag EnumFlag(f) " << std::setw(10) << flag
              << " abbrev Abbrev(flag)   " << std::setw(3) << abbrev
              << " flag2 AbbrevToFlag( abbrev ) " << std::setw(10) << flag2
              << " f2 BitPos(flag2) " << std::setw(4) << f2
              << " bad_flag " << bad_flag
              << std::endl
              ;

        if(bad_flag) break ;   // only the last so not continue
        assert( flag2 == flag );
        assert( f2 == f );
    }


    unsigned flag_non_existing = OpticksPhoton::AbbrevToFlag("ZZ") ;
    bool flag_non_existing_expect = flag_non_existing == 0 ;
    assert( flag_non_existing_expect );
    if(!flag_non_existing_expect ) std::raise(SIGINT);

    unsigned flag_NULL = OpticksPhoton::AbbrevToFlag(NULL) ;
    bool flag_NULL_expect = flag_NULL == 0 ;
    assert( flag_NULL_expect );
    if(!flag_NULL_expect) std::raise(SIGINT);


    return 0 ;
}

int test_AbbrevToFlagSequence(const char* abbseq)
{
    unsigned long long seqhis = OpticksPhoton::AbbrevToFlagSequence(abbseq);
    std::string abbseq2 = OpticksPhoton::FlagSequence( seqhis );
    const char* abbseq2_ = sstr::TrimTrailing(abbseq2.c_str());

    bool match = strcmp( abbseq2_ , abbseq) == 0 ;

    LOG(match ? info : fatal)
           << " abbseq [" << abbseq << "]"
           << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec
           << " abbseq2 [" << abbseq2_ << "]"
           ;

    assert(match);
    return 0 ;
}

int test_AbbrevToFlagSequence()
{
    LOG(info);
    test_AbbrevToFlagSequence("TO SR SA");
    test_AbbrevToFlagSequence("TO SC SR SA");
    //test_AbbrevToFlagSequence("TO ZZ SC SR SA");
    return 0 ;
}



int test_AbbrevToFlagValSequence(const char* seqmap, const char* x_abbseq, unsigned long long x_seqval)
{
    unsigned long long seqhis(0ull) ;
    unsigned long long seqval(0ull) ;

    OpticksPhoton::AbbrevToFlagValSequence(seqhis, seqval, seqmap );

    unsigned long long x_seqhis = OpticksPhoton::AbbrevToFlagSequence(x_abbseq) ;

    bool seqhis_match = seqhis == x_seqhis  ;
    bool seqval_match = seqval == x_seqval  ;

    LOG( seqhis_match ? info : fatal )
            << " seqmap " << std::setw(20) << seqmap
            << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec
            << " x_seqhis " << std::setw(16) << std::hex << x_seqhis << std::dec
            << " x_abbseq " << x_abbseq
            ;

    LOG( seqval_match ? info : fatal )
            << " seqmap " << std::setw(20) << seqmap
            << " seqval " << std::setw(16) << std::hex << seqval << std::dec
            << " x_seqval " << std::setw(16) << std::hex << x_seqval << std::dec
            ;

    assert( seqhis_match ) ;
    assert( seqval_match ) ;
    return 0 ;
}


int test_AbbrevToFlagValSequence()
{
    LOG(info);
    test_AbbrevToFlagValSequence("TO:0 SR:1 SA:0", "TO SR SA", 0x121ull );

    //test_AbbrevToFlagValSequence("TO:0 SC: SR:1 SA:0", "TO SC SR SA", 0x1201ull );   // migrated imp does like empties
    //test_AbbrevToFlagValSequence("TO:0 SC:0 SR:1 SA:0", "TO SC SR SA", 0x1201ull );  // but this fails seqval match
    return 0 ;
}

int test_PointAbbrev()
{
    LOG(info);
    unsigned long long seqhis = 0x4ad ;
    for(unsigned p=0 ; p < 5 ; p++) LOG(info) << p << " " << OpticksPhoton::PointAbbrev(seqhis, p ) ;
    return 0 ;
}
int test_PointVal1()
{
    LOG(info);
    unsigned long long seqval = 0x121 ;
    for(unsigned p=0 ; p < 5 ; p++) LOG(info) << p << " " << OpticksPhoton::PointVal1(seqval, p ) ;
    return 0 ;
}

int test_AbbrevSequenceToMask()
{
    LOG(info);
    const char* abrseq = "TO,SD,BT" ;
    unsigned x_mask = TORCH | SURFACE_DETECT | BOUNDARY_TRANSMIT ;
    char delim = ',' ;
    unsigned mask = OpticksPhoton::AbbrevSequenceToMask( abrseq, delim );

    bool mask_expect =  mask == x_mask ;
    assert( mask_expect ) ;
    if(!mask_expect) std::raise(SIGINT);

    LOG(info)
          << " abrseq " << abrseq
          << " mask " << mask
          ;
    return 0 ;
}



int test_Abbrev_Flag()
{
    LOG(info);
    unsigned lastBit = 17 ;
    for(unsigned n=0 ; n <= lastBit ; n++ )
    {
        unsigned flag = 0x1 << n ;
        std::cout
            << " n " << std::setw(10) << n
            << " (0x1 << n) " << std::setw(10) << flag
            << " OpticksPhoton::Flag " << std::setw(20) << OpticksPhoton::Flag(flag)
            << " OpticksPhoton::Abbrev " << std::setw(20) << OpticksPhoton::Abbrev(flag)
            << std::endl
            ;
    }

    return 0 ;
}

int test_load_seq()
{
    const char* _path = "$TMP/GEOM/$GEOM/G4CXTest/ALL0/p001/seq.npy" ;
    const char* path = spath::Resolve(_path) ;
    NP* a = NP::LoadIfExists(path);
    std::cout
        << "OpticksPhotonTest:test_load_seq"
        << std::endl
        << " _path " << _path
        << std::endl
        << " path  " << path
        << std::endl
        << " a " << ( a ? a->sstr() : "-" )
        << std::endl
         ;
    if(a == nullptr) return 0 ;

    const uint64_t* aa = a->cvalues<uint64_t>();
    int num = a->shape[0] ;
    int edge = 10 ;

    for(int i=0 ; i < num ; i++)
    {
        if( i < edge || i > (num - edge) )
            std::cout << OpticksPhoton::FlagSequence_( aa + 4*i, 2 ) << std::endl ;
        else if( i == edge )
            std::cout << "..." << std::endl ;
    }
    return 0 ;
}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    LOG(info) << " sysrap.OpticksPhotonTest " ;

    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;

    if(ALL||0==strcmp(TEST, "FlagAbbrevPairs"))         rc += test_FlagAbbrevPairs() ;
    if(ALL||0==strcmp(TEST, "FlagMask_0"))              rc += test_FlagMask_0() ;
    if(ALL||0==strcmp(TEST, "FlagMask_1"))              rc += test_FlagMask_1() ;
    if(ALL||0==strcmp(TEST, "GetHitMask"))              rc += test_GetHitMask() ;
    if(ALL||0==strcmp(TEST, "AbbrevToFlag"))            rc += test_AbbrevToFlag() ;
    if(ALL||0==strcmp(TEST, "AbbrevToFlagSequence"))    rc += test_AbbrevToFlagSequence() ;
    if(ALL||0==strcmp(TEST, "AbbrevToFlagValSequence")) rc += test_AbbrevToFlagValSequence() ;
    if(ALL||0==strcmp(TEST, "PointAbbrev"))             rc += test_PointAbbrev() ;
    if(ALL||0==strcmp(TEST, "PointVal1"))               rc += test_PointVal1() ;
    if(ALL||0==strcmp(TEST, "AbbrevSequenceToMask"))    rc += test_AbbrevSequenceToMask() ;
    if(ALL||0==strcmp(TEST, "Abbrev_Flag"))             rc += test_Abbrev_Flag() ;
    if(ALL||0==strcmp(TEST, "load_seq"))                rc += test_load_seq() ;

    return rc ;
}
