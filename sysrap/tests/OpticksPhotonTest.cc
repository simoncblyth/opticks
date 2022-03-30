// om-;TEST=OpticksPhotonTest om-t    ## faster building when just making changes to this test  
#include <cstring>
#include "OPTICKS_LOG.hh"
#include "SStr.hh"
#include "OpticksPhoton.h"
#include "OpticksPhoton.hh"


void test_FlagAbbrevPairs()
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
}


void test_FlagMask_0()
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
}

void test_FlagMask_1()
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
}

void test_AbbrevToFlag()
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
    assert( flag_non_existing == 0 );
    unsigned flag_NULL = OpticksPhoton::AbbrevToFlag(NULL) ; 
    assert( flag_NULL == 0 );


}

void test_AbbrevToFlagSequence(const char* abbseq)
{
    unsigned long long seqhis = OpticksPhoton::AbbrevToFlagSequence(abbseq);
    std::string abbseq2 = OpticksPhoton::FlagSequence( seqhis ); 
    const char* abbseq2_ = SStr::TrimTrailing(abbseq2.c_str());

    bool match = strcmp( abbseq2_ , abbseq) == 0 ;

    LOG(match ? info : fatal) 
           << " abbseq [" << abbseq << "]"
           << " seqhis " << std::setw(16) << std::hex << seqhis << std::dec 
           << " abbseq2 [" << abbseq2_ << "]"
           ;
            
    assert(match);
}

void test_AbbrevToFlagSequence()
{
    LOG(info); 
    test_AbbrevToFlagSequence("TO SR SA");
    test_AbbrevToFlagSequence("TO SC SR SA");
    //test_AbbrevToFlagSequence("TO ZZ SC SR SA");
}



void test_AbbrevToFlagValSequence(const char* seqmap, const char* x_abbseq, unsigned long long x_seqval)
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
}


void test_AbbrevToFlagValSequence()
{
    LOG(info); 
    test_AbbrevToFlagValSequence("TO:0 SR:1 SA:0", "TO SR SA", 0x121ull );

    //test_AbbrevToFlagValSequence("TO:0 SC: SR:1 SA:0", "TO SC SR SA", 0x1201ull );   // migrated imp does like empties
    //test_AbbrevToFlagValSequence("TO:0 SC:0 SR:1 SA:0", "TO SC SR SA", 0x1201ull );  // but this fails seqval match
}

void test_PointAbbrev()
{
    LOG(info); 
    unsigned long long seqhis = 0x4ad ;  
    for(unsigned p=0 ; p < 5 ; p++) LOG(info) << p << " " << OpticksPhoton::PointAbbrev(seqhis, p ) ; 
}
void test_PointVal1()
{
    LOG(info); 
    unsigned long long seqval = 0x121 ;  
    for(unsigned p=0 ; p < 5 ; p++) LOG(info) << p << " " << OpticksPhoton::PointVal1(seqval, p ) ; 
}

void test_AbbrevSequenceToMask()
{
    LOG(info); 
    const char* abrseq = "TO SD BT" ; 
    unsigned x_mask = TORCH | SURFACE_DETECT | BOUNDARY_TRANSMIT ;       
    char delim = ' ' ; 
    unsigned mask = OpticksPhoton::AbbrevSequenceToMask( abrseq, delim ); 

    assert( mask == x_mask ) ; 

    LOG(info) 
          << " abrseq " << abrseq 
          << " mask " << mask 
          ;
}







int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) << " sysrap.OpticksPhotonTest " ; 

    /*
    test_FlagAbbrevPairs(); 
    test_FlagMask_0();
    test_FlagMask_1();
    test_AbbrevToFlag();
    test_AbbrevToFlagSequence();
    test_AbbrevToFlagValSequence();
    test_PointAbbrev();
    test_PointVal1();
    */
    test_AbbrevSequenceToMask(); 
    /*
    */


    return 0 ; 
}
