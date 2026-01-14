/**

::

    EMM=0,63 SGeoConfigTest

    TEST=ELVSelection ~/o/sysrap/tests/SGeoConfigTest.sh

**/


#include "ssys.h"
#include "SGeoConfig.hh"
#include "SName.h"

struct SGeoConfigTest
{
    static int Arglist();
    static int CXSkipLV();
    static int EMM();

    static int ELVString();
    static int ELVSelection();
    static int ELV();

    static int Main();
};


inline int SGeoConfigTest::Arglist()
{
    std::vector<std::string>* arglist = SGeoConfig::Arglist() ;
    if(arglist == nullptr) return 0 ;
    std::cout << "[SGeoConfigTest::Arglist " << arglist->size() << "\n" ;
    for(unsigned i=0 ; i < arglist->size() ; i++) std::cout << (*arglist)[i] << std::endl ;
    std::cout << "]SGeoConfigTest::Arglist " << arglist->size() << "\n" ;
    return 0 ;
}

inline int SGeoConfigTest::CXSkipLV()
{
    if(SGeoConfig::CXSkipLV() == nullptr) return 0 ;
    SName* id = SName::GEOMLoad();
    int num_name = id->getNumName() ;
    std::cout << " num_name " << num_name << "\n" ;
    for(int i=0 ; i < num_name ; i++)
    {
        const char* name = id->getName(i);
        std::cout
            << std::setw(4) << i
            << " SGeoConfig::IsCXSKipLV " << SGeoConfig::IsCXSkipLV(i)
            << " name " << name
            << std::endl
            ;
    }
    return 0;
}

inline int SGeoConfigTest::EMM()
{
    std::cout << SGeoConfig::DescEMM() ;
    return 0;
}


inline int SGeoConfigTest::ELVString()
{
    SName* id = SName::GEOMLoad();
    const char* elv = SGeoConfig::ELVString(id) ;
    std::cout
        << "[SGeoConfigTest:ELVString\n"
        << " elv " << ( elv ? elv : "-" )
        << "]SGeoConfigTest:ELVString\n"
        ;

    return 0 ;
}



inline int SGeoConfigTest::ELVSelection()
{
    SName* id = SName::GEOMLoad();
    const char* elv = SGeoConfig::ELVSelection(id) ;
    std::cout
        << "[SGeoConfigTest:ELVSelection\n"
        << " elv [" << ( elv ? elv : "-" ) << "] "
        << "]SGeoConfigTest:ELVSelection\n"
        ;

    return 0 ;
}

inline int SGeoConfigTest::ELV()
{
    SName* id = SName::GEOMLoad();
    const SBitSet* elv = id ? SGeoConfig::ELV(id) : nullptr ;
    std::cout
         << "[SGeoConfigTest:ELV\n"
         << " id  " << ( id ? "YES" : "NO " )
         << " elv " << ( elv ? "YES" : "NO " )
         << "]SGeoConfigTest:ELV\n"
         ;

    return 0 ;
}

inline int SGeoConfigTest::Main()
{
    const char* TEST = ssys::getenvvar("TEST","ELV");
    bool ALL = 0 == strcmp("ALL", TEST) ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Arglist"))      rc += Arglist();
    if(ALL||0==strcmp(TEST,"CXSkipLV"))     rc += CXSkipLV();
    if(ALL||0==strcmp(TEST,"EMM"))          rc += EMM();
    if(ALL||0==strcmp(TEST,"ELVString"))    rc += ELVString();
    if(ALL||0==strcmp(TEST,"ELVSelection")) rc += ELVSelection();
    if(ALL||0==strcmp(TEST,"ELV"))          rc += ELV();

    return rc ;
}

int main(){ return SGeoConfigTest::Main(); }
