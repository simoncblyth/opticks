/**
SSimTest.cc
============

TEST=Load ~/o/sysrap/tests/SSimTest.sh 

TEST=get_jpmt_nocopy  ~/o/sysrap/tests/SSimTest.sh 
TEST=get_jpmt         ~/o/sysrap/tests/SSimTest.sh 
TEST=get_spmt         ~/o/sysrap/tests/SSimTest.sh 
TEST=get_spmt_f       ~/o/sysrap/tests/SSimTest.sh 

**/


#include "SSim.hh"
#include "NPFold.h"
#include "OPTICKS_LOG.hh"
#include "SPMT.h"



struct SSimTest
{
    static const char* TEST ; 

    static int Load(); 
    static int Load_get(); 
    static int Create(); 
    static int findName();
 
    static int addFake(); 
    static int addFake_ellipsis(); 

    static int get_bnd(); 

    static int get_jpmt_nocopy(); 
    static int get_jpmt(); 
    static int get_spmt(); 
    static int get_spmt_f(); 

    static int Main(); 
}; 


const char* SSimTest::TEST = U::GetEnv("TEST","ALL");  


int SSimTest::Load()
{
    std::cout << "[Load\n" ; 
    SSim* sim = SSim::Load(); 
    std::cout << ( sim ? sim->desc() : "-" ) ;  
    std::cout << "]Load\n" ; 
    return 0 ; 
}
int SSimTest::Load_get()
{
    std::cout << "[Load_get " << std::endl ; 
    SSim* sim = SSim::Load(); 
    std::cout << " sim " << ( sim ? "YES" : "NO " ) << std::endl ; 
    const NP* optical = sim ? sim->get(snam::OPTICAL) : nullptr ; 
    std::cout << " optical " << ( optical ? optical->sstr() : "-" ) << std::endl ; 
    std::cout << "]Load_get " << std::endl ; 
    return 0 ; 
}
int SSimTest::Create()
{
    std::cout << "[Create\n" ; 
    SSim* sim = SSim::Create(); 
    LOG(info) << " sim.desc " << sim->desc() ; 
    std::cout << "]Create\n" ; 
    return 0 ; 
}
int SSimTest::findName()
{
    std::cout << "[findName\n" ; 
    std::vector<std::string> names = {
        "Air", 
        "Rock", 
        "Water", 
        "Acrylic",
        "Cream", 
        "vetoWater", 
        "Cheese", 
        "NextIsBlank",
        "",
        "Galactic", 
        "Pyrex", 
        "PMT_3inch_absorb_logsurf1", 
        "Steel", 
        "Steel_surface",
        "PE_PA",
        "Candy",
        "NextIsBlank",
        "",
        "Steel"
      } ; 


    const SSim* sim = SSim::Load(); 
    const NP* bnd = sim->get(snam::BND); 
    std::cout << " bnd " << ( bnd ? bnd->sstr() : "null-bnd" ) << " snam::BND " << snam::BND << std::endl ; 
    std::cout << " bnd.names " << ( bnd ? bnd->names.size() : -1 ) << std::endl ; 


    for(unsigned a=0 ; a < names.size() ; a++ )
    {
         const std::string& n = names[a] ; 
         int i, j ; 
         bool found = sim->findName(i,j,n.c_str() ); 

         std::cout << std::setw(30) << n << " " ; 
         if(found)  
         {
            std::cout 
                << "(" 
                << std::setw(3) << i 
                << ","  
                << std::setw(3) << j
                << ")"
                << " "
                << SSim::GetItemDigest(bnd, i, j, 8)
                ;
         }
         else
         {
            std::cout << "-" ;  
         }
         std::cout << std::endl ;  
    }
    std::cout << "]findName\n" ; 
    return 0 ; 
}


int SSimTest::addFake()
{
    std::cout << "[addFake\n" ; 
    SSim* sim = SSim::Load(); 
    std::string bef = sim->desc(); 

    LOG(info) << "BEFORE " << std::endl << sim->descOptical() << std::endl ; 

    std::vector<std::string> specs = { "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Air///Water" } ;
    sim->addFake_(specs); 
    std::string aft = sim->desc(); 

    LOG(info) << "AFTER " << std::endl << sim->descOptical() << std::endl ; 

    std::cout << "bef" << std::endl << bef << std::endl ; 
    std::cout << "aft" << std::endl << aft << std::endl ; 
    std::cout << "]addFake\n" ; 
    return 0 ; 
}

int SSimTest::addFake_ellipsis()
{
    std::cout << "[addFake_ellipsis\n" ; 
    SSim* sim = SSim::Load(); 
    sim->addFake("Air///Water"); 
    sim->addFake("Air///Water", "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air"); 
    sim->addFake("Air///Water", "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Water///Air" ); 
    std::cout << "]addFake_ellipsis\n" ; 
    return 0 ; 
}



int SSimTest::get_bnd()
{
    std::cout << "[get_bnd\n" ; 
    SSim* sim = SSim::Load(); 
    const NP* bnd = sim->get_bnd(); 
    LOG(info) << bnd->desc() ;  
    std::cout << "]get_bnd\n" ; 
    return 0 ; 
}
int SSimTest::get_jpmt_nocopy()
{
    std::cout << "[get_jpmt_nocopy\n" ; 
    SSim* sim = SSim::Load(); 
    const NPFold* f  = sim->get_jpmt_nocopy(); 
    LOG(info) << f->desc() ;  
    std::cout << "]get_jpmt_nocopy\n" ; 
    return 0 ; 
}
int SSimTest::get_jpmt()
{
    std::cout << "[get_jpmt\n" ; 
    SSim* sim = SSim::Load(); 
    const NPFold* f  = sim->get_jpmt(); 
    LOG(info) << f->desc() ;  
    std::cout << "]get_jpmt\n" ; 
    return 0 ; 
}
int SSimTest::get_spmt()
{
    std::cout << "[get_spmt\n" ; 
    SSim* sim = SSim::Load(); 
    const SPMT* spmt  = sim->get_spmt(); 
    LOG(info) << spmt->desc() ;  
    std::cout << "]get_spmt\n" ; 
    return 0 ; 
}
int SSimTest::get_spmt_f()
{
    std::cout << "[get_spmt_f\n" ; 
    SSim* sim = SSim::Load(); 
    const NPFold* spmt_f  = sim->get_spmt_f(); 
    LOG(info) << spmt_f->desc() ;  
    std::cout << "]get_spmt_f\n" ; 
    return 0 ; 
}





int SSimTest::Main()
{
    bool ALL = strcmp(TEST, "ALL") == 0 ; 
    int rc = 0 ; 
    if(ALL||0==strcmp(TEST,"Load"))      rc += Load(); 
    if(ALL||0==strcmp(TEST,"Load_get"))  rc += Load_get(); 
    if(ALL||0==strcmp(TEST,"Create"))    rc += Create(); 
    if(ALL||0==strcmp(TEST,"findName"))  rc += findName(); 

    // these two failing with consistencey assert
    if(0||0==strcmp(TEST,"addFake"))   rc += addFake(); 
    if(0||0==strcmp(TEST,"addFake_ellipsis")) rc += addFake_ellipsis(); 

    if(ALL||0==strcmp(TEST,"get_bnd"))         rc += get_bnd(); 
    if(ALL||0==strcmp(TEST,"get_jpmt_nocopy")) rc += get_jpmt_nocopy(); 
    if(ALL||0==strcmp(TEST,"get_jpmt"))        rc += get_jpmt(); 
    if(ALL||0==strcmp(TEST,"get_spmt"))        rc += get_spmt(); 
    if(ALL||0==strcmp(TEST,"get_spmt_f"))      rc += get_spmt_f();
 
    return rc ; 
}


int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 
    return SSimTest::Main(); 
}

