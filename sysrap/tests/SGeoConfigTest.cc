/**

::

    EMM=0,63 SGeoConfigTest

**/


#include "OPTICKS_LOG.hh"
#include "SGeoConfig.hh"
#include "SName.h"

void test_Arglist()
{
    std::vector<std::string>* arglist = SGeoConfig::Arglist() ; 
    if(arglist == nullptr) return ; 
    LOG(info) << " SGeoConfig::Arglist " << arglist->size() ; 
    for(unsigned i=0 ; i < arglist->size() ; i++) std::cout << (*arglist)[i] << std::endl ; 
}

void test_CXSkipLV(const SName* id)
{
    if(SGeoConfig::CXSkipLV() == nullptr) return ; 
    int num_name = id->getNumName() ; 
    LOG(info) << " num_name " << num_name ; 

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
}

void test_ELVSelection(const SName* id )
{
    const char* elv = SGeoConfig::ELVSelection(id) ; 
    std::cout 
        << " test_ELVSelection " 
        << std::endl 
        << " elv " << ( elv ? elv : "-" )
        << std::endl 
        ;

}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //const char* idp = "$CFBase/CSGFoundry/meshname.txt" ; 
    const char* idp = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/meshname.txt" ; 

    SName* id = SName::Load(idp); 
    std::cout << id->detail() << std::endl ; 

    /*
    SGeoConfig::GeometrySpecificSetup(id); 
    LOG(info) << SGeoConfig::Desc() ; 
    LOG(info) << SGeoConfig::DescEMM() ; 
    test_Arglist(); 
    test_CXSkipLV(id); 
    */

    test_ELVSelection(id); 

    return 0 ; 
}
