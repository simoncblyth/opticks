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


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SName* id = SName::Load("$CFBase/CSGFoundry/meshname.txt"); 
    std::cout << id->detail() << std::endl ; 

    SGeoConfig::GeometrySpecificSetup(id); 
    LOG(info) << SGeoConfig::Desc() ; 
    LOG(info) << SGeoConfig::DescEMM() ; 

    test_Arglist(); 
    test_CXSkipLV(id); 

    return 0 ; 
}
