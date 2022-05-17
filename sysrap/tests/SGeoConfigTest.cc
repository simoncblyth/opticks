/**

::

    EMM=0,63 SGeoConfigTest

**/


#include "OPTICKS_LOG.hh"
#include "SGeoConfig.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) << SGeoConfig::Desc() ; 
    LOG(info) << SGeoConfig::DescEMM() ; 

    std::vector<std::string>* arglist = SGeoConfig::Arglist() ; 
    if(arglist) 
    {
         LOG(info) << " SGeoConfig::Arglist " << arglist->size() ; 
         for(unsigned i=0 ; i < arglist->size() ; i++) std::cout << (*arglist)[i] << std::endl ; 
    }

    if(SGeoConfig::CXSkipLV())
    {
        for(int i=0 ; i < 20 ; i++) std::cout << std::setw(4) << i << " SGeoConfig::IsCXSKipLV " << SGeoConfig::IsCXSkipLV(i) << std::endl ; 
    } 

    return 0 ; 

}
