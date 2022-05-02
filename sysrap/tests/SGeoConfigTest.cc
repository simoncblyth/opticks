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

    return 0 ; 

}
