// TEST=OK_PROFILE_Test om-t

#include "Opticks.hh"
#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);

    Opticks* m_ok = new Opticks(argc, argv); 
    m_ok->configure(); 

    
    LOG(info) << argv[0] ;

    std::vector<double> times ; 
    OK_PROFILE("head");   


    for(unsigned i=0 ; i < 100 ; i++)
    {
        OK_PROFILE("body");   
    } 
    OK_PROFILE("tail");   


    m_ok->dumpProfile();

    //m_ok->saveProfile(); 


    return 0 ;
}


