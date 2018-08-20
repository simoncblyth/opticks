#include "OPTICKS_LOG.hh"
#include "CGDML.hh"


struct Demo 
{
   int answer ; 
};


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    Demo* d = new Demo { 42 } ; 
   
    LOG(info) << CGDML::GenerateName( "Demo", d, true );  


    return 0 ; 
}
