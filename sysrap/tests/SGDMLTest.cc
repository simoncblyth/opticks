#include "OPTICKS_LOG.hh"
#include "SGDML.hh"


struct Demo
{
   int answer ;
};


int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv);
    Demo* d = new Demo { 42 } ; 
    LOG(info) << SGDML::GenerateName( "Demo", d, true );
    return 0 ;
}   

