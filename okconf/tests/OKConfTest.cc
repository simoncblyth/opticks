#include "OKConf.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
   
    LOG(info) << "OKConf::OptiXVersionInteger() " << OKConf::OptiXVersionInteger() ; 
    LOG(info) << "OKConf::Geant4VersionInteger() " << OKConf::Geant4VersionInteger() ; 

}


