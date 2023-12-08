/**
CSGMakerTest.cc
=================

HMM: this was stomping on standard GEOM folder  
USING MakeGeom/LoadGeom needs to effectively change GEOM to the name argument
and standard corresponding folder 

NOPE : THIS IS DOING SOMETHING NON-STANDARD
SO SHOULD DO THINGS IN NON-STANDARD MANUAL WAY 

**/

#include <csignal>
#include "ssys.h"
#include "spath.h"
#include "SSim.hh"
#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "OPTICKS_LOG.hh"

/**
GetNames
---------

Populate the names vector with all CSGMaker names or just the 
one provided when envvar CSGMakerTest_GEOM is defined. 
For listnames:true dump the names to stdout. 

**/

void GetNames( std::vector<std::string>& names, bool listnames )
{
     const char* geom = ssys::getenvvar("CSGMakerTest_GEOM", nullptr ); 
     // NB ExecutableName_GEOM is treated as GEOM override 

     if( geom == nullptr ) 
     {
         CSGMaker::GetNames(names); 
     }
     else
     { 
         names.push_back(geom); 
     }
     LOG(info) << " names.size " << names.size() ; 
     if(listnames) for(unsigned i=0 ; i < names.size() ; i++) std::cout << names[i] << std::endl ; 
}

int main(int argc, char** argv)
{
     const char* arg =  argc > 1 ? argv[1] : nullptr ; 
     bool listnames = arg && ( strcmp(arg,"N") == 0 || strcmp(arg,"n") == 0 ) ; 
     OPTICKS_LOG(argc, argv); 

     SSim* sim = SSim::Create(); 
     assert(sim); 
     if(!sim) std::raise(SIGINT); 

     std::vector<std::string> names ; 
     GetNames(names, listnames); 
     if(listnames) return 0 ; 

     for(unsigned i=0 ; i < names.size() ; i++)
     {
         const char* name = names[i].c_str() ; 
         LOG(info) << name ; 

         CSGFoundry* fd = CSGMaker::MakeGeom( name ); 
         LOG(info) << fd->desc();    

         const char* base = spath::Join("$TMP/CSGMakerTest",name) ; 

         fd->save(base);  

         //CSGFoundry* lfd = CSGFoundry::LoadGeom( name );
         CSGFoundry* lfd = CSGFoundry::Load(base); 


         LOG(info) << " lfd.loaddir " << lfd->loaddir ; 

         int rc = CSGFoundry::Compare(fd, lfd );  
         assert( 0 == rc );
         if(0!=rc) std::raise(SIGINT); 
     }

     return 0 ; 
}
