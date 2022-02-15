#include "SSys.hh"
#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
     const char* arg =  argc > 1 ? argv[1] : nullptr ; 
     OPTICKS_LOG(argc, argv); 

     //const char* _geom = "sphere" ;   
     //const char* _geom = "UnionBoxSphere" ;
     const char* _geom = "OverlapBoxSphere" ;  
     //const char* _geom = nullptr ; 
     const char* geom = SSys::getenvvar("GEOM", _geom ); 

     std::vector<std::string> names ; 
     if( geom == nullptr ) CSGMaker::GetNames(names); 
     else names.push_back(geom); 

     LOG(info) << " names.size " << names.size() ; 
     if( arg && ( strcmp(arg,"N") == 0 || strcmp(arg,"n") == 0 ))
     {
         for(unsigned i=0 ; i < names.size() ; i++) std::cout << names[i] << std::endl ; 
         return 0 ;   
     }

     for(unsigned i=0 ; i < names.size() ; i++)
     {
         const char* name = names[i].c_str() ; 
         LOG(info) << name ; 

         CSGFoundry* fd = CSGFoundry::MakeGeom( name ); 
         LOG(info) << fd->desc();    

         fd->write(); 

         CSGFoundry* lfd = CSGFoundry::LoadGeom( name ); 
         int rc = CSGFoundry::Compare(fd, lfd );  
         assert( 0 == rc );
     }



     return 0 ; 
}
