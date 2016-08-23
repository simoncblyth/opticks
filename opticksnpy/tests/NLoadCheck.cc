#include <cassert>
/*

::

    NLoadCheck ~/opticksdata/gensteps/dayabay/natural/1.npy f
    NLoadCheck ~/opticksdata/gensteps/dayabay/natural/1.npy d

*/
#include "NPY.hpp"
#include "NLoad.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"

int main(int argc, char** argv)
{
     PLOG_(argc, argv);
     NPY_LOG__ ; 

     NPYBase::setGlobalVerbose(true);

     if(argc < 2)
     {
         LOG(warning) << "expecting first argument with path to NPY array to load" ; 
         exit(0) ;
     }
      
     char* path = argv[1] ;
     char* typ = argc > 2 ? argv[2] : (char*)"f" ;  


     if( typ[0] == 'f' )
     {
         NPY<float>* af = NPY<float>::load(path) ;
         if(af == NULL)
         {
             LOG(info) << "NPY<float>::load FAILED try debugload " ; 
             af = NPY<float>::debugload(path) ; 
         }
         if(af) af->dump();
     }
     else if( typ[0] == 'd' )
     {
        NPY<double>* ad = NPY<double>::load(path);
        if(ad == NULL)
        {
             LOG(info) << "NPY<double>::load FAILED try debugload " ; 
             ad = NPY<double>::debugload(path) ; 
        }

        if(ad) ad->dump();
     }
     else
     {
         LOG(warning) << "2nd argument needs to be an f or d to pick type" ;
     }
 
     return 0 ; 
}
