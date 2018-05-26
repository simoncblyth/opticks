#include <iostream>
#include "UseG4DAE.hh"

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        std::cout << "Usage " << argv[0] << " /path/to/input.gdml /path/to/output.dae " << std::endl ; 
        return 1 ;  
    }

    const char* gdml_path = argv[1] ; 
    const char* dae_path = argv[2] ;
 
    UseG4DAE_gdml2dae( gdml_path, dae_path );    

    return 0 ; 
}


