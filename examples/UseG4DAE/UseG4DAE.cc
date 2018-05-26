#include <iostream>
#include "G4GDMLParser.hh"
#include "G4DAEParser.hh"

#include "UseG4DAE.hh"

void UseG4DAE_gdml2dae( const char* gdml_path, const char* dae_path )
{
    G4VPhysicalVolume* world = NULL ; 

    std::cout << "reading from " << gdml_path << std::endl ; 
    {
        G4GDMLParser parser;

        bool validate = false ; 
        bool trimPtr = false ; 

        parser.SetStripFlag(trimPtr);
        parser.Read(gdml_path, validate);

        world = parser.GetWorldVolume() ;
    } 

    std::cout << "writing to " << dae_path << std::endl ; 
    {
        G4DAEParser parser ; 
    
        bool refs = true ;
        bool recreatePoly = false ;
        int nodeIndex = -1 ;        // so World is volume 0 
     
        parser.Write(dae_path, world, refs, recreatePoly, nodeIndex );
    }
    std::cout << "wrote to " << dae_path << std::endl ; 
}


#ifdef WITH_MAIN
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
#endif

