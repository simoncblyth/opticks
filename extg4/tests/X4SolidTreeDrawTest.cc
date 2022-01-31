#include "OPTICKS_LOG.hh"
#include "X4SolidMaker.hh"
#include "X4SolidTree.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* qname = argc > 1 ? argv[1] : nullptr ; 
    if( qname == nullptr ) return 0 ; 

    const G4VSolid* solid = X4SolidMaker::Make( qname ); 
    assert( solid ); 

    LOG(info) << " qname " << qname << " solid " << solid ; 

    X4SolidTree::Draw(solid); 

  
    return 0 ; 
}
