#include <iomanip>
#include <sstream>

#include "BStr.hh"
#include "NNode.hpp"
#include "NCSG.hpp"
#include "X4SolidRec.hh"
#include "G4VSolid.hh"

X4SolidRec::X4SolidRec( const G4VSolid* solid_,  const nnode* node_, const NCSG* csg_, unsigned soIdx_, unsigned lvIdx_ )
    :
    solid(solid_),
    node(node_),
    csg(csg_),
    soIdx(soIdx_),
    lvIdx(lvIdx_)
{
}


std::string X4SolidRec::desc() const 
{
    std::stringstream ss ; 
    ss
        << " so:" << BStr::utoa(soIdx, 3, true)  
        << " lv:" << BStr::utoa(lvIdx, 3, true)  
        << " mx:" << BStr::utoa(node->maxdepth(), 2, true )  
        << " soName: " << solid->GetName() 
        ;

    return ss.str(); 
}



