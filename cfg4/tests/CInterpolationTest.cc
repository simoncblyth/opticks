#include <cassert>

#include "CFG4_BODY.hh"
#include <climits>

// npy-
#include "NPY.hpp"


// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"

#include "GGeo.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"

// cfg4-
#include "CG4.hh"
#include "CMaterialBridge.hh"

// g4-
#include "G4Material.hh"


#include "GGEO_LOG.hh"
#include "CFG4_LOG.hh"
#include "PLOG.hh"


/**
CInterpolationTest
====================

The GPU analogue of this is oxrap-/tests/OInterpolationTest



**/


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    CFG4_LOG__ ; 
    GGEO_LOG__ ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok) ;

    CG4 g4(&hub);
    CMaterialBridge* mbr = g4.getMaterialBridge();
    //mbr->dump();

    GGeo* gg = hub.getGGeo();
    GBndLib* blib = gg->getBndLib(); 

    NPY<float>* tex = blib->createBuffer();   // zipping together the dynamic buffer from materials and surfaces
    NPY<float>* out = NPY<float>::make_like(tex) ;

    LOG(info) 
       << " tex " << tex->getShapeString()
       << " out " << out->getShapeString()
       ;

    unsigned ndim = out->getDimensions() ;
    assert( ndim == 5 );

    unsigned ni = out->getShape(0);   // ~123: number of bnd
    unsigned nj = out->getShape(1);   //    4: omat/osur/isur/imat
    unsigned nk = out->getShape(2);   //    2: prop groups
    unsigned nl = out->getShape(3);   //   39: wl samples  
    unsigned nm = out->getShape(4);   //    4: float4 props

    unsigned nb = blib->getNumBnd();
    assert( ni == nb );

    // getting from Opticks boundary omat/imat to the G4 materials
    LOG(info) << " nb " << nb ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        guint4 bnd = blib->getBnd(i);

        unsigned omat = bnd.x ; 
        unsigned osur = bnd.y ; 
        unsigned isur = bnd.z ; 
        unsigned imat = bnd.w ; 

        const G4Material* om = mbr->getG4Material(omat);
        const G4Material* im = mbr->getG4Material(imat);
        //  surface bridge ??

        LOG(info) << std::setw(5) << i 
                  << "(" 
                  << std::setw(2) << omat << ","
                  << std::setw(2) << ( osur == UINT_MAX ? -1 : (int)osur ) << ","
                  << std::setw(2) << ( isur == UINT_MAX ? -1 : (int)isur ) << ","
                  << std::setw(2) << imat 
                  << ")" 
                  << std::setw(60) << blib->shortname(bnd) 
                  << " om " << std::setw(30) << ( om ? om->GetName() : "-" ) 
                  << " im " << std::setw(30) << ( im ? im->GetName() : "-" ) 
                  ; 



    } 


    return 0 ; 
}
