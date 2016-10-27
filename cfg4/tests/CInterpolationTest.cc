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
#include "CMPT.hh"
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

    //bool interpolate = false ; 
    bool interpolate = true ; 
    unsigned nl_interpolate = unsigned(Opticks::DOMAIN_HIGH) - unsigned(Opticks::DOMAIN_LOW) + 1u ; 

    NPY<float>* tex = blib->createBuffer();   // zipping together the dynamic buffer from materials and surfaces
    unsigned ndim = tex->getDimensions() ;
    assert( ndim == 5 );

    unsigned ni = tex->getShape(0);   // ~123: number of bnd
    unsigned nj = tex->getShape(1);   //    4: omat/osur/isur/imat
    unsigned nk = tex->getShape(2);   //    2: prop groups
    unsigned nl = interpolate ? nl_interpolate : tex->getShape(3);   //   39: wl samples   OR   820 - 60 + 1 = 761
    unsigned nm = tex->getShape(4);   //    4: float4 props

    const char* path = interpolate ? 
             "$TMP/InterpolationTest/CInterpolationTest_interpol.npy"
          :
             "$TMP/InterpolationTest/CInterpolationTest_identity.npy"
          ;


    NPY<float>* out = NPY<float>::make(ni,nj,nk,nl,nm);
    out->zero();

    LOG(info) 
       << " path " << path
       << " tex " << tex->getShapeString()
       << " out " << out->getShapeString()
       ;

    unsigned nb = blib->getNumBnd();
    assert( ni == nb );
    assert( nj == 4 && nm == 4);

    glm::vec4 boundary_domain = Opticks::getDefaultDomainSpec() ;

    float wlow = boundary_domain.x ; 
    float wstep = interpolate ? 1.0f : boundary_domain.z ;   // 1.0f OR 20.0f  (nanometer)

    LOG(info) << " wlow " << wlow 
              << " wstep " << wstep 
              << " nl " << nl 
              ;


    const char* keys_0 = "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB" ;
    const char* keys_1 = "GROUPVEL,,," ;


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

        CMPT* ompt = new CMPT(om->GetMaterialPropertiesTable()); 
        CMPT* impt = new CMPT(im->GetMaterialPropertiesTable()); 
         
        for(unsigned k = 0 ; k < nk ; k++)
        { 
            const char* keys = k == 0 ? keys_0 : keys_1 ; 

            unsigned o_offset = out->getValueIndex(i, GBndLib::OMAT, k ) ;
            unsigned i_offset = out->getValueIndex(i, GBndLib::IMAT, k ) ;

            ompt->sample(out, o_offset, keys, wlow, wstep, nl );  
            impt->sample(out, i_offset, keys, wlow, wstep, nl );  
        }


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

    out->save(path);

    return 0 ; 
}
