/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>
#include <climits>

#include "CFG4_BODY.hh"
#include "OPTICKS_LOG.hh"

// npy-
#include "NPY.hpp"


// okc-
#include "Opticks.hh"
#include "OpticksHub.hh"

//#include "GGeo.hh"
#include "GVector.hh"
#include "GGeoBase.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"

// cfg4-
#include "CG4.hh"
#include "CMPT.hh"
#include "CMaterialBridge.hh"
#include "CSurfaceBridge.hh"

// g4-
#include "G4Material.hh"
#include "G4OpticalSurface.hh"



/**
CInterpolationTest
====================

The GPU analogue of this is oxrap-/tests/interpolationTest

::

    CInterpolationTest --nointerpol   ## identity check
    CInterpolationTest                ## interpol check


**/


const char* TMPDIR = "$TMP/cfg4/CInterpolationTest" ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    Opticks ok(argc, argv);
    OpticksHub hub(&ok) ;

    LOG(error) << "[ CG4::CG4" ; 
    CG4 g4(&hub);
    LOG(error) << "] CG4::CG4" ; 

    CMaterialBridge* mbr = g4.getMaterialBridge();
    CSurfaceBridge*  sbr = g4.getSurfaceBridge();
    //mbr->dump();

    GGeoBase* ggb = hub.getGGeoBase();
    GBndLib* blib = ggb->getBndLib(); 

    bool interpolate = ok.hasOpt("nointerpol") ? false : true ; 

    unsigned nl_interpolate = unsigned(Opticks::DOMAIN_HIGH) - unsigned(Opticks::DOMAIN_LOW) + 1u ; 


    // PROBABLY NOT NEEDED ANYMORE : DONE IN GBndLib::load 
    NPY<double>* tex = blib->createBuffer();   // zipping together the dynamic buffer from materials and surfaces
    unsigned ndim = tex->getDimensions() ;
    assert( ndim == 5 );

    unsigned ni = tex->getShape(0);   // ~123: number of bnd
    unsigned nj = tex->getShape(1);   //    4: omat/osur/isur/imat
    unsigned nk = tex->getShape(2);   //    2: prop groups
    unsigned nl = interpolate ? nl_interpolate : tex->getShape(3);   //   39: wl samples   OR   820 - 60 + 1 = 761
    unsigned nm = tex->getShape(4);   //    4: float4 props


    

    const char* name = interpolate ? 
             "CInterpolationTest_interpol.npy"
          :
             "CInterpolationTest_identity.npy"
          ;



    NPY<double>* out = NPY<double>::make(ni,nj,nk,nl,nm);
    out->fill(-1.f);  // align unset to -1.f

    LOG(info) 
       << " interpolate (control with option: --nointerpol) " << interpolate
       << " name " << name
       << " tex " << tex->getShapeString()
       << " out " << out->getShapeString()
       ;

    unsigned nb = blib->getNumBnd();
    assert( ni == nb );
    assert( nj == 4 && nm == 4);

    glm::vec4 boundary_domain = Opticks::getDefaultDomainSpec() ;

    double wlow = boundary_domain.x ; 
    double wstep = interpolate ? 1.0f : boundary_domain.z ;   // 1.0f OR 20.0f  (nanometer)

    LOG(info) << " wlow " << wlow 
              << " wstep " << wstep 
              << " nl " << nl 
              ;


    const char* mkeys_0 = "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB" ;
    const char* mkeys_1 = "GROUPVEL,,, " ;  // <-- without trailing space the split giving 3 not 4 


    // gathering an Opticks tex buffer  from G4 material and surface properties
    // as machinery and interpolation test 
    //
    //  cf with almost the reverse action done by 
    //       CSurLib::addProperties
    //             converts GGeo properties detect/absorb/reflect_specular/reflect_diffuse 
    //
    //       GSurfaceLib::createStandardSurface
    //
    //      into G4 props
    //                  REFLECTIVITY 
    //                  EFFICIENCY (when sensor)


    LOG(info) << " nb " << nb ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        guint4 bnd = blib->getBnd(i);

        unsigned omat = bnd.x ; 
        unsigned osur = bnd.y ; 
        unsigned isur = bnd.z ; 
        unsigned imat = bnd.w ;   // these are zero based indices, UINT_MAX for unset 

        LOG(info) 
            << " i "    << std::setw(3) << i
            << " omat " << std::setw(3) << omat
            << " osur " << std::setw(3) << osur
            << " isur " << std::setw(3) << isur
            << " imat " << std::setw(3) << imat
            ;

        const G4Material* om = mbr->getG4Material(omat);
        const G4Material* im = mbr->getG4Material(imat);
        assert(om) ; 
        assert(im) ; 

        const G4OpticalSurface* os = osur == UINT_MAX ? NULL : sbr->getG4Surface(osur) ;
        const G4OpticalSurface* is = isur == UINT_MAX ? NULL : sbr->getG4Surface(isur) ;

        CMPT* ompt = new CMPT(om->GetMaterialPropertiesTable(), om->GetName().c_str()); 
        CMPT* impt = new CMPT(im->GetMaterialPropertiesTable(), im->GetName().c_str()); 
        CMPT* ospt = os == NULL ? NULL : new CMPT(os->GetMaterialPropertiesTable(), os->GetName().c_str());
        CMPT* ispt = is == NULL ? NULL : new CMPT(is->GetMaterialPropertiesTable(), is->GetName().c_str());
         
        for(unsigned k = 0 ; k < nk ; k++)
        { 
            const char* mkeys = k == 0 ? mkeys_0 : mkeys_1 ; 

            unsigned om_offset = out->getValueIndex(i, GBndLib::OMAT, k ) ;
            unsigned os_offset = out->getValueIndex(i, GBndLib::OSUR, k ) ;
            unsigned is_offset = out->getValueIndex(i, GBndLib::ISUR, k ) ;
            unsigned im_offset = out->getValueIndex(i, GBndLib::IMAT, k ) ;

            ompt->sample(out, om_offset, mkeys, wlow, wstep, nl );  
            impt->sample(out, im_offset, mkeys, wlow, wstep, nl );  

            if(k == 0)  // nothing in 2nd group for surfaces yet ???
            {
                if(ospt) 
                {
                    bool ospecular = os->GetFinish() == polished ;
                    ospt->sampleSurf(out, os_offset, wlow, wstep, nl, ospecular );  
                } 
                if(ispt) 
                {
                    bool ispecular = is->GetFinish() == polished ;
                    ispt->sampleSurf(out, is_offset, wlow, wstep, nl, ispecular  );  
                }
            }
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

    out->save(TMPDIR,name);

    return 0 ; 
}
