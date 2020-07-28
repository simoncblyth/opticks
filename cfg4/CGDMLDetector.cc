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

// cfg4--;op --cgdmldetector

#include "CFG4_BODY.hh"

#include "BFile.hh"
#include "GLMFormat.hpp"

// ggeo-
#include "GMaterial.hh"
#include "GSurfaceLib.hh"
#include "GProperty.hh"

// okc-
#include "Opticks.hh"
#include "OpticksResource.hh"

// okg-
#include "OpticksHub.hh"   

#include "CMaterialSort.hh"
#include "GMaterialLib.hh"
#include "CMaterialLib.hh"
#include "CTraverser.hh"    // m_traverser resides in base 
#include "CGDMLDetector.hh"

// hmm not much use of X4 from CGF4 yet, this might be the first
#include "X4MaterialLib.hh"

#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4GDMLParser.hh"


#include "PLOG.hh"

const plog::Severity CGDMLDetector::LEVEL = PLOG::EnvLevel("CGDMLDetector", "DEBUG") ; 


CGDMLDetector::CGDMLDetector(OpticksHub* hub, OpticksQuery* query, CSensitiveDetector* sd)
    : 
    CDetector(hub, query, sd)
{
    LOG(LEVEL) << "[" ; 
    init();
    LOG(LEVEL) << "]" ; 
}

CGDMLDetector::~CGDMLDetector()
{
}


void CGDMLDetector::init()
{
    const char* path = m_ok->getCurrentGDMLPath() ;

    bool exists = BFile::ExistsFile(path);
    if( !exists )
    {
         LOG(error)
              << "CGDMLDetector::init" 
              << " PATH DOES NOT EXIST "
              << " path " << path
              ; 

         setValid(false);  
         return ; 
    }

    LOG(LEVEL) << "parse " << path ; 

    G4VPhysicalVolume* world = parseGDML(path);

    sortMaterials();

    setTop(world);   // invokes *CDetector::traverse*

    addMPTLegacyGDML(); 


    if(m_ok->isLegacy())
    {
        LOG(error) << " skip standardizeGeant4MaterialProperties in legacy running " ; 
    } 
    else
    {
        standardizeGeant4MaterialProperties();
    }

    attachSurfaces();
    // kludge_cathode_efficiency(); 

#ifdef OLD_SENSOR
    hookupSD(); 
#endif

}

G4VPhysicalVolume* CGDMLDetector::parseGDML(const char* path) const 
{
    bool validate = false ; 
    bool trimPtr = false ; 
    G4GDMLParser parser;
    parser.SetStripFlag(trimPtr);
    parser.Read(path, validate);
    return parser.GetWorldVolume() ;
}

void CGDMLDetector::sortMaterials()
{
    GMaterialLib* mlib = getGMaterialLib();     

    //const std::map<std::string, unsigned>& order = mlib->getOrder();  
    //  old order was from preferences

    std::map<std::string, unsigned> order ;  
    mlib->getCurrentOrder(order); 
    // new world order, just use the current Opticks material order : which should correspond to Geant4 creation order
    // unlike following a trip thru  GDML that reverses the material order 
    // see notes/issues/ckm-okg4-material-rindex-mismatch.rst
 
    CMaterialSort msort(order);  
}



void CGDMLDetector::saveBuffers()
{
    // split off to allow setting idpath override whilst testing
    CDetector::saveBuffers("CGDMLDetector", 0);    
}


/**
CGDMLDetector::addMPTLegacyGDML
-----------------------------------

The GDML exported by geant4 that comes with nuwa lack material properties 
so use the properties from the G4DAE export, to enable recovery of the materials.

Have observed, Bialkali looses its EFFICIENCY, once thru Opticks standardization, the
EFFICIENCY gets planted onto the fake SensorSurfaces::

**/

void CGDMLDetector::addMPTLegacyGDML()
{
    unsigned nmat = m_traverser->getNumMaterials();
    unsigned nmat_without_mpt = m_traverser->getNumMaterialsWithoutMPT();
    //assert( nmat > 0 && nmat_without_mpt == 0 );  // hmm this will prevent use of old GDML   
    LOG(LEVEL) 
        << " nmat " << nmat
        << " nmat_without_mpt " << nmat_without_mpt
        ; 

    if(nmat > 0 && nmat_without_mpt == 0 )
    {
        LOG(error) 
            << " Looks like GDML has succeded to load material MPTs  "
            << " nmat " << nmat
            << " nmat_without_mpt " << nmat_without_mpt 
            << " skipping the fixup " 
                     ;
        return ; 
 
    }
    if(nmat > 0 && nmat_without_mpt == nmat )
    {
        LOG(error) 
            << " ALL G4 MATERIALS LACK MPT "
            << " FIXING USING Opticks MATERIALS " 
            ;
    } 
    else
    {
        LOG(fatal) 
            << " UNEXPECTED TO SEE ONLY SOME Geant4 MATERIALS WITHOUT MPT " 
            << " nmat " << nmat 
            << " nmat_without_mpt " << nmat_without_mpt
            ;
        //assert(0);
    }
 

    for(unsigned int i=0 ; i < nmat_without_mpt ; i++)
    {
        G4Material* g4mat = m_traverser->getMaterialWithoutMPT(i) ;
        const char* name = g4mat->GetName() ;

        const std::string base = BFile::Name(name);
        const char* shortname = base.c_str();

        const GMaterial* ggmat = m_mlib->getMaterial(shortname);          
        assert(ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material") ;

        LOG(verbose) 
            << " g4mat " << std::setw(45) << name
            << " shortname " << std::setw(25) << shortname
            ;

        G4MaterialPropertiesTable* mpt = m_mlib->makeMaterialPropertiesTable(ggmat);
        g4mat->SetMaterialPropertiesTable(mpt);
        //m_mlib->dumpMaterial(g4mat, "CGDMLDetector::addMPT");        
         
    }

    LOG(info) << "CGDMLDetector::addMPT added MPT to " <<  nmat_without_mpt << " g4 materials " ; 


}

/**
CGDMLDetector::standardizeGeant4MaterialProperties
-----------------------------------------------------

Duplicates G4Opticks::standardizeGeant4MaterialProperties

**/


void CGDMLDetector::standardizeGeant4MaterialProperties()   
{
    LOG(info) << "[" ;
    X4MaterialLib::Standardize() ;
    LOG(info) << "]" ;
}









/**
CGDMLDetector::kludge_cathode_efficiency
-----------------------------------------

NOT NEEDED Cathode Efficiency fixup is done by CPropLib AFTER FIXING A KEY BUG 

See :doc:`notes/issues/direct_route_needs_AssimpGGeo_convertSensors_equivalent`


void CGDMLDetector::kludge_cathode_efficiency()
{
    GSurfaceLib* gsl = getGSurfaceLib(); 

    unsigned num_surf = gsl->getNumSurfaces() ; 

    std::vector<unsigned> indices ;  
    gsl->getIndicesWithNameEnding(indices, GSurfaceLib::SENSOR_SURFACE) ; 
    unsigned num_sens = indices.size(); 

    LOG(error) << " gsl " 
               << " num_surf " << num_surf
               << " num_sens " << num_sens 
               ; 

    for( unsigned i=0 ; i < num_sens ; i++)
    {
        unsigned idx = indices[i] ; 

        GPropertyMap<float>* surf = gsl->getSurface(idx) ;    
        const char* name = surf->getName(); 
        std::string sslv = surf->getSSLV(); 

        bool is_sensor_surface = GSurfaceLib::NameEndsWithSensorSurface( name ) ; 
        assert( is_sensor_surface ); 

        const char* stem = GSurfaceLib::NameWithoutSensorSurface( name ) ; 

        GProperty<float>* detect = surf->getProperty("detect") ; 
        assert( detect ); 

        LOG(error) 
            << " i " << i 
            << " idx " << idx 
            << " name " << name 
            << " stem " << stem 
            << " detect " << detect->brief()
            << " sslv " << sslv  
            ;
    }
}

**/


 
