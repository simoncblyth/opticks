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

#pragma once

#include <cstddef>
#include <string>
#include <map>
#include "plog/Severity.h"

class Opticks ;    // okc-
class OpticksHub ; // okg-

// ggeo-
class GBndLib ;
class GMaterialLib ;
class GSurfaceLib ;
class GScintillatorLib ;
class GMaterial ;

template <typename T> class GProperty ; 
template <typename T> class GPropertyMap ; 
template <typename T> class GDomain ; 


// g4-
class G4Material ; 
class G4MaterialPropertiesTable ; 
class G4VPhysicalVolume ;
class G4LogicalBorderSurface ;
class G4OpticalSurface ;
class G4PhysicsVector ;

/**
CPropLib
==========

CPropLib is base class of CMaterialLib which is a constituent of CDetector (eg CTestDector and CGDMLDetector)
that converts Opticks GGeo materials and surfaces into G4 materials and surfaces.

TODO:
------

* remove duplications between CPropLib and tests/CInterpolationTest 


**/


#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
class CFG4_API CPropLib {
   public:
       static const plog::Severity LEVEL ;
       static const char* SENSOR_MATERIAL ;
   public:
       CPropLib(OpticksHub* hub, int verbosity=0);
   private:
       void init();
       void initCheckConstants(); 
       void initSetupOverrides(); 
   public:
       GSurfaceLib* getSurfaceLib();
   public:
       // GGeo material access
       unsigned int getNumMaterials();
       const GMaterial* getMaterial(unsigned int index);
       const GMaterial* getMaterial(const char* shortname);
       bool hasMaterial(const char* shortname); 
   public:
       std::string getMaterialKeys(const G4Material* mat);
   public:
       G4LogicalBorderSurface* makeConstantSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2, double effi=0.f, double refl=0.f);
       G4LogicalBorderSurface* makeCathodeSurface(const char* name, G4VPhysicalVolume* pv1, G4VPhysicalVolume* pv2);
   private:
       G4OpticalSurface* makeOpticalSurface(const char* name);

   public:
       // used by CGDMLDetector::addMPT TODO: privatize
       G4MaterialPropertiesTable* makeMaterialPropertiesTable(const GMaterial* kmat);

   protected:
       void addProperties(G4MaterialPropertiesTable* mpt, GPropertyMap<double>* pmap, const char* _keys, bool keylocal=true, bool constant=false);
       void addProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<double>* prop );
       void addConstProperty(G4MaterialPropertiesTable* mpt, const char* matname, const char* lkey,  GProperty<double>* prop );
       GProperty<double>* convertVector(G4PhysicsVector* pvec);
       GPropertyMap<double>* convertTable(G4MaterialPropertiesTable* mpt, const char* name);
   private:
       void addSensorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name ); 
       void addScintillatorMaterialProperties( G4MaterialPropertiesTable* mpt, const char* name ); 
   protected:
       OpticksHub*        m_hub ; 
       Opticks*           m_ok ; 
       int                m_verbosity ; 

       GBndLib*           m_bndlib ; 
       GMaterialLib*      m_mlib ; 
       GSurfaceLib*       m_slib ; 
       GScintillatorLib*  m_sclib ; 
       GDomain<double>*    m_domain ; 
       double              m_dscale ;  
       plog::Severity     m_level ; 
       GPropertyMap<double>* m_sensor_surface ; 
   private:
       std::map<std::string, std::map<std::string, double> > m_const_override ; 

};
#include "CFG4_TAIL.hh"


