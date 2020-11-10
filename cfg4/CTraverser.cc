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

#include "CFG4_BODY.hh"
#include <algorithm>
#include <sstream>

#include "CFG4_PUSH.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "CFG4_POP.hh"

// brap-
#include "BStr.hh"

// okc-
#include "Opticks.hh"
#include "OpticksQuery.hh"

// npy-
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "NPY.hpp"
#include "NBoundingBox.hpp"
#include "PLOG.hh"

#include "CSolid.hh"
#include "CTraverser.hh"


const plog::Severity CTraverser::LEVEL = PLOG::EnvLevel("CTraverser", "DEBUG"); 

const char* CTraverser::GROUPVEL = "GROUPVEL" ; 

CTraverser::CTraverser(Opticks* ok, G4VPhysicalVolume* top, NBoundingBox* bbox, OpticksQuery* query ) 
    :
    m_ok(ok),
    m_top(top),
    m_bbox(bbox ? bbox : new NBoundingBox),
    m_query(query),
    m_verbosity(1),
    m_lcount(0),
    m_ltransforms(NULL),
    m_gcount(0),
    m_ancestor_index(0),
    m_center_extent(NULL),
    m_gtransforms(NULL)
{
    init();
}


void CTraverser::init()
{
    m_ltransforms = NPY<float>::make(0, 4, 4);
    m_gtransforms = NPY<float>::make(0, 4, 4);
    m_center_extent = NPY<float>::make(0, 4);
}


unsigned int CTraverser::getNumMaterials() const 
{
   return m_materials.size();
}
unsigned int CTraverser::getNumSelected() const 
{
   return m_selection.size();
}

unsigned int CTraverser::getNumMaterialsWithoutMPT() const 
{
   return m_materials_without_mpt.size();
}
const G4Material* CTraverser::getMaterial(unsigned index) const 
{
   return m_materials.at(index);
}
G4Material* CTraverser::getMaterialWithoutMPT(unsigned index) const
{
   return m_materials_without_mpt.at(index);
}
void CTraverser::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

NPY<float>* CTraverser::getGlobalTransforms() const 
{
    return m_gtransforms ; 
}
NPY<float>* CTraverser::getLocalTransforms() const
{ 
    return m_ltransforms ; 
}
NPY<float>* CTraverser::getCenterExtent() const
{
    return m_center_extent  ; 
}







void CTraverser::Summary(const char* msg) const
{
    LOG(LEVEL) 
        << msg 
        << " numMaterials " << getNumMaterials() 
        << " numMaterialsWithoutMPT " << getNumMaterialsWithoutMPT() 
        ;
}


std::string CTraverser::description() const
{   
    std::stringstream ss ; 
    ss 
       << " numSelected " << getNumSelected()
       << " bbox " << m_bbox->description()
       << " pvs.size " << m_pvs.size()
       << " lvs.size " << m_lvs.size()
       ;
    return ss.str();
}

void CTraverser::Traverse()
{
    if(m_ok->isDbgSurf())
        LOG(info) << "( [--dbgsurf] " ;

    VolumeTreeTraverse();
    AncestorTraverse();

    if(m_ok->isDbgSurf())
    {
        LOG(info) << ") [--dbgsurf] " 
                  << description() 
                  ;
    }
}

void CTraverser::VolumeTreeTraverse()
{
     assert(m_top) ;
     G4LogicalVolume* lv = m_top->GetLogicalVolume() ;
     VolumeTreeTraverse(lv, 0 );
}

void CTraverser::AncestorTraverse()
{
     assert(m_top) ;
     m_pvnames.clear();
     m_pvs.clear();
     m_lvs.clear();
     m_lvm.clear();

     m_ancestor_index = 0 ; 
     std::vector<const G4VPhysicalVolume*> ancestors ; 

     AncestorTraverse(ancestors, m_top, 0, false);

     LOG(debug) << description() ;
}


void CTraverser::AncestorTraverse(std::vector<const G4VPhysicalVolume*> ancestors, const G4VPhysicalVolume* pv, unsigned int depth, bool recursive_select )
{
     ancestors.push_back(pv); 
     
     const char* pvname = pv->GetName() ; 

     bool selected = m_query ? m_query->selected(pvname, m_ancestor_index, depth, recursive_select) : true ;

     LOG(debug) 
         << " pvname " << pvname
         << " selected " << selected
         ;

     if(selected)
     {
         m_selection.push_back(m_ancestor_index);
     }

     AncestorVisit(ancestors, selected);   // collects into m_gtransforms


     G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     for (int i=0 ; i<lv->GetNoDaughters() ;i++) AncestorTraverse(ancestors, lv->GetDaughter(i), depth+1, recursive_select ); 
}


void CTraverser::AncestorVisit(std::vector<const G4VPhysicalVolume*> ancestors, bool selected)
{
    G4Transform3D T ; 

    for(unsigned int i=0 ; i < ancestors.size() ; i++)
    {
        const G4VPhysicalVolume* apv = ancestors[i] ;

        G4RotationMatrix rot, invrot;
        if (apv->GetFrameRotation() != 0)
        {   
            rot = *(apv->GetFrameRotation());
            invrot = rot.inverse();
        }

        G4Transform3D P(invrot,apv->GetObjectTranslation()); 

        T = T*P ; 
    }
    const G4VPhysicalVolume* pv = ancestors.back() ; 
    const G4LogicalVolume* lv = pv->GetLogicalVolume() ;


    const std::string& pvn = pv->GetName();
    const std::string& lvn = lv->GetName();

    LOG(verbose) << " pvn " << pvn ; 
    LOG(verbose) << " lvn " << lvn ; 


    updateBoundingBox(lv->GetSolid(), T, selected);

    LOG(debug) 
        << " size " << std::setw(3) << ancestors.size() 
        << " gcount " << std::setw(6) << m_gcount 
        << " pvname " << pv->GetName() 
        ;
    m_gcount += 1 ; 


    collectTransformT(m_gtransforms, T );
    m_pvnames.push_back(pvn);

    m_pvs.push_back(pv);
    m_lvs.push_back(lv);  // <-- hmm will be many of the same lv in m_lvs 

    m_lvm[lvn] = lv ; 


    m_ancestor_index += 1 ; 
}



void CTraverser::dumpLV(const char* msg) const
{
    LOG(info) << msg ; 
    typedef std::map<std::string, const G4LogicalVolume*> SV ; 
    for(SV::const_iterator it=m_lvm.begin() ; it != m_lvm.end() ; it++)
    {
        std::string name = it->first ; 
        const G4LogicalVolume* lv = it->second ; 

        std::cout
                  << ( lv ? lv->GetName() : "NULL" )
                  << std::endl ; 

        assert(lv);
        std::string check = lv->GetName();
        assert(strcmp(check.c_str(), name.c_str()) == 0);

    }
}



void CTraverser::updateBoundingBox(const G4VSolid* solid, const G4Transform3D& transform, bool selected)
{
    glm::vec3 low ; 
    glm::vec3 high ; 
    glm::vec4 center_extent ; 

    CSolid csolid(solid);
    csolid.extent(transform, low, high, center_extent);
    m_center_extent->add(center_extent);

    if(selected)
    {
        m_bbox->update(low, high);
    }

    LOG(debug)
        << " low " << gformat(low)
        << " high " << gformat(high)
        << " ce " << gformat(center_extent)
        << " bb " << m_bbox->description()
        ;

}




unsigned CTraverser::getNumPV() const
{
    return m_pvs.size();
}

const G4VPhysicalVolume* CTraverser::getTop() const
{
    return m_top ; 
}
const G4VPhysicalVolume* CTraverser::getPV(unsigned index) const
{
    return index < m_pvs.size() ? m_pvs.at(index) : NULL ;  
}

const G4VPhysicalVolume* CTraverser::getPV(const char* name) const
{
    int index = BStr::index_first(m_pvnames, name) ;

    std::vector<unsigned> indices ; 
    int num_indices = BStr::index_all( indices, m_pvnames, name) ;

    LOG(LEVEL) 
        << " name " << name
        << " index " << index 
        << " num_indices " << num_indices
        ;     


    assert( m_pvs.size() == m_pvnames.size() );

    bool valid = index > -1 && unsigned(index) < m_pvs.size()  ;

    if(!valid)
    {
        LOG(fatal) 
            << " name " << name 
            << " index " << index 
            << " m_pvs " << m_pvs.size()
            << " m_pvnames " << m_pvnames.size()
            ;
    }
    assert( valid );
    const G4VPhysicalVolume* pv = getPV(index) ;
    assert( pv );
    return pv ;   
}




unsigned CTraverser::getNumLV() const
{
    return m_lvs.size();
}
const G4LogicalVolume* CTraverser::getLV(unsigned index) const
{
    return index < m_lvs.size() ? m_lvs.at(index) : NULL ;  
}

const G4LogicalVolume* CTraverser::getLV(const char* name) const
{
   return m_lvm.count(name) == 1 ? m_lvm.at(name) : NULL ; 
}


/*
const G4LogicalVolume* CTraverser::getLVMatch(const char* name) const
{
   return m_lvm.count(name) == 1 ? m_lvm.at(name) : NULL ; 
}

*/



glm::mat4 CTraverser::getGlobalTransform(unsigned int index) const
{
    return m_gtransforms->getMat4(index);
}
glm::mat4 CTraverser::getLocalTransform(unsigned int index) const
{
    return m_ltransforms->getMat4(index);
}
glm::vec4 CTraverser::getCenterExtent(unsigned int index) const
{
    return m_center_extent->getQuad_(index);
}




unsigned int CTraverser::getNumGlobalTransforms() const
{
    return m_gtransforms->getShape(0);
}
unsigned int CTraverser::getNumLocalTransforms() const
{
    return m_ltransforms->getShape(0);
}



const char* CTraverser::getPVName(unsigned int index) const
{
    return m_pvnames.at(index).c_str();
}



G4Transform3D CTraverser::VolumeTreeTraverse(const G4LogicalVolume* const lv, const G4int depth)
{
     G4Transform3D R, invR ;  // huh invR stays identity, see g4dae/src/G4DAEWriteStructure.cc
     Visit(lv);

     const G4int daughterCount = lv->GetNoDaughters();    
     for (G4int i=0;i<daughterCount;i++) 
     {
         const G4VPhysicalVolume* const physvol = lv->GetDaughter(i);

         G4Transform3D daughterR;

         G4RotationMatrix rot, invrot;
         if (physvol->GetFrameRotation() != 0)
         {   
            rot = *(physvol->GetFrameRotation());
            invrot = rot.inverse();
         }

         daughterR = VolumeTreeTraverse(physvol->GetLogicalVolume(),depth+1);  // recursive call  

         // G4Transform3D P(rot,physvol->GetObjectTranslation());  GDML does this : not inverting the rotation portion 
         G4Transform3D P(invrot,physvol->GetObjectTranslation());

         VisitPV(physvol, invR*P*daughterR);  // postorder (visit after recursive call), collecting into m_ltransforms

        // This mimicks what is done in g4dae/src/G4DAEWriteStructure.cc which follows GDML (almost) 
        // despite trying to look like it accounts for all the transforms through the tree
        // it aint doing that as:
        //
        //   * R and invR always stay identity -> daughterR is always identity 
        //   * so only P is relevant, which is in inverse of the frame rotation and the object translation   
        //
        //  So the G4DAE holds just the one level transforms, relying on the post processing
        //  tree traverse to multiply them to give global transforms 
     }


     G4Material* material = lv->GetMaterial(); 

     addMaterial(material);

     G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
     if(mpt == NULL)
         addMaterialWithoutMPT(material);


     return R ;  
}

void CTraverser::VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T )
{
    LOG(debug) 
        << " lcount " << std::setw(6) << m_lcount 
        << " pvname " << pv->GetName() 
        ;
    m_lcount += 1 ; 

    collectTransformT(m_ltransforms, T );
}

void CTraverser::collectTransform(NPY<float>* buffer, const G4Transform3D& T)
{
    unsigned int nval = 16 ; 
    float* vals = new float[nval]; 

    vals[0] = T.xx() ; 
    vals[1] = T.xy() ; 
    vals[2] = T.xz() ; 
    vals[3] = T.dx() ; 

    vals[4] = T.yx() ; 
    vals[5] = T.yy() ; 
    vals[6] = T.yz() ; 
    vals[7] = T.dy() ; 

    vals[8] = T.zx() ; 
    vals[9] = T.zy() ; 
    vals[10] = T.zz() ; 
    vals[11] = T.dz() ; 

    vals[12] = 0 ; 
    vals[13] = 0 ; 
    vals[14] = 0 ; 
    vals[15] = 1 ; 

    buffer->add(vals, nval) ;

    delete[] vals ; 
}



void CTraverser::collectTransformT(NPY<float>* buffer, const G4Transform3D& T)
{
    unsigned int nval = 16 ; 
    float* vals = new float[nval]; 

    vals[0] = T.xx() ; 
    vals[1] = T.yx() ; 
    vals[2] = T.zx() ; 
    vals[3] = 0 ; 

    vals[4] = T.xy() ; 
    vals[5] = T.yy() ; 
    vals[6] = T.zy() ; 
    vals[7] = 0 ; 

    vals[8] = T.xz() ; 
    vals[9] = T.yz() ; 
    vals[10] = T.zz() ; 
    vals[11] = 0 ; 

    vals[12] = T.dx() ; 
    vals[13] = T.dy() ; 
    vals[14] = T.dz() ; 
    vals[15] = 1 ; 

    buffer->add(vals, nval) ;

    delete[] vals ; 
}


void CTraverser::Visit(const G4LogicalVolume* const lv)
{
    const G4String lvname = lv->GetName();
    G4VSolid* solid = lv->GetSolid();
    G4Material* material = lv->GetMaterial();
     
    const G4String geoname = solid->GetName() ;
    const G4String matname = material->GetName();

    if(m_verbosity > 1 )
        LOG(info) 
            << std::setw(20) << lvname
            << std::setw(50) << geoname
            << std::setw(20) << matname
            ;
}


bool CTraverser::hasMaterial(const G4Material* material) const
{
     return std::find(m_materials.begin(), m_materials.end(), material) != m_materials.end()  ;
}

bool CTraverser::hasMaterialWithoutMPT(G4Material* material) const
{
     return std::find(m_materials_without_mpt.begin(), m_materials_without_mpt.end(), material) != m_materials_without_mpt.end()  ;
}

void CTraverser::addMaterial(const G4Material* material)
{
     if(!hasMaterial(material)) 
     {
         m_materials.push_back(material) ;
     }
}

void CTraverser::addMaterialWithoutMPT(G4Material* material)
{
     if(!hasMaterialWithoutMPT(material)) 
     {
         m_materials_without_mpt.push_back(material) ;
     }
}






unsigned CTraverser::getNumSurfaces() const
{
   return 0 ;
}
void CTraverser::addBorderSurface(const G4LogicalBorderSurface*)
{
    assert(0);
}
void CTraverser::addSkinSurface(const G4LogicalSkinSurface*)
{
    assert(0);
}
void CTraverser::addOpticalSurface(const G4OpticalSurface*)
{
    assert(0);
}






void CTraverser::dumpMaterials(const char* msg) const
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_materials.size() ; i++)
    {
        const G4Material* material = m_materials.at(i);
        dumpMaterial(material);
    } 
}


void CTraverser::createGroupVel()
{
    // First get of GROUPVEL property creates it 
    // based on RINDEX property

    for(unsigned int i=0 ; i < m_materials.size() ; i++)
    {
        const G4Material* material = m_materials[i];
        G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
        if(mpt)
        {
            G4MaterialPropertyVector* gv = mpt->GetProperty(GROUPVEL);  
            unsigned int len = gv->GetVectorLength() ;
            if(m_verbosity > 1 )
                 LOG(info)
                     << " material " << material->GetName()
                     << " groupvel len " << len
                     ;
        }
        else
        {
            LOG(warning) 
                << " material lacks MPT " << i << " " << material->GetName() ;
        } 
    } 
}


void CTraverser::dumpMaterial(const G4Material* material) const
{
    LOG(info) << material->GetName() ;
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    if(!mpt) return ;

    typedef std::map<G4String,G4MaterialPropertyVector*,std::less<G4String> >  PMAP ;
    //typedef std::map< G4String, G4double,std::less<G4String> > CMAP ; 

    const PMAP* pm = mpt->GetPropertiesMap();
    //const CMAP* cm = mpt->GetPropertiesCMap(); 

     std::stringstream ss ; 
    for(PMAP::const_iterator it=pm->begin() ; it!=pm->end() ; it++)
    {
        G4String pname = it->first ; 
        G4MaterialPropertyVector* pvec = it->second ; 
        ss << pname << " " ;  
        if(m_verbosity > 1 )
        dumpMaterialProperty(pname, pvec);
    }    
    std::string props = ss.str() ;

    LOG(info) <<  props ; 
}


void CTraverser::dumpMaterialProperty(const G4String& name, const G4MaterialPropertyVector* pvec) const
{
    unsigned int len = pvec->GetVectorLength() ;

    LOG(info) 
        << name 
        << " len " << len 
        << " h_Planck*c_light/nm " << h_Planck*c_light/nm
        ;

    for (unsigned int i=0; i<len; i++)
    {   
        G4double energy = pvec->Energy(i) ;
        G4double wavelength = h_Planck*c_light/energy ;
        G4double val = (*pvec)[i] ;

        LOG(info)
            << std::fixed << std::setprecision(3) 
            << " eV " << std::setw(10) << energy/eV
            << " nm " << std::setw(10) << wavelength/nm
            << " v  " << std::setw(10) << val ;
    }   
}





