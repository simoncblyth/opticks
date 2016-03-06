#include "CTraverser.hh"

#include <algorithm>
#include <sstream>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


#include "NLog.hpp"

void CTraverser::Traverse()
{
     G4LogicalVolume* lv = m_top->GetLogicalVolume() ;
     TraverseVolumeTree(lv, 0 );
}

void CTraverser::Visit(const G4LogicalVolume* const lv)
{
    const G4String lvname = lv->GetName();
    G4VSolid* solid = lv->GetSolid();
    G4Material* material = lv->GetMaterial();
     
    const G4String geoname = solid->GetName() ;
    const G4String matname = material->GetName();

     LOG(info) << "CTraverser::Visit"
               << std::setw(20) << lvname
               << std::setw(50) << geoname
               << std::setw(20) << matname
               ;
}

G4Transform3D CTraverser::TraverseVolumeTree(const G4LogicalVolume* const lv, const G4int depth)
{
     G4Transform3D R ;
     Visit(lv);

     const G4int daughterCount = lv->GetNoDaughters();    
     for (G4int i=0;i<daughterCount;i++) 
     {
         const G4VPhysicalVolume* const d_pv = lv->GetDaughter(i);
         G4Transform3D daughterR;
         daughterR = TraverseVolumeTree(d_pv->GetLogicalVolume(),depth+1); 
     }

     addMaterial(lv->GetMaterial());
     return R ;  
}


bool CTraverser::hasMaterial(const G4Material* material)
{
     return std::find(m_materials.begin(), m_materials.end(), material) != m_materials.end()  ;
}

void CTraverser::addMaterial(const G4Material* material)
{
     if(!hasMaterial(material)) m_materials.push_back(material) ;
}

void CTraverser::dumpMaterials(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_materials.size() ; i++)
    {
        const G4Material* material = m_materials[i];
        dumpMaterial(material);
    } 
}

void CTraverser::dumpMaterial(const G4Material* material)
{
    LOG(info) << material->GetName() ;
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    if(!mpt) return ;

    typedef std::map<G4String,G4MaterialPropertyVector*,std::less<G4String> >  PMAP ;
    typedef std::map< G4String, G4double,std::less<G4String> > CMAP ; 

    const PMAP* pm = mpt->GetPropertiesMap();
    const CMAP* cm = mpt->GetPropertiesCMap(); 

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


void CTraverser::dumpMaterialProperty(const G4String& name, const G4MaterialPropertyVector* pvec)
{
    unsigned int len = pvec->GetVectorLength() ;

    LOG(info) << name 
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





