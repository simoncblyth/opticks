
#include "G4Material.hh"
#include "X4MaterialLib.hh"
#include "X4PropertyMap.hh"
#include "GPropertyMap.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"

#include "PLOG.hh"

/**
X4MaterialLib::Standardize
----------------------------

* requires: both Geant4 G4MaterialTable and Opticks GMaterialLib 

* must be same number/names/order of the materials from both 

* for Geant4 materials with MPT (G4MaterialPropertiesTable) replaces it
  with an MPT converted from the Opticks GMaterial property map

* "Standardize" not a good name, its more "AdoptOpticksMaterialProperties"
  
   * BUT on the other hand it does standardize, because Opticks standardizes 
     materials to common wavelength domain when they are added to the GMaterialLib

* looks like this is not currently invoked by OKX4Test, only G4Opticks::TranslateGeometry

* Convertion with::

   G4MaterialPropertiesTable* mpt = X4PropertyMap::Convert( pmap ) ;



**/


void X4MaterialLib::Standardize()
{
    G4MaterialTable* mtab = G4Material::GetMaterialTable();
    const GMaterialLib* mlib = GMaterialLib::GetInstance();
    X4MaterialLib::Standardize( mtab, mlib ) ; 
}

void X4MaterialLib::Standardize( G4MaterialTable* mtab, const GMaterialLib* mlib )
{
    X4MaterialLib xmlib(mtab, mlib) ;  
}


X4MaterialLib::X4MaterialLib(G4MaterialTable* mtab, const GMaterialLib* mlib)
    :
    m_mtab(mtab),
    m_mlib(mlib)
{
    init();
}


void X4MaterialLib::init()
{
    unsigned num_materials = m_mlib->getNumMaterials();
    unsigned num_m4 = G4Material::GetNumberOfMaterials() ;  
    assert( num_materials == num_m4 ); 

    for(unsigned i=0 ; i < num_materials ; i++)
    {
        GMaterial*  pmap = m_mlib->getMaterial(i); 
        G4Material* m4 = (*m_mtab)[i] ; 
        assert( pmap && m4 );  

        const char* pmap_name = pmap->getName(); 
        const std::string& m4_name = m4->GetName();  
        bool name_match = strcmp( m4_name.c_str(), pmap_name) == 0 ;

        if(!name_match)
            LOG(fatal) 
                << " MATERIAL NAME MISMATCH " 
                << " index " << i 
                << " pmap_name " << pmap_name
                << " m4_name " << m4_name
                ;

        assert(name_match ); 
        if( m4->GetMaterialPropertiesTable() == NULL ) continue ; 

        G4MaterialPropertiesTable* mpt = X4PropertyMap::Convert( pmap ) ; 
        m4->SetMaterialPropertiesTable( mpt ) ; 
    }
}


