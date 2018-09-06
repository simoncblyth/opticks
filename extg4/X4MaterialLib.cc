
#include "G4Material.hh"
#include "X4MaterialLib.hh"
#include "X4PropertyMap.hh"
#include "GPropertyMap.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"

#include "PLOG.hh"


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











