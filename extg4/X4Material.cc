#include <string>
#include "G4Material.hh"
#include "X4PhysicsVector.hh"
#include "X4Material.hh"
#include "X4MaterialPropertiesTable.hh"
#include "GMaterial.hh"

#include "SDigest.hh"
#include "PLOG.hh"


std::string X4Material::Digest()
{
    const G4MaterialTable* mtab  = G4Material::GetMaterialTable();
    const std::vector<G4Material*>& materials = *mtab ; 
    return Digest(materials);  
}

std::string X4Material::Digest( const std::vector<G4Material*>& materials )
{
    SDigest dig ;
    for(unsigned i=0 ; i < materials.size() ; i++)
    {
        const G4Material* material = materials[i] ; 
        std::string idig = Digest(material);
        dig.update( const_cast<char*>(idig.data()), idig.size() );  
    } 
    return dig.finalize();
}

std::string X4Material::Digest( const G4Material* material )
{
    if(!material) return "" ; 
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable() ; 
    const G4String& name = material->GetName();    
    std::string dmpt = X4MaterialPropertiesTable::Digest(mpt) ; 
    SDigest dig ;
    dig.update( const_cast<char*>(name.data()), name.size() );  
    dig.update( const_cast<char*>(dmpt.data()), dmpt.size() );  
    return dig.finalize();
}

GMaterial* X4Material::Convert( const G4Material* material )
{
    X4Material xmat(material);
    GMaterial* mat = xmat.getMaterial(); 
    return mat ; 
}

GMaterial* X4Material::getMaterial()
{
    return m_mat ; 
}

X4Material::X4Material( const G4Material* material ) 
   :
   m_material(material),
   m_mpt(material->GetMaterialPropertiesTable()),
   m_mat(NULL)
{
   init() ;
}

void X4Material::init()
{
    G4String name_ = m_material->GetName() ;
    const char* name = name_.c_str();
    unsigned index = m_material->GetIndex() ;

    // FORMERLY set the index on collecting into GMaterialLib, 
    // now are just passing the creation index along  

    m_mat = new GMaterial(name, index) ; 
    assert( m_mpt );

    X4MaterialPropertiesTable::Convert( m_mat, m_mpt );
}






