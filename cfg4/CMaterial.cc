#include "G4Material.hh"

#include "SDigest.hh"

#include "CMPT.hh"
#include "CMaterial.hh"

std::string CMaterial::Digest(G4Material* material)
{
    if(!material) return "" ; 
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable() ; 
    const G4String& name = material->GetName();      
    std::string dmpt = CMPT::Digest(mpt) ; 
    SDigest dig ;
    dig.update( const_cast<char*>(name.data()), name.size() );  
    dig.update( const_cast<char*>(dmpt.data()), dmpt.size() );  
    return dig.finalize();
}
    
CMaterial::CMaterial(G4Material* mat) 
   :
   m_material(mat)
{
}

G4Material* CMaterial::getMaterial() const 
{
    return m_material ; 
}

std::string CMaterial::digest() const 
{
    return Digest(m_material); 
}


