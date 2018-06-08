#include <string>
#include "G4Material.hh"
#include "X4PhysicsVector.hh"
#include "X4Material.hh"
#include "GMaterial.hh"

#include "PLOG.hh"


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
    unsigned index = 0 ;  // set the index on collecting into GMaterialLib
    m_mat = new GMaterial(name, index) ; 


    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = m_mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ;  
        // G4MaterialPropertyVector is typedef to G4PhysicsOrderedFreeVector with most of imp in G4PhysicsVector

        GProperty<float>* prop = X4PhysicsVector<float>::Convert(pvec) ; 


      //   m_mat->addProperty( pname.c_str(), prop );  // non-interpolating collection
        m_mat->addPropertyStandardized( pname.c_str(), prop );  // interpolates onto standard domain 

    }  


    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = m_mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4double pvalue = it->second ;  
        float value = pvalue ; 

        m_mat->addConstantProperty( pname.c_str(), value );   // asserts without standard domain
    }  
}


