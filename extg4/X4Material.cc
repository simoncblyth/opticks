#include <string>
#include "G4Material.hh"
#include "X4PhysicsVector.hh"
#include "X4Material.hh"
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
    std::string dmpt = Digest(mpt) ; 
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
    unsigned index = 0 ;  // set the index on collecting into GMaterialLib
    m_mat = new GMaterial(name, index) ; 

    AddProperties( m_mat, m_mpt ); 
}


void X4Material::AddProperties(GMaterial* mat, const G4MaterialPropertiesTable* mpt)
{

    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ;  
        // G4MaterialPropertyVector is typedef to G4PhysicsOrderedFreeVector with most of imp in G4PhysicsVector

        GProperty<float>* prop = X4PhysicsVector<float>::Convert(pvec) ; 


      //   mat->addProperty( pname.c_str(), prop );  // non-interpolating collection
        mat->addPropertyStandardized( pname.c_str(), prop );  // interpolates onto standard domain 

    }  

    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4double pvalue = it->second ;  
        float value = pvalue ; 

        mat->addConstantProperty( pname.c_str(), value );   // asserts without standard domain
    }     
}


std::string X4Material::Digest(const G4MaterialPropertiesTable* mpt)  
{
    if(!mpt) return "" ; 

    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    SDigest dig ;
    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)  
    {   
        const std::string&  n = it->first ;
        G4MaterialPropertyVector* v = it->second ; 

        std::string vs = X4PhysicsVector<float>::Digest(v) ; 
        dig.update( const_cast<char*>(n.data()),  n.size() );  
        dig.update( const_cast<char*>(vs.data()), vs.size() );  
    }   

    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        const std::string& n = it->first ;
        double pvalue = it->second ;  

        dig.update( const_cast<char*>(n.data()), n.size() );  
        dig.update( reinterpret_cast<char*>(&pvalue), sizeof(double) );  
    }  
    return dig.finalize();
}




