
#include "G4MaterialPropertiesTable.hh"

#include "X4MaterialPropertiesTable.hh"
#include "X4PhysicsVector.hh"

#include "SDigest.hh"
#include "GPropertyMap.hh"
#include "GProperty.hh"
#include "PLOG.hh"


void X4MaterialPropertiesTable::Convert( GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt )
{
    X4MaterialPropertiesTable xtab(pmap, mpt);
}

X4MaterialPropertiesTable::X4MaterialPropertiesTable( GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt )
    :
    m_pmap(pmap),
    m_mpt(mpt)
{
    init();
}

void X4MaterialPropertiesTable::init()
{ 
    AddProperties( m_pmap, m_mpt );    
}


void X4MaterialPropertiesTable::AddProperties(GPropertyMap<float>* pmap, const G4MaterialPropertiesTable* const mpt)   // static
{
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {   
        G4String pname = it->first ;

        LOG(error) << pname ; 

        G4MaterialPropertyVector* pvec = it->second ;  
        // G4MaterialPropertyVector is typedef to G4PhysicsOrderedFreeVector with most of imp in G4PhysicsVector

        GProperty<float>* prop = X4PhysicsVector<float>::Convert(pvec) ; 


      //   pmap->addProperty( pname.c_str(), prop );  // non-interpolating collection
        pmap->addPropertyStandardized( pname.c_str(), prop );  // interpolates onto standard domain 

    }  

    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4double pvalue = it->second ;  
        float value = pvalue ; 

        pmap->addConstantProperty( pname.c_str(), value );   // asserts without standard domain
    }     
}

std::string X4MaterialPropertiesTable::Digest(const G4MaterialPropertiesTable* mpt)  // static
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


