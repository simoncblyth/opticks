#include "GPropertyMap.hh"

#include "G4MaterialPropertiesTable.hh"
#include "X4MaterialPropertiesTable.hh"
#include "X4PhysicsVector.hh"


GPropertyMap<float>* X4MaterialPropertiesTable::Convert(const G4MaterialPropertiesTable* mpt) 
{
    X4MaterialPropertiesTable xmpt(mpt) ; 
    GPropertyMap<float>* pmap = xmpt.getPropertyMap();
    return pmap ; 
}


X4MaterialPropertiesTable::X4MaterialPropertiesTable(const G4MaterialPropertiesTable* mpt)
   :
   m_mpt(mpt),
   m_pmap(new GPropertyMap<float>("MaterialPropertiesTable"))
{
   init();
}

GPropertyMap<float>* X4MaterialPropertiesTable::getPropertyMap()
{
   return m_pmap ;  
}

void X4MaterialPropertiesTable::init()
{
    typedef const std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MKP ;
    MKP* pm = m_mpt->GetPropertiesMap() ;

    for(MKP::const_iterator it=pm->begin() ; it != pm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4MaterialPropertyVector* pvec = it->second ;  
        // G4MaterialPropertyVector is typedef to G4PhysicsOrderedFreeVector with most of imp in G4PhysicsVector

        GProperty<float>* prop = X4PhysicsVector<float>::Convert(pvec) ; 

        //GAry<float>* dom = prop->getDomain(); 

        m_pmap->addProperty( pname.c_str(), prop );  // non-interpolating  
    }  


    typedef const std::map< G4String, G4double, std::less<G4String> > CKP ; 
    CKP* cm = m_mpt->GetPropertiesCMap();

    for(CKP::const_iterator it=cm->begin() ; it != cm->end() ; it++)
    {   
        G4String pname = it->first ;
        G4double pvalue = it->second ;  
        float value = pvalue ; 

        m_pmap->addConstantProperty( pname.c_str(), value );   // asserts without standard domain
    }  
}

