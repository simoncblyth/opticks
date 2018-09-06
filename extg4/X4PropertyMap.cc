
#include "G4MaterialPropertyVector.hh"
#include "G4MaterialPropertiesTable.hh"

#include "X4PropertyMap.hh"
#include "X4Property.hh"

#include "GPropertyMap.hh"
#include "GProperty.hh"
#include "GMaterialLib.hh"


G4MaterialPropertiesTable* X4PropertyMap::Convert( const GPropertyMap<float>* pmap )
{  
    X4PropertyMap xpm(pmap); 
    return xpm.getMPT(); 
}

X4PropertyMap::X4PropertyMap(const GPropertyMap<float>* pmap) 
    :
    m_pmap(pmap),
    m_mpt(new G4MaterialPropertiesTable),
    m_mlib(GMaterialLib::GetInstance())
{
    init();
}

G4MaterialPropertiesTable* X4PropertyMap::getMPT() const 
{
    return m_mpt ; 
}

void X4PropertyMap::init()
{
    unsigned num_prop = m_pmap->getNumProperties();    
    for(unsigned i=0 ; i<num_prop ; i++) 
    {
         const char* key =  m_pmap->getPropertyNameByIndex(i);  // refractive_index absorption_length scattering_length reemission_prob
         const char* lkey = m_mlib->getLocalKey(key) ;      // RINDEX ABSLENGTH RAYLEIGH REEMISSIONPROB
         GProperty<float>* prop = m_pmap->getPropertyByIndex(i);

         G4PhysicsVector* pvec = X4Property<float>::Convert( prop ) ; 

         G4MaterialPropertyVector* mpv =  dynamic_cast<G4MaterialPropertyVector*>(pvec); 

         m_mpt->AddProperty( lkey, mpv ); 
    }
}






