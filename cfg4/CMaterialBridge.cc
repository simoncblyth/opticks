#include "BStr.hh"
#include <iomanip>

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "GMaterialLib.hh"
#include "CMaterialBridge.hh"

#include "PLOG.hh"

CMaterialBridge::CMaterialBridge( GMaterialLib* mlib) 
    :
    m_mlib(mlib)
{
    initMap();
//    dump("CMaterialBridge::CMaterialBridge");
}


void CMaterialBridge::initMap()
{
    const G4MaterialTable* mtab = G4Material::GetMaterialTable();
    unsigned nmat = G4Material::GetNumberOfMaterials();

    LOG(info) << "CMaterialBridge::initMap" 
              << " nmat (G4Material::GetNumberOfMaterials) " << nmat 
              ;

    for(unsigned i=0 ; i < nmat ; i++)
    {
        const G4Material* material = (*mtab)[i];

        std::string name = material->GetName() ;

        const char* shortname = BStr::afterLastOrAll( name.c_str(), '/' );

        unsigned index =  m_mlib->getIndex( shortname );

        m_g4toix[material] = index ; 

        m_ixtoname[index] = shortname ;


        LOG(info) << " i " << std::setw(3) << i 
                  << " name " << std::setw(35) << name 
                  << " shortname " << std::setw(35) << shortname 
                  << " index " << std::setw(5)  << index
                  ; 
    }

    LOG(info)
            << " nmat " << nmat 
            << " m_g4toix.size() "   << m_g4toix.size() 
            << " m_ixtoname.size() " << m_ixtoname.size() 
             ; 


    assert( m_g4toix.size() == nmat );
    assert( m_ixtoname.size() == nmat && "there is probably a duplicated material name");
}



void CMaterialBridge::dumpMap(const char* msg)
{
    LOG(info) << msg << " g4toix.size " << m_g4toix.size() ;

    typedef std::map<const G4Material*, unsigned> MU ; 
    for(MU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
         const G4Material* mat = it->first ; 
         unsigned index = it->second ; 

         std::cout << std::setw(50) << mat->GetName() 
                   << std::setw(10) << index 
                   << std::endl ; 

         unsigned check = getMaterialIndex(mat);
         assert(check == index);
    }
}


void CMaterialBridge::dump(const char* msg)
{
    LOG(info) << msg << " g4toix.size " << m_g4toix.size() ;

    typedef std::vector<const G4Material*> M ; 
    M materials ; 

    typedef std::map<const G4Material*, unsigned> MU ; 
    for(MU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++) 
         materials.push_back(it->first);

    std::stable_sort( materials.begin(), materials.end(), *this );          

    for(M::const_iterator it=materials.begin() ; it != materials.end() ; it++)
    {
        const G4Material* mat = *it ;  
        unsigned index = getMaterialIndex(mat);
        const char* shortname = getMaterialName(index);

        std::cout << std::setw(50) << mat->GetName() 
                  << std::setw(10) << index 
                  << std::setw(30) << shortname 
                  << std::endl ; 
    }
}


bool CMaterialBridge::operator()(const G4Material* a, const G4Material* b)
{
    unsigned ia = getMaterialIndex(a);
    unsigned ib = getMaterialIndex(b);
    return ia < ib ; 
}



unsigned int CMaterialBridge::getMaterialIndex(const G4Material* material)
{
    // used from CSteppingAction::UserSteppingActionOptical to CRecorder::setBoundaryStatus
    return m_g4toix[material] ;
}
const char* CMaterialBridge::getMaterialName(unsigned int index)
{
    return m_ixtoname[index].c_str() ;
}



std::string CMaterialBridge::MaterialSequence(unsigned long long seqmat)
{
    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
        unsigned long long msk = (seqmat >> i*4) & 0xF ; 

        unsigned int idx = unsigned(msk - 1);    // -> 0-based

        ss << ( msk > 0 ? getMaterialName(idx) : "-" ) << " " ;
        // using 1-based material indices, so 0 represents None
    }   
    return ss.str();
}

