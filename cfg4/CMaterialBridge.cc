#include "BStr.hh"
#include <iomanip>

#include "G4StepPoint.hh"
#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "GMaterialLib.hh"
#include "CStep.hh"
#include "CMaterialBridge.hh"

#include "PLOG.hh"


const plog::Severity CMaterialBridge::LEVEL = PLOG::EnvLevel("CMaterialBridge", "DEBUG") ; 


CMaterialBridge::CMaterialBridge( GMaterialLib* mlib) 
    :
    m_mlib(mlib)
{
    initMap();
    if(LEVEL > info)
        dump("CMaterialBridge::CMaterialBridge");
}


void CMaterialBridge::initMap()
{
    const G4MaterialTable* mtab = G4Material::GetMaterialTable();
    unsigned nmat = G4Material::GetNumberOfMaterials();
    LOG(LEVEL) << " nmat (G4Material::GetNumberOfMaterials) " << nmat ; 

    for(unsigned i=0 ; i < nmat ; i++)
    {
        const G4Material* material = (*mtab)[i];

        std::string name = material->GetName() ;

        const char* shortname = BStr::afterLastOrAll( name.c_str(), '/' );

        std::string abbr = shortname ; //  

        unsigned index =  m_mlib->getIndex( shortname );

        m_g4toix[material] = index ; 

        m_ixtoname[index] = shortname ;

        m_ixtoabbr[index] = m_mlib->getAbbr(shortname) ;


        pLOG(LEVEL,+1) << " i " << std::setw(3) << i 
                  << " name " << std::setw(35) << name 
                  << " shortname " << std::setw(35) << shortname 
                  << " abbr " << std::setw(35) << abbr 
                  << " index " << std::setw(5)  << index
                  ; 
    }

    LOG(LEVEL)
        << " nmat " << nmat 
        << " m_g4toix.size() "   << m_g4toix.size() 
        << " m_ixtoname.size() " << m_ixtoname.size() 
        ; 


    assert( m_g4toix.size() == nmat );
    assert( m_ixtoname.size() == nmat && "there is probably a duplicated material name");
    assert( m_ixtoabbr.size() == nmat && "there is probably a duplicated material name");
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
        const char* shortname = getMaterialName(index, false);
        const char* abbr = getMaterialName(index, true);

        std::cout << std::setw(50) << mat->GetName() 
                  << std::setw(10) << index 
                  << std::setw(30) << shortname 
                  << std::setw(30) << abbr
                  << std::endl ; 
    }
}


bool CMaterialBridge::operator()(const G4Material* a, const G4Material* b)
{
    unsigned ia = getMaterialIndex(a);
    unsigned ib = getMaterialIndex(b);
    return ia < ib ; 
}




unsigned CMaterialBridge::getPreMaterial(const G4Step* step) const
{
    const G4Material* preMat  = CStep::PreMaterial(step);
    unsigned preMaterial = preMat ? getMaterialIndex(preMat) + 1 : 0 ;
    return preMaterial ; 
}

unsigned CMaterialBridge::getPostMaterial(const G4Step* step) const
{
    const G4Material* postMat  = CStep::PostMaterial(step);
    unsigned postMaterial = postMat ? getMaterialIndex(postMat) + 1 : 0 ;
    return postMaterial ;
}

unsigned CMaterialBridge::getPointMaterial(const G4StepPoint* point) const
{
    const G4Material* pointMat  = point->GetMaterial() ;
    unsigned pointMaterial = pointMat ? getMaterialIndex(pointMat) + 1 : 0 ;
    return pointMaterial ;
}


unsigned int CMaterialBridge::getMaterialIndex(const G4Material* material) const 
{
    // used from CSteppingAction::UserSteppingActionOptical to CRecorder::setBoundaryStatus
    return m_g4toix.at(material) ;
}
const char* CMaterialBridge::getMaterialName(unsigned int index, bool abbrev)
{
    return abbrev ? m_ixtoabbr[index].c_str() : m_ixtoname[index].c_str() ;
}


const G4Material* CMaterialBridge::getG4Material(unsigned int qindex) // 0-based Opticks material index to G4Material
{
    typedef std::map<const G4Material*, unsigned> MU ; 
    const G4Material* mat = NULL ; 

    std::stringstream ss ; 

    for(MU::const_iterator it=m_g4toix.begin() ; it != m_g4toix.end() ; it++)
    {
         unsigned index = it->second ; 
         ss << index << " " ; 
         if(index == qindex)
         {
             mat = it->first ; 
             break ;
         }
    }


    std::string indices = ss.str(); 
 
    if( mat == NULL )
    {
        LOG(fatal) 
             << " failed to find a G4Material with index " << qindex 
             << " in all the indices " << indices 
             ;
    }

    return mat ; 
}


std::string CMaterialBridge::MaterialSequence(unsigned long long seqmat, bool abbrev)
{
    std::stringstream ss ;
    assert(sizeof(unsigned long long)*8 == 16*4);
    for(unsigned int i=0 ; i < 16 ; i++)
    {   
        unsigned long long msk = (seqmat >> i*4) & 0xF ; 

        unsigned int idx = unsigned(msk - 1);    // -> 0-based

        ss << ( msk > 0 ? getMaterialName(idx, abbrev) : "-" ) << " " ;
        // using 1-based material indices, so 0 represents None
    }   
    return ss.str();
}

