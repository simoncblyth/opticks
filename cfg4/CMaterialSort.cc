
#include "BFile.hh"
#include "CMaterialSort.hh"
#include "G4Material.hh"
#include "PLOG.hh"

CMaterialSort::CMaterialSort(const std::map<std::string, unsigned>& order )
    :
    m_order(order),
    m_mtab(G4Material::GetMaterialTable()),
    m_dbg(false)
{
    init();
}

void CMaterialSort::init()
{
    if(m_dbg)
    {
        dumpOrder("order from ctor argument"); 
        dump("before");
    }

    sort(); 

    if(m_dbg)
    {
        dump("after");
    }
}


void CMaterialSort::dumpOrder(const char* msg) const
{
    LOG(info) << msg ; 
    for( MSU::const_iterator it=m_order.begin() ; it != m_order.end() ; it++) 
        std::cout 
           << " v " << std::setw(5) << it->second   
           << " k " << std::setw(30) << it->first   
           << std::endl ; 
}


void CMaterialSort::dump(const char* msg) const 
{
    LOG(info) << msg << " size : " << m_mtab->size() << " G4 materials from G4Material::GetMaterialTable " ;  
    for( unsigned i=0 ; i < m_mtab->size() ; i++)
    {
        std::string name = (*m_mtab)[i]->GetName() ; 
        LOG(info) 
           << " i " << std::setw(3) << i 
           << " name " << std::setw(3) << name
           ;
    }
}

void CMaterialSort::sort()
{
    if(m_order.size() == 0) 
    {
        LOG(verbose) << " SKIP sorting G4MaterialTable as order.size() zero " << m_order.size()  ; 
        return ;
    }
    LOG(fatal) << " sorting G4MaterialTable using order kv " << m_order.size()  ; 
    std::stable_sort( m_mtab->begin(), m_mtab->end(), *this );
}

bool CMaterialSort::operator()(const G4Material* a_, const G4Material* b_)
{
    const std::string& a = a_->GetName();
    const std::string& b = b_->GetName();

    std::string aa = BFile::Name(a.c_str()) ; 
    std::string bb = BFile::Name(b.c_str()) ; 

    MSU::const_iterator end = m_order.end() ; 
    unsigned ia = m_order.find(aa) == end ? UINT_MAX :  m_order.at(aa) ; 
    unsigned ib = m_order.find(bb) == end ? UINT_MAX :  m_order.at(bb) ; 

    return ia < ib ; 
}



