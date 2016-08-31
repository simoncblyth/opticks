#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "PLOG.hh"

#include "CMaterialTable.hh"
#include "G4Material.hh"
#include "G4MaterialTable.hh"

CMaterialTable::CMaterialTable(const char* prefix)
    :
    m_prefix(strdup(prefix)) 
{
    init();
}
   
void CMaterialTable::init()
{
    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    unsigned numOfMaterials = G4Material::GetNumberOfMaterials();

    LOG(info) << "CMaterialTable::init "
              << " numOfMaterials " << numOfMaterials
              << " prefix " << m_prefix 
              ;
    
    for(unsigned i=0 ; i < numOfMaterials ; i++)
    {
        G4Material* material = (*theMaterialTable)[i];
        G4String name_ = material->GetName() ;
        const char* name = name_.c_str();

        if(strncmp(name, m_prefix, strlen(m_prefix)) == 0)
        {
            const char* shortname = name + strlen(m_prefix) ;
            m_name2index[shortname] = i ;   
            m_index2name[i] = shortname ;   
        }
        else
        {
            LOG(warning) << "CMaterialTable::init material with incorrect prefix " 
                         << " name " << name
                         << " prefix " << m_prefix 
                         ; 
        }
    }

    assert(m_name2index.size() == m_index2name.size());
}

void CMaterialTable::dump(const char* msg)
{
    LOG(info) << msg << " prefix " << m_prefix ; 

/*
    typedef std::map<std::string, unsigned> MSU ; 
    for(MSU::const_iterator it=m_name2index.begin() ; it != m_name2index.end() ; it++)
        std::cout 
             << std::setw(35) << it->first 
             << std::setw(25) << it->second
             << std::endl ; 
*/

    typedef std::map<unsigned, std::string> MUS ; 
    for(MUS::const_iterator it=m_index2name.begin() ; it != m_index2name.end() ; it++)
        std::cout 
             << std::setw(35) << it->first 
             << std::setw(25) << it->second
             << std::endl ; 

}


