#include <cstring>
#include <iostream>
#include <iomanip>
#include <vector>

#include "PLOG.hh"

#include "G4Material.hh"
#include "G4MaterialTable.hh"
#include "G4MaterialPropertiesTable.hh"

#include "CMPT.hh"
#include "CMaterialTable.hh"


const plog::Severity CMaterialTable::LEVEL = debug ; 


CMaterialTable::CMaterialTable(const char* prefix)
    :
    m_prefix(prefix ? strdup(prefix) : NULL)
{
    init();
}
void CMaterialTable::init()
{
    initNameIndex();
}

void CMaterialTable::initNameIndex()
{
    const G4MaterialTable* mtab = G4Material::GetMaterialTable();
    unsigned nmat = G4Material::GetNumberOfMaterials();

    LOG(LEVEL)
              << " numOfMaterials " << nmat
              << " prefix " << ( m_prefix ? m_prefix : "NULL" ) 
              ;
    
    for(unsigned i=0 ; i < nmat ; i++)
    {
        G4Material* material = (*mtab)[i];
        G4String name_ = material->GetName() ;
        const char* name = name_.c_str();
        const char* shortname =  m_prefix && strncmp(name, m_prefix, strlen(m_prefix)) == 0 ? name + strlen(m_prefix) : name ; 

        pLOG(LEVEL,+1) 
            << " index " << std::setw(3) << i 
            << " name " << std::setw(30) << name
            << " shortname " << std::setw(30) << shortname
            ;

        m_name2index[shortname] = i ;   
        m_index2name[i] = shortname ;   
    }
}

void CMaterialTable::dump(const char* msg)
{
    LOG(info) << msg << " prefix " << m_prefix ; 

    typedef std::map<unsigned, std::string> MUS ; 
    for(MUS::const_iterator it=m_index2name.begin() ; it != m_index2name.end() ; it++)
        std::cout 
             << std::setw(35) << it->first 
             << std::setw(25) << it->second
             << std::endl ; 

}

void CMaterialTable::fillMaterialIndexMap( std::map<std::string, unsigned int>&  mixm )
{
    typedef std::map<std::string, unsigned> MSU ; 
    for(MSU::const_iterator it=m_name2index.begin() ; it != m_name2index.end() ; it++)
    {
         std::string name = it->first ; 
         unsigned index = it->second ; 
         mixm[name] = index ;  
    }
}

const std::map<std::string, unsigned>& CMaterialTable::getMaterialMap() const 
{
   return m_name2index ;  
}


unsigned CMaterialTable::getMaterialIndex(const char* shortname)
{
    return m_name2index.count(shortname) == 1 ? m_name2index[shortname] : -1 ; 
}

void CMaterialTable::dumpMaterial(const char* shortname)
{
     unsigned index = getMaterialIndex(shortname);
     dumpMaterial(index);
}


void CMaterialTable::dumpMaterial(unsigned index)
{
    const G4MaterialTable* theMaterialTable = G4Material::GetMaterialTable();
    unsigned numOfMaterials = G4Material::GetNumberOfMaterials();

    G4Material* material = index < numOfMaterials ? (*theMaterialTable)[index] : NULL ;
    dumpMaterial(material);
}


void CMaterialTable::dumpMaterial(G4Material* material)
{
    if(!material) return ; 
    G4String name = material->GetName() ;

    CMPT mpt(material->GetMaterialPropertiesTable());

    LOG(info) << "CMaterialTable::dumpMaterial "
              << name 
              ;


    mpt.dump("MPT:");

    mpt.dumpProperty("RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB");

}




