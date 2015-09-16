#pragma once

#include "glm/fwd.hpp"
#include <map>
#include <string>
#include <vector>

#include "Types.hpp"
#include "NPY.hpp"

class RecordsNPY ; 
class Index ; 

//
// detailed host based photon and record dumper 
//
class PhotonsNPY {
   public:  
       PhotonsNPY(NPY<float>* photons); 
   public:  
       void                  setTypes(Types* types);
       void                  setRecs(RecordsNPY* recs);
   public:  
       NPY<float>*           getPhotons();
       RecordsNPY*           getRecs();
       Types*                getTypes();

   public:  
       void dump(unsigned int photon_id);
   public:  
       void dumpPhotonRecord(unsigned int photon_id, const char* msg="phr");
       void dumpPhoton(unsigned int i, const char* msg="pho");
   public:  
       void dumpPhotons(const char* msg="PhotonsNPY::dumpPhotons", unsigned int ndump=5);
       //glm::ivec4 getFlags();
   public:
       void debugdump(const char* msg);

   private:
       NPY<float>*                  m_photons ; 
       RecordsNPY*                  m_recs ; 
       Types*                       m_types ; 
       unsigned int                 m_maxrec ; 

};

inline PhotonsNPY::PhotonsNPY(NPY<float>* photons) 
       :  
       m_photons(photons),
       m_recs(NULL),
       m_types(NULL),
       m_maxrec(0)
{
}

inline void PhotonsNPY::setTypes(Types* types)
{  
    m_types = types ; 
}
inline NPY<float>* PhotonsNPY::getPhotons()
{
    return m_photons ; 
}
inline RecordsNPY* PhotonsNPY::getRecs()
{
    return m_recs ; 
}
inline Types* PhotonsNPY::getTypes()
{
    return m_types ; 
}



