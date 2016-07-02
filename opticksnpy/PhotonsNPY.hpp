#pragma once

template <typename T> class NPY ; 
class Typ ; 
class Types ; 
class RecordsNPY ; 
class Index ; 

// detailed host based photon and record dumper 

#include "NPY_API_EXPORT.hh"
class NPY_API PhotonsNPY {
   public:  
       PhotonsNPY(NPY<float>* photons); 
   public:  
       void                  setTypes(Types* types);
       void                  setTyp(Typ* typ);
       void                  setRecs(RecordsNPY* recs);
   public:  
       NPY<float>*           make_pathinfo();
       NPY<float>*           getPhotons();
       RecordsNPY*           getRecs();
       Types*                getTypes();

   public:  
       void dump(unsigned int photon_id, const char* msg="PhotonsNPY::dump");
   public:  
       void dumpPhotonRecord(unsigned int photon_id, const char* msg="phr");
       void dumpPhoton(unsigned int i, const char* msg="pho");
   public:  
       void dumpPhotons(const char* msg="PhotonsNPY::dumpPhotons", unsigned int ndump=5);
   public:
       void debugdump(const char* msg);

   private:
       NPY<float>*                  m_photons ; 
       bool                         m_flat ;
       RecordsNPY*                  m_recs ; 
       Types*                       m_types ; 
       Typ*                         m_typ ; 
       unsigned int                 m_maxrec ; 

};

