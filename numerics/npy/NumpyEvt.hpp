#pragma once
#include <string>

#include "NPY.hpp"
class MultiVecNPY ;

class NumpyEvt {
   public:
       NumpyEvt();
      
   public:
       void setGenstepData(NPY<float>* genstep_data);
   private:
       // only set internally 
       void setPhotonData(NPY<float>* photon_data);
       void setRecordData(NPY<short>* record_data);

   public:
       NPY<float>*  getGenstepData();
       NPY<float>*  getPhotonData();
       NPY<short>*  getRecordData();

       MultiVecNPY* getGenstepAttr();
       MultiVecNPY* getPhotonAttr();
       MultiVecNPY* getRecordAttr();

   public:
       unsigned int getNumPhotons();
       std::string description(const char* msg);

   public:
       void         dumpPhotonData();
       static void  dumpPhotonData(NPY<float>* photon_data);

   private:
       NPY<float>*   m_genstep_data ;
       NPY<float>*   m_photon_data ;
       NPY<short>*   m_record_data ;

       MultiVecNPY*   m_genstep_attr ;
       MultiVecNPY*   m_photon_attr  ;
       MultiVecNPY*   m_record_attr  ;

       unsigned int   m_num_photons ; 

};


inline NumpyEvt::NumpyEvt() 
          :
          m_genstep_data(NULL),
          m_photon_data(NULL),
          m_record_data(NULL),
          m_genstep_attr(NULL),
          m_photon_attr(NULL),
          m_record_attr(NULL),
          m_num_photons(0)
{
}

inline NPY<float>* NumpyEvt::getGenstepData()
{
    return m_genstep_data ;
}
inline MultiVecNPY* NumpyEvt::getGenstepAttr()
{
    return m_genstep_attr ;
}

inline NPY<float>* NumpyEvt::getPhotonData()
{
    return m_photon_data ;
}
inline MultiVecNPY* NumpyEvt::getPhotonAttr()
{
    return m_photon_attr ;
}

inline NPY<short>* NumpyEvt::getRecordData()
{
    return m_record_data ;
}
inline MultiVecNPY* NumpyEvt::getRecordAttr()
{
    return m_record_attr ;
}


inline unsigned int NumpyEvt::getNumPhotons()
{
    return m_num_photons ; 
}





