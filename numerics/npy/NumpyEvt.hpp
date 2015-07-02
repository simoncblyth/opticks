#pragma once
#include <string>

#include "NPY.hpp"
class ViewNPY ;
class MultiViewNPY ;

class NumpyEvt {
   public:
       NumpyEvt();

       static const char* genstep ;
       static const char* photon ;
       static const char* record  ;
      
       typedef unsigned long long Sequence_t ;

   public:
       void setGenstepData(NPY<float>* genstep_data);
       void setMaxRec(unsigned int maxrec);         // maximum record slots per photon
   public:
       void setRecselData(NPY<unsigned char>* recsel_data);
   private:
       // only set internally 
       void setPhotonData(NPY<float>* photon_data);
       void setSequenceData(NPY<Sequence_t>* history_data);
       void setRecordData(NPY<short>* record_data);

   public:
       NPY<float>*  getGenstepData();
       NPY<float>*  getPhotonData();
       NPY<short>*  getRecordData();
       NPY<unsigned char>*  getRecselData();
       NPY<Sequence_t>*  getSequenceData();

       MultiViewNPY* getGenstepAttr();
       MultiViewNPY* getPhotonAttr();
       MultiViewNPY* getRecordAttr();
       MultiViewNPY* getRecselAttr();
       MultiViewNPY* getSequenceAttr();

       ViewNPY* operator [](const char* spec);

   public:
       unsigned int getNumPhotons();
       unsigned int getNumRecords();
       unsigned int getMaxRec();  // per-photon
       std::string description(const char* msg);

   public:
       void         dumpPhotonData();
       static void  dumpPhotonData(NPY<float>* photon_data);

   private:
       NPY<float>*   m_genstep_data ;
       NPY<float>*   m_photon_data ;
       NPY<short>*   m_record_data ;
       NPY<unsigned char>*   m_recsel_data ;
       NPY<Sequence_t>*   m_sequence_data ;

       MultiViewNPY*   m_genstep_attr ;
       MultiViewNPY*   m_photon_attr  ;
       MultiViewNPY*   m_record_attr  ;
       MultiViewNPY*   m_recsel_attr  ;
       MultiViewNPY*   m_sequence_attr  ;

       unsigned int   m_num_photons ; 
       unsigned int   m_maxrec ; 

};


inline NumpyEvt::NumpyEvt() 
          :
          m_genstep_data(NULL),
          m_photon_data(NULL),
          m_record_data(NULL),
          m_recsel_data(NULL),
          m_sequence_data(NULL),
          m_genstep_attr(NULL),
          m_photon_attr(NULL),
          m_record_attr(NULL),
          m_recsel_attr(NULL),
          m_sequence_attr(NULL),
          m_num_photons(0),
          m_maxrec(1)
{
}



inline unsigned int NumpyEvt::getNumPhotons()
{
    return m_num_photons ; 
}
inline unsigned int NumpyEvt::getNumRecords()
{
    return m_num_photons * m_maxrec ; 
}
inline unsigned int NumpyEvt::getMaxRec()
{
    return m_maxrec ; 
}
inline void NumpyEvt::setMaxRec(unsigned int maxrec)
{
    m_maxrec = maxrec ; 
}





inline NPY<float>* NumpyEvt::getGenstepData()
{
    return m_genstep_data ;
}
inline MultiViewNPY* NumpyEvt::getGenstepAttr()
{
    return m_genstep_attr ;
}

inline NPY<float>* NumpyEvt::getPhotonData()
{
    return m_photon_data ;
}
inline MultiViewNPY* NumpyEvt::getPhotonAttr()
{
    return m_photon_attr ;
}

inline NPY<short>* NumpyEvt::getRecordData()
{
    return m_record_data ;
}
inline MultiViewNPY* NumpyEvt::getRecordAttr()
{
    return m_record_attr ;
}



inline NPY<unsigned char>* NumpyEvt::getRecselData()
{
    return m_recsel_data ;
}
inline MultiViewNPY* NumpyEvt::getRecselAttr()
{
    return m_recsel_attr ;
}


inline NPY<NumpyEvt::Sequence_t>* NumpyEvt::getSequenceData()
{
    return m_sequence_data ;
}
inline MultiViewNPY* NumpyEvt::getSequenceAttr()
{
    return m_sequence_attr ;
}


