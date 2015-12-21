#pragma once
#include <string>
#include <cstring>

#include "NPY.hpp"

class Timer ; 
class Parameters ;
class ViewNPY ;
class MultiViewNPY ;
class RecordsNPY ; 
class PhotonsNPY ; 

class NumpyEvt {
   public:
       NumpyEvt(const char* typ, const char* tag, const char* det, const char* cat=NULL);
   private:
       void init();
   public:
       static const char* genstep ;
       static const char* photon ;
       static const char* record  ;
       static const char* phosel ;
       static const char* recsel  ;
       static const char* sequence  ;
       static const char* aux ;
   public:
       NPY<float>* loadGenstepFromFile(int modulo=0);
       void setGenstepData(NPY<float>* genstep_data);
       void zero();
   public:
       void setIncomingData(NPY<float>* incoming_data);
   public:
       void setMaxRec(unsigned int maxrec);         // maximum record slots per photon
   public:
       // domains used for record compression
       void setCenterExtent(const glm::vec4& center_extent);
       void setTimeDomain(const glm::vec4& time_domain);
       void setBoundaryDomain(const glm::vec4& boundary_domain);
       const glm::vec4& getCenterExtent();
       const glm::vec4& getTimeDomain();
       const glm::vec4& getBoundaryDomain();
   private:
       void updateDomainsBuffer();
   public:
       void save(bool verbose=false);
   private:
       // invoked internally, as knock on from setGenstepData
       void createHostBuffers(); 
       void seedPhotonData();
       
       void setPhotonData(NPY<float>* photon_data);
       void setSequenceData(NPY<unsigned long long>* history_data);
       void setRecordData(NPY<short>* record_data);
       void setAuxData(NPY<short>* aux_data);
       void setRecselData(NPY<unsigned char>* recsel_data);
       void setPhoselData(NPY<unsigned char>* phosel_data);

   public:
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);
   public:
       NPY<float>*          getIncomingData();
       NPY<float>*          getGenstepData();
       NPY<float>*          getPhotonData();
       NPY<short>*          getRecordData();
       NPY<short>*          getAuxData();
       NPY<unsigned char>*  getPhoselData();
       NPY<unsigned char>*  getRecselData();
       NPY<unsigned long long>*  getSequenceData();

   public:
       // optionals lodged here for debug dumping single photons/records  
       void setRecordsNPY(RecordsNPY* recs);
       void setPhotonsNPY(PhotonsNPY* pho);
       RecordsNPY*          getRecordsNPY();
       PhotonsNPY*          getPhotonsNPY();
       NPY<float>*          getFDomain();
       NPY<int>*            getIDomain();
       void dumpDomains(const char* msg="NumpyEvt::dumpDomains");
   public:
       MultiViewNPY* getGenstepAttr();
       MultiViewNPY* getPhotonAttr();
       MultiViewNPY* getRecordAttr();
       MultiViewNPY* getAuxAttr();
       MultiViewNPY* getPhoselAttr();
       MultiViewNPY* getRecselAttr();
       MultiViewNPY* getSequenceAttr();

       ViewNPY* operator [](const char* spec);

   private:
   public:
       unsigned int getNumGensteps();
       unsigned int getNumPhotons();
       unsigned int getNumRecords();
       unsigned int getMaxRec();  // per-photon

   public:
       std::string  description(const char* msg);
       void         dumpPhotonData();
       static void  dumpPhotonData(NPY<float>* photon_data);

       const char*  getTyp();
       const char*  getTag();
       const char*  getDet();
       const char*  getCat();
   private:
       const char*           m_typ ; 
       const char*           m_tag ; 
       const char*           m_det ; 
       const char*           m_cat ; 

       Timer*                m_timer ;
       Parameters*           m_parameters ;

       NPY<float>*           m_incoming_data ; 
       NPY<float>*           m_genstep_data ;
       NPY<float>*           m_photon_data ;
       NPY<short>*           m_record_data ;
       NPY<short>*           m_aux_data ;
       NPY<unsigned char>*   m_phosel_data ;
       NPY<unsigned char>*   m_recsel_data ;
       NPY<unsigned long long>*  m_sequence_data ;

       NPY<float>*           m_fdom ; 
       NPY<int>*             m_idom ; 

       MultiViewNPY*   m_genstep_attr ;
       MultiViewNPY*   m_photon_attr  ;
       MultiViewNPY*   m_record_attr  ;
       MultiViewNPY*   m_aux_attr  ;
       MultiViewNPY*   m_phosel_attr  ;
       MultiViewNPY*   m_recsel_attr  ;
       MultiViewNPY*   m_sequence_attr  ;

       RecordsNPY*     m_records ; 
       PhotonsNPY*     m_photons ; 

       unsigned int    m_num_gensteps ; 
       unsigned int    m_num_photons ; 
       unsigned int    m_maxrec ; 

       glm::vec4       m_center_extent ; 
       glm::vec4       m_time_domain ; 
       glm::vec4       m_boundary_domain ; 


};


inline NumpyEvt::NumpyEvt(const char* typ, const char* tag, const char* det, const char* cat) 
          :
          m_typ(strdup(typ)),
          m_tag(strdup(tag)),
          m_det(strdup(det)),
          m_cat(strdup(cat)),

          m_timer(NULL),
          m_parameters(NULL),

          m_incoming_data(NULL),
          m_genstep_data(NULL),
          m_photon_data(NULL),
          m_record_data(NULL),
          m_aux_data(NULL),
          m_phosel_data(NULL),
          m_recsel_data(NULL),
          m_sequence_data(NULL),

          m_fdom(NULL),
          m_idom(NULL),

          m_genstep_attr(NULL),
          m_photon_attr(NULL),
          m_record_attr(NULL),
          m_aux_attr(NULL),
          m_phosel_attr(NULL),
          m_recsel_attr(NULL),
          m_sequence_attr(NULL),

          m_records(NULL),
          m_photons(NULL),
          m_num_gensteps(0),
          m_num_photons(0),
          m_maxrec(1)
{
    init();
}






inline unsigned int NumpyEvt::getNumGensteps()
{
    return m_num_gensteps ; 
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

inline NPY<short>* NumpyEvt::getAuxData()
{
    return m_aux_data ;
}
inline MultiViewNPY* NumpyEvt::getAuxAttr()
{
    return m_aux_attr ;
}





inline NPY<unsigned char>* NumpyEvt::getPhoselData()
{
    return m_phosel_data ;
}
inline MultiViewNPY* NumpyEvt::getPhoselAttr()
{
    return m_phosel_attr ;
}


inline NPY<unsigned char>* NumpyEvt::getRecselData()
{
    return m_recsel_data ;
}
inline MultiViewNPY* NumpyEvt::getRecselAttr()
{
    return m_recsel_attr ;
}




inline NPY<unsigned long long>* NumpyEvt::getSequenceData()
{
    return m_sequence_data ;
}
inline MultiViewNPY* NumpyEvt::getSequenceAttr()
{
    return m_sequence_attr ;
}




inline NPY<float>* NumpyEvt::getIncomingData()
{
    return m_incoming_data ;
}

inline void NumpyEvt::setIncomingData(NPY<float>* incoming_data)
{
    m_incoming_data = incoming_data ;
}

inline void NumpyEvt::setRecordsNPY(RecordsNPY* records)
{
    m_records = records ; 
}
inline RecordsNPY* NumpyEvt::getRecordsNPY()
{
    return m_records ;
}

inline void NumpyEvt::setPhotonsNPY(PhotonsNPY* photons)
{
    m_photons = photons ; 
}
inline PhotonsNPY* NumpyEvt::getPhotonsNPY()
{
    return m_photons ;
}




inline void NumpyEvt::setFDomain(NPY<float>* fdom)
{
    m_fdom = fdom ; 
}
inline void NumpyEvt::setIDomain(NPY<int>* idom)
{
    m_idom = idom ; 
}

inline NPY<float>* NumpyEvt::getFDomain()
{
    return m_fdom ; 
}
inline NPY<int>* NumpyEvt::getIDomain()
{
    return m_idom ; 
}

inline const char* NumpyEvt::getTyp()
{
    return m_typ ; 
}
inline const char* NumpyEvt::getTag()
{
    return m_tag ; 
}
inline const char* NumpyEvt::getDet()
{
    return m_det ; 
}
inline const char* NumpyEvt::getCat()
{
    return m_cat ; 
}


inline void NumpyEvt::setCenterExtent(const glm::vec4& center_extent)
{
    m_center_extent = center_extent ; 
}
inline void NumpyEvt::setTimeDomain(const glm::vec4& time_domain)
{
    m_time_domain = time_domain  ; 
}
inline void NumpyEvt::setBoundaryDomain(const glm::vec4& boundary_domain)
{
    m_boundary_domain = boundary_domain  ; 
}


inline const glm::vec4& NumpyEvt::getCenterExtent()
{
    return m_center_extent ; 
}
inline const glm::vec4& NumpyEvt::getTimeDomain()
{
    return m_time_domain ;
}
inline const glm::vec4& NumpyEvt::getBoundaryDomain()
{ 
    return m_boundary_domain ; 
}
 
