#pragma once

#include "glm/fwd.hpp"

#include <map>
#include <string>
#include <vector>

#include "Types.hpp"
#include "NPY.hpp"

class RecordsNPY ; 
class Index ; 

class PhotonsNPY {
   public:  
       enum {
              e_seqhis , 
              e_seqmat 
            };

   public:  
       PhotonsNPY(NPY<float>* photons); 
   public:  
       void                  setTypes(Types* types);
       void                  setRecords(NPY<short>* records);
       void                  setRecs(RecordsNPY* recs);
   public:  
       NPY<float>*           getPhotons();
       NPY<short>*           getRecords();
       RecordsNPY*           getRecs();
       Types*                getTypes();

       NPY<unsigned char>*   getSeqIdx();
       Index*                getSeqHis(); 
       Index*                getSeqMat(); 

       void prepSequenceIndex();

       Index* makeSequenceCountsIndex( Types::Item_t etype, 
            std::map<std::string, unsigned int>& su,
            std::map<std::string, std::vector<unsigned int> >&  sv,
            unsigned int cutoff
       );

       void fillSequenceIndex(
                unsigned int k,
                Index* idx, 
                std::map<std::string, std::vector<unsigned int> >&  sv );



       void dumpMaskCounts(const char* msg, Types::Item_t etype, std::map<unsigned int, unsigned int>& uu, unsigned int cutoff);

       void dumpSequenceCounts(const char* msg, Types::Item_t etype, 
                                 std::map<std::string, unsigned int>& su,
                                 std::map<std::string, std::vector<unsigned int> >& sv,
                                 unsigned int cutoff
                              );

       void dumpPhotonRecord(unsigned int photon_id, const char* msg="phr");
       void dumpPhoton(unsigned int i, const char* msg="pho");

       void dumpPhotons(const char* msg="PhotonsNPY::dumpPhotons", unsigned int ndump=5);


       //NPYBase*    getItem(Types::Item_t item);
       const char* getItemName(Types::Item_t item);





   public:
       // precise agreement between Photon and Record histories
       // demands setting a bounce max less that maxrec
       // in order to avoid any truncated and top record slot overwrites 
       //
       // eg for maxrec 10 bounce max of 9 (option -b9) 
       //    succeeds to give perfect agreement  
       //                 
       void examinePhotonHistories();


       glm::ivec4 getFlags();

   public:
   public:  
       // ivec4 containing 1st four boundary codes provided by the selection
       glm::ivec4                                  getSelection();

   public:  
       // interface to ImGui checkboxes that make the boundary selection
       bool*        getBoundariesSelection(); 

       typedef std::vector< std::pair<int, std::string> >  Choices_t ; 
       typedef std::vector< std::pair<unsigned int, std::string> >  UChoices_t ; 

       Choices_t*   getBoundariesPointer(); 
       Choices_t&   getBoundaries(); 

       // signed mode : signs the boundary code according to the sign of (2,0) vpol.x (currently cos_theta)
       void setBoundaryNames(std::map<int, std::string> names);    
       void indexBoundaries(bool sign=true);
   private:
       void dumpBoundaries(const char* msg);

   public:  
       // decoding records
       void dump(const char* msg);

   private:
       NPY<float>*                  m_photons ; 
       NPY<short>*                  m_records ; 
       RecordsNPY*                  m_recs ; 
       Types*                       m_types ; 
       NPY<unsigned char>*          m_seqidx  ; 
       unsigned int                 m_maxrec ; 
       Index*                       m_seqhis ; 
       Index*                       m_seqmat ; 

   protected:
       std::map<int, std::string>   m_names ; 

       Choices_t                    m_boundaries ; 
       bool*                        m_boundaries_selection ; 


};



inline PhotonsNPY::PhotonsNPY(NPY<float>* photons) 
       :  
       m_photons(photons),
       m_records(NULL),
       m_recs(NULL),
       m_types(NULL),
       m_seqidx(NULL),
       m_maxrec(0),
       m_seqhis(NULL),
       m_seqmat(NULL),
       m_boundaries_selection(NULL)
{
}




inline void PhotonsNPY::setTypes(Types* types)
{  
    m_types = types ; 
}
inline void PhotonsNPY::setRecords(NPY<short>* records)
{
    m_records = records ; 
}

inline NPY<float>* PhotonsNPY::getPhotons()
{
    return m_photons ; 
}
inline NPY<short>* PhotonsNPY::getRecords()
{
    return m_records ; 
}
inline RecordsNPY* PhotonsNPY::getRecs()
{
    return m_recs ; 
}
inline Types* PhotonsNPY::getTypes()
{
    return m_types ; 
}




inline NPY<unsigned char>* PhotonsNPY::getSeqIdx()
{
    if(!m_seqidx)
    { 
        unsigned int ni = m_photons->m_len0 ;
        m_seqidx = NPY<unsigned char>::make_vec4(ni,1,0) ;
    }
    return m_seqidx ; 
}


inline Index* PhotonsNPY::getSeqHis()
{
    return m_seqhis ; 
}
inline Index* PhotonsNPY::getSeqMat()
{
    return m_seqmat ; 
}







/*
inline NPYBase* PhotonsNPY::getItem(Types::Item_t item)
{
    NPYBase* npy = NULL ; 
    switch(item)
    {
        case PHOTONS: npy = m_photons ; break ; 
        case RECORDS: npy = m_records  ; break ; 
        case MATERIAL:                 ; break ; 
        case HISTORY:                  ; break ; 
        case MATERIALSEQ:              ; break ; 
        case HISTORYSEQ:               ; break ; 

    } 
    return npy ;
}
*/




inline void PhotonsNPY::setBoundaryNames(std::map<int, std::string> names)
{
    m_names = names ; 
}


inline bool* PhotonsNPY::getBoundariesSelection()
{
    return m_boundaries_selection ; 
}


inline PhotonsNPY::Choices_t& PhotonsNPY::getBoundaries()
{
    return m_boundaries ; 
}

inline PhotonsNPY::Choices_t* PhotonsNPY::getBoundariesPointer()
{
    return &m_boundaries ; 
}


