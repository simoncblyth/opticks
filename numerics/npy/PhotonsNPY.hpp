#pragma once

#include "glm/fwd.hpp"

#include <map>
#include <string>
#include <vector>

#include "NPY.hpp"

class PhotonsNPY {
   public:  
       static const char* PHOTONS_ ; 
       static const char* RECORDS_ ; 
       typedef enum { PHOTONS, RECORDS } Item_t ;

       typedef std::vector< std::pair<int, std::string> >  Choices_t ; 
       typedef std::vector< std::pair<unsigned int, std::string> >  UChoices_t ; 


       PhotonsNPY(NPY<float>* photons, NPY<short>* record=NULL); // weak references only

       void setRecords(NPY<short>* records);
       NPY<float>* getPhotons();
       NPY<short>* getRecords();
       void dumpRecords(const char* msg="PhotonsNPY::dumpRecords", unsigned int ndump=5, unsigned int maxrec=10);
       NPYBase*    getItem(Item_t item);
       const char* getItemName(Item_t item);


       // boundary names corresponding to absolute integer codes 
       // TODO: offset codes by one to avoid confusion regarding sign of Vacuum/Vacuum 0 

       void setBoundaryNames(std::map<int, std::string> names);    

       // signed mode : signs the boundary code according to the sign of (2,0) vpol.x (currently cos_theta)
       void classify(bool sign=false);

   public:
       void examineHistories(Item_t item);
       std::string getHistoryString(unsigned int flags);
       void readFlags(const char* path); // parse enum flags from photon.h
       void dumpFlags(const char* msg="PhotonsNPY::dumpFlags");
       glm::ivec4 getFlags();

   public:  
       // ivec4 containing 1st four boundary codes provided by the selection
       glm::ivec4                                  getSelection();

   public:  
       // interface to ImGui checkboxes that make the boundary selection
       bool*        getBoundariesSelection(); 
       Choices_t*   getBoundariesPointer(); 
       Choices_t&   getBoundaries(); 

   private:
       bool* initBooleanSelection(unsigned int n);
       Choices_t findBoundaries(bool sign);
       void dumpBoundaries(const char* msg);

   public:  
       void dump(const char* msg);

   private:
       NPY<float>*                  m_photons ; 
       NPY<short>*                  m_records ; 

   protected:
       std::map<int, std::string>   m_names ; 

       Choices_t                    m_boundaries ; 
       bool*                        m_boundaries_selection ; 

       UChoices_t                   m_flags ; 
       bool*                        m_flags_selection ; 
 
};



inline PhotonsNPY::PhotonsNPY(NPY<float>* photons, NPY<short>* records) 
       :  
       m_photons(photons),
       m_records(records),
       m_boundaries_selection(NULL)
{
}

inline NPY<float>* PhotonsNPY::getPhotons()
{
    return m_photons ; 
}
inline NPY<short>* PhotonsNPY::getRecords()
{
    return m_records ; 
}
inline NPYBase* PhotonsNPY::getItem(Item_t item)
{
    NPYBase* npy = NULL ; 
    switch(item)
    {
        case PHOTONS: npy = m_photons ; break ; 
        case RECORDS: npy = m_records  ; break ; 
    } 
    return npy ;
}
inline void PhotonsNPY::setRecords(NPY<short>* records)
{
    m_records = records ; 
}






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



