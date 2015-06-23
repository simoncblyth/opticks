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


       PhotonsNPY(NPY<float>* photons, NPY<short>* record=NULL, unsigned int maxrec=10); // weak references only

       void setRecords(NPY<short>* records);
       NPY<float>* getPhotons();
       NPY<short>* getRecords();
       void dumpRecord(unsigned int i, const char* msg="rec");
       void dumpRecords(const char* msg="PhotonsNPY::dumpRecords", unsigned int ndump=5);
       NPYBase*    getItem(Item_t item);
       const char* getItemName(Item_t item);


       // boundary names corresponding to absolute integer codes 
       // TODO: offset codes by one to avoid confusion regarding sign of Vacuum/Vacuum 0 

       void setBoundaryNames(std::map<int, std::string> names);    

       // signed mode : signs the boundary code according to the sign of (2,0) vpol.x (currently cos_theta)
       void classify(bool sign=false);

   public:
       // precise agreement between Photon and Record histories
       // demands setting a bounce max less that maxrec
       // in order to avoid any truncated and top record slot overwrites 
       //
       // eg for maxrec 10 bounce max of 9 (option -b9) 
       //    succeeds to give perfect agreement  
       //                 
       void examinePhotonHistories();
       void examineRecordHistories();
       std::string getHistoryString(unsigned int flags);
       std::string getStepFlagString(unsigned char flag);
       glm::ivec4 getFlags();

   public:
       void readFlags(const char* path); // parse enum flags from photon.h
       void readMaterials(const char* idpath, const char* name="GMaterialIndexLocal.json");    

       void dumpFlags(const char* msg="PhotonsNPY::dumpFlags");
       void dumpMaterials(const char* msg="PhotonsNPY::dumpMaterials");

       std::string findMaterialName(unsigned int index);

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
       // decoding records
       void dump(const char* msg);

       void setCenterExtent(glm::vec4& ce);
       void setTimeDomain(glm::vec4& td);
       void setWavelengthDomain(glm::vec4& wd);

       float unshortnorm(short value, float center, float extent );
       float unshortnorm_position(short uv, unsigned int k );
       float unshortnorm_time(    short uv, unsigned int k );

       float uncharnorm_polarization(unsigned char value);
       float uncharnorm_wavelength(unsigned char value);
       float uncharnorm(unsigned char value, float center, float extent, float bitmax );


       void unpack_position_time(glm::vec4& post, unsigned int i, unsigned int j);
       void unpack_polarization_wavelength(glm::vec4& polw, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1);
       void unpack_material_flags(glm::uvec4& flag, unsigned int i, unsigned int j, unsigned int k0, unsigned int k1);


   private:
       NPY<float>*                  m_photons ; 
       NPY<short>*                  m_records ; 
       unsigned int                 m_maxrec ; 

   protected:
       std::map<int, std::string>   m_names ; 

       Choices_t                    m_boundaries ; 
       bool*                        m_boundaries_selection ; 

       UChoices_t                   m_flags ; 
       bool*                        m_flags_selection ; 

       std::map<std::string, unsigned int>  m_materials ;

       glm::vec4                   m_center_extent ; 
       glm::vec4                   m_time_domain ; 
       glm::vec4                   m_wavelength_domain ; 
 
};



inline PhotonsNPY::PhotonsNPY(NPY<float>* photons, NPY<short>* records, unsigned int maxrec) 
       :  
       m_photons(photons),
       m_records(records),
       m_maxrec(maxrec),
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


inline void PhotonsNPY::setCenterExtent(glm::vec4& ce)
{
    m_center_extent = ce ; 
}
inline void PhotonsNPY::setTimeDomain(glm::vec4& td)
{
    m_time_domain = td ; 
}
inline void PhotonsNPY::setWavelengthDomain(glm::vec4& wd)
{
    m_wavelength_domain = wd ; 
}


