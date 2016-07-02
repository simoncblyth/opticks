#pragma once

#include "NGLM.hpp"
#include "Types.hpp"

template <typename T> class NPY ; 
class Index ; 
class Typ ;

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API RecordsNPY {
   public:  
       RecordsNPY(NPY<short>* records, unsigned int maxrec, bool flat); 
   public:  
       NPY<short>*           getRecords();
       void                  setTypes(Types* types);
       void                  setTyp(Typ* typ);
       unsigned int          getMaxRec();
       bool                  isFlat();
   public:  
       NPY<unsigned long long>* makeSequenceArray(Types::Item_t etype);

       void constructFromRecord(unsigned int photon_id, unsigned int& bounce, unsigned int& history, unsigned int& material);
       void appendMaterials(std::vector<unsigned int>& materials, unsigned int photon_id);
   public:
       // unpacking records related 
       void setDomains(NPY<float>* domains);
       float unshortnorm(short value, float center, float extent );
       float unshortnorm_position(short uv, unsigned int k );
       float unshortnorm_time(    short uv, unsigned int k );

       float uncharnorm_polarization(unsigned char value);
       float uncharnorm_wavelength(unsigned char value);
       float uncharnorm(unsigned char value, float center, float extent, float bitmax );

       void unpack_position_time(glm::vec4& post, unsigned int i, unsigned int j, unsigned int k);
       void unpack_polarization_wavelength(glm::vec4& polw, unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1);
       void unpack_material_flags(glm::uvec4& flag, unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1);
       void unpack_material_flags_i(glm::ivec4& flag, unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1);

       void dumpRecord(unsigned int i, unsigned int j, const char* msg="rec");
       void dumpRecords(const char* msg="RecordsNPY::dumpRecords", unsigned int ndump=5);

   public:
       // geometric properties of photon path
       glm::vec4 getCenterExtent(unsigned int photon_id);
       glm::vec4 getLengthDistanceDuration(unsigned int photon_id);
   private:
       void tracePath(unsigned int photon_id, float& length, float& distance, float& duration );
   private:
       void setCenterExtent(glm::vec4& ce);
       void setTimeDomain(glm::vec4& td);
       void setWavelengthDomain(glm::vec4& wd);

   public:
       std::string getSequenceString(unsigned int photon_id, Types::Item_t etype);
       unsigned long long getSequence(unsigned int photon_id, Types::Item_t etype);
   public:
       // higer level access
       void unpack_material_flags(glm::uvec4& flag, unsigned int photon_id , unsigned int r );
       bool exists(unsigned int photon_id , unsigned int r );

   private:
       NPY<short>*      m_records ; 
       unsigned int     m_maxrec ; 
       bool             m_flat ;
       Types*           m_types ; 
       Typ*             m_typ ; 

   private:
       glm::vec4        m_center_extent ; 
       glm::vec4        m_time_domain ; 
       glm::vec4        m_wavelength_domain ; 

}; 

#include "NPY_TAIL.hh"


