#pragma once

#include <vector>
#include "NGLM.hpp"
#include "Types.hpp"

template <typename T> class NPY ; 
class Index ; 
class Typ ;

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"


struct NPY_API NRec
{
    glm::vec4 post ; 
    glm::vec4 polw ; 
    glm::uvec4 flag ; 
    glm::ivec4 iflag ; 

    const char* m1 ; 
    const char* m2 ; 
    const char* hs ; 

};


class NPY_API RecordsNPY {
    public:  
        RecordsNPY(NPY<short>* records, unsigned maxrec, unsigned verbosity=0 ); 
    public:  
        NPY<short>*           getRecords();
        void                  setTypes(Types* types);
        void                  setTyp(Typ* typ);
        unsigned int          getMaxRec();
        bool                  isFlat();

        void                  dumpTyp(const char* msg="RecordsNPY::dumpTyp") const ;

        std::string m1String( const glm::uvec4& flag );
        std::string m2String( const glm::uvec4& flag );
        std::string historyString( const glm::uvec4& flag );

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

       void unpack( NRec& rec, unsigned i, unsigned j );
       void unpack( glm::vec4& post, glm::vec4& polw, glm::uvec4& uflag, glm::ivec4& iflag, unsigned i, unsigned j );

       void dumpRecord(unsigned int i, unsigned int j, const char* msg="rec");
       void dumpRecords(const char* msg="RecordsNPY::dumpRecords", unsigned int ndump=5);

   public:
       // geometric properties of photon path
       glm::vec4 getCenterExtent(unsigned int photon_id);
       glm::vec4 getLengthDistanceDuration(unsigned int photon_id);
       glm::vec4 getLengthDistanceDurationRecs(std::vector<NRec>& recs, unsigned int photon_id);
   private:
       void tracePath(unsigned int photon_id, std::vector<NRec>& recs, float& length, float& distance, float& duration );
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
       unsigned         m_maxrec ; 
       unsigned         m_verbosity ; 

       bool             m_flat ;
       Types*           m_types ; 
       Typ*             m_typ ; 

   private:
       glm::vec4        m_center_extent ; 
       glm::vec4        m_time_domain ; 
       glm::vec4        m_wavelength_domain ; 

}; 

#include "NPY_TAIL.hh"


