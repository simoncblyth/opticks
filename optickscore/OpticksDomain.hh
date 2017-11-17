#pragma once

template <typename T> class NPY ; 
#include <glm/fwd.hpp>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksDomain {
    public:
       OpticksDomain();
       void updateBuffer();
       void importBuffer();
       void dump(const char* msg="OpticksDomains::dump");
    public:
       unsigned getMaxRng() const ;
       unsigned getMaxRec() const ;
       unsigned getMaxBounce() const ;
       void setMaxRng(unsigned maxrng);
       void setMaxRec(unsigned maxrec);
       void setMaxBounce(unsigned maxbounce);
    public:
       NPY<float>* getFDomain();
       NPY<int>*   getIDomain();
       void setFDomain(NPY<float>* fdom);
       void setIDomain(NPY<int>* idom);
   public:
       // domains used for record compression
       void setSpaceDomain(const glm::vec4& space_domain);
       void setTimeDomain(const glm::vec4& time_domain);
       void setWavelengthDomain(const glm::vec4& wavelength_domain);

       const glm::vec4& getSpaceDomain();
       const glm::vec4& getTimeDomain();
       const glm::vec4& getWavelengthDomain();

     private:
        void init();

     private:
        NPY<float>*     m_fdom ; 
        NPY<int>*       m_idom ; 

        glm::vec4       m_space_domain ; 
        glm::vec4       m_time_domain ; 
        glm::vec4       m_wavelength_domain ; 
        glm::ivec4      m_settings ; 

};

#include "OKCORE_TAIL.hh"


