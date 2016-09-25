#pragma once

/**
GSur
=====

Instances of GSur are created post-cache, as a 
way of analysing surfaces and to reconstruct them.

**/


#include <set>
#include <string>
#include <vector>

template <typename T> class GPropertyMap ; 

#include "GGEO_API_EXPORT.hh"

class GGEO_API GSur 
{
         friend class GSurLib ; 
    public:
         GSur(GPropertyMap<float>* pm, char type);
         const char* getName();
         std::string brief();
         std::string check();
         void dump(const char* msg="GSur::dump");

         char getType();
         const char* getPV1();
         const char* getPV2();
         const char* getLV();

         unsigned getNumPVPair();
         unsigned getNumLV();
    private:
          
         void addInner(unsigned vol, unsigned bnd);
         void addOuter(unsigned vol, unsigned bnd);
         void addPVPair(const char* pv1, const char* pv2);
         void addLV(const char* lv);

         void assignVolumes(); 
         void setPVP(const char* pv1, const char* pv2);
         void setLV(const char* lv);

         const std::set<unsigned>& getIBnd();
         const std::set<unsigned>& getOBnd();
    private:
         GPropertyMap<float>*  m_pmap  ;
         char                  m_type ; 
         const char*  m_pv1 ; 
         const char*  m_pv2 ; 
         const char*  m_lv ; 
    private:

         std::vector<unsigned> m_ivol ; 
         std::vector<unsigned> m_ovol ; 

         std::set<unsigned> m_ibnd ; 
         std::set<unsigned> m_obnd ; 

         std::set<std::pair<std::string,std::string> > m_pvp ; 
         std::set<std::string>                         m_slv ; 

};




