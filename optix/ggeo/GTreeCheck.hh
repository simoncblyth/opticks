#pragma once

#include <string>
#include <map>
#include <vector>

class GGeo ; 
class GNode ; 
class GSolid ; 
template<class T> class Counts ;

class GTreeCheck {
   public:
        typedef std::map<std::string, std::vector<GNode*> > MSVN ; 
   public:
        GTreeCheck(GGeo* ggeo);
   public:
        void init();
        void traverse();
   public:
        bool operator()(const std::string& dig) ;
   private:
        void traverse( GNode* node, unsigned int depth );
        void findRepeatCandidates(unsigned int minrep);
        void dumpRepeatCandidates();
        void dumpRepeatCandidate(unsigned int index);
        bool isContainedRepeat( const std::string& pdig, unsigned int levels ) const ;
   private:
       GGeo*                     m_ggeo ; 
       GSolid*                   m_root ; 
       unsigned int              m_count ;  
       Counts<unsigned int>*     m_digest_count ; 
       std::vector<std::string>  m_repeat_candidates ; 
   private:
       MSVN                      m_repeat ; 
 
};


inline GTreeCheck::GTreeCheck(GGeo* ggeo) 
       :
       m_ggeo(ggeo),
       m_root(NULL),
       m_count(0)
       {
          init();
       }




