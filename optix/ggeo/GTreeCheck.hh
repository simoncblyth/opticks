#pragma once

#include <string>
#include <map>
#include <vector>

class GGeo ; 
class GNode ; 
class GSolid ; 
class GBuffer ;
template<class T> class Counts ;

class GTreeCheck {
   public:
        typedef std::map<std::string, std::vector<GNode*> > MSVN ; 
   public:
        GTreeCheck(GGeo* ggeo);
   public:
        void init();
        void traverse();
        unsigned int getRepeatIndex(const std::string& pdig );
        void labelTree();
        unsigned int getNumRepeats(); 
        GNode* getRepeatExample(unsigned int ridx);
        GBuffer* makeInstanceTransformsBuffer(unsigned int ridx);
   public:
        bool operator()(const std::string& dig) ;
   private:
        void traverse( GNode* node, unsigned int depth );
        void findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min);
        void dumpRepeatCandidates();
        void dumpRepeatCandidate(unsigned int index, bool verbose=false);
        bool isContainedRepeat( const std::string& pdig, unsigned int levels ) const ;
        void labelTree( GNode* node, unsigned int ridx );

   private:
       GGeo*                     m_ggeo ; 
       unsigned int              m_repeat_min ; 
       unsigned int              m_vertex_min ; 
       GSolid*                   m_root ; 
       unsigned int              m_count ;  
       bool                      m_delta_check ; 
       unsigned int              m_labels ;  
       Counts<unsigned int>*     m_digest_count ; 
       std::vector<std::string>  m_repeat_candidates ; 
 
};


inline GTreeCheck::GTreeCheck(GGeo* ggeo) 
       :
       m_ggeo(ggeo),
       m_repeat_min(120),
       m_vertex_min(300),   // aiming to include leaf? sStrut and sFasteners
       m_root(NULL),
       m_count(0),
       m_delta_check(false),
       m_labels(0)
       {
          init();
       }


inline unsigned int GTreeCheck::getNumRepeats()
{
    return m_repeat_candidates.size();
}
