#pragma once

#include <string>
#include <map>
#include <vector>

class GGeo ; 
class GGeoLib ; 
class GNode ; 
class GSolid ; 
class GBuffer ;
template<class T> class Counts ;
template<class T> class NPY ;



class GTreeCheck {
   public:
        typedef std::map<std::string, std::vector<GNode*> > MSVN ; 
   public:
        GTreeCheck(GGeo* ggeo);
        void setRepeatMin(unsigned int repeat_min);
        void setVertexMin(unsigned int vertex_min);
        void createInstancedMergedMeshes(bool deltacheck=false); // still canonically invoked by GLoader
   public:
        void init();
        void traverse();
        void deltacheck();
        unsigned int getRepeatIndex(const std::string& pdig );
        void labelTree();
        unsigned int getNumRepeats(); 
        GNode* getRepeatExample(unsigned int ridx);

        // canonically invoked by CreateInstancedMergedMeshes
        GBuffer* makeInstanceTransformsBuffer(unsigned int ridx);
        GBuffer* makeInstanceIdentityBuffer(unsigned int ridx) ;
        NPY<unsigned int>* makeAnalyticInstanceIdentityBuffer(unsigned int ridx);
   public:
        bool operator()(const std::string& dig) ;
   private:
        void traverse( GNode* node, unsigned int depth );
        void deltacheck( GNode* node, unsigned int depth );
        void findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min);
        void dumpRepeatCandidates();
        void dumpRepeatCandidate(unsigned int index, bool verbose=false);
        bool isContainedRepeat( const std::string& pdig, unsigned int levels ) const ;
        void labelTree( GNode* node, unsigned int ridx );
        std::vector<GNode*> getPlacements(unsigned int ridx);

   private:
       GGeo*                     m_ggeo ; 
       GGeoLib*                  m_geolib ; 
       unsigned int              m_repeat_min ; 
       unsigned int              m_vertex_min ; 
       GSolid*                   m_root ; 
       unsigned int              m_count ;  
       unsigned int              m_labels ;  
       Counts<unsigned int>*     m_digest_count ; 
       std::vector<std::string>  m_repeat_candidates ; 
 
};


// Following enabling of vertex de-duping (done at AssimpWrap level) 
// the below criteria are finding fewer repeats, for DYB only the hemi pmt
// TODO: retune

inline GTreeCheck::GTreeCheck(GGeo* ggeo) 
       :
       m_ggeo(ggeo),
       m_geolib(NULL),
       m_repeat_min(120),
       m_vertex_min(300),   // aiming to include leaf? sStrut and sFasteners
       m_root(NULL),
       m_count(0),
       m_labels(0)
       {
          init();
       }


inline unsigned int GTreeCheck::getNumRepeats()
{
    return m_repeat_candidates.size();
}

inline void GTreeCheck::setRepeatMin(unsigned int repeat_min)
{
   m_repeat_min = repeat_min ; 
}
inline void GTreeCheck::setVertexMin(unsigned int vertex_min)
{
   m_vertex_min = vertex_min ; 
}


