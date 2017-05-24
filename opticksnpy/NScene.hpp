#pragma once
#include "NPY_API_EXPORT.hh"

class NCSG ; 
class NParameters ; 
struct nd ; 
template<class T> class Counts ;

#include "NGLTF.hpp"

class NPY_API NScene : public NGLTF 
{
    public:
        NScene(const char* base, const char* name, const char* config, int scene_idx=0  );
        nd*   getRoot();
        nd*   getNd(unsigned node_idx);
        NCSG* getCSG(unsigned mesh_idx);
        void dumpNdTree(const char* msg="NScene::dumpNdTree");
        unsigned getVerbosity();

    private:
        template<typename T> T getCSGMeta(unsigned mesh_id, const char* key, const char* fallback ) const ;
        std::string lvname(unsigned mesh_id) const ;
        std::string soname(unsigned mesh_id) const ;
        int         height(unsigned mesh_id) const ;
        int         nchild(unsigned mesh_id) const ;
        bool        isSkip(unsigned mesh_id) const ;
        std::string meshmeta(unsigned mesh_id) const ;
    private:
        void load_asset_extras();
        void load_csg_metadata();
        void load_mesh_extras();
        void find_repeat_candidates();
    public:
        bool operator()(const std::string& pdig);
        bool is_contained_repeat( const std::string& pdig, unsigned levels ) ;
        void dump_repeat_candidate(unsigned i) const ;
        void dump_repeat_candidates() const;
    private:
        nd*  import_r(int idx, nd* parent, int depth);
        void dumpNdTree_r(nd* n);
        void count_progeny_digests();
        void count_progeny_digests_r(nd* n);
        void compare_trees();
        void compare_trees_r(int idx);
        void markGloballyUsedMeshes_r(nd* n);
    private:
        unsigned deviseRepeatIndex(const std::string& pdig);
        unsigned deviseRepeatIndex_0(nd* n);
        void labelTree();

#ifdef OLD_LABEL_TREE
        void labelTree_r(nd* n, unsigned /*ridx*/);
#else
        void labelTree_r(nd* n, unsigned ridx);
#endif

    public:
        void     dumpRepeatCount();
        unsigned getRepeatCount(unsigned ridx);
        unsigned getNumRepeats();
    private:
        nd*                               m_root ; 
        std::map<unsigned, nd*>           m_nd ; 
        std::map<unsigned, NParameters*>  m_csg_metadata ;
        std::map<unsigned, NCSG*>         m_csg ; 
        std::map<unsigned, unsigned>      m_mesh2ridx ;
        std::map<unsigned, unsigned>      m_repeat_count ;

        unsigned                          m_verbosity ; 
        unsigned                          m_num_global ; 
        unsigned                          m_num_csgskip ; 
        unsigned                          m_node_count ; 
        unsigned                          m_label_count ; 
        Counts<unsigned>*                 m_digest_count ;
        std::vector<std::string>          m_repeat_candidates ;
     


};

