#pragma once
#include "NPY_API_EXPORT.hh"

class NCSG ; 
struct nd ; 

#include "NGLTF.hpp"

class NPY_API NScene : public NGLTF 
{
    public:
        NScene(const char* base, const char* name, const char* config, int scene_idx=0  );
        nd*   getRoot();
        nd*   getNd(unsigned node_idx);
        NCSG* getCSG(unsigned mesh_idx);
        void dumpNdTree(const char* msg="NScene::dumpNdTree");
    private:
        void load_mesh_extras();
    private:
        nd*  import();
        nd*  import_r(int idx, nd* parent, int depth);
        void dumpNdTree_r(nd* n);
        void compare_trees();
        void compare_trees_r(int idx);
    private:
        unsigned deviseRepeatIndex(nd* n);
        void labelTree_r(nd* n);
    public:
        void     dumpRepeatCount();
        unsigned getRepeatCount(unsigned ridx);
        unsigned getNumRepeats();
    private:
        nd*                       m_root ; 
        std::map<unsigned, nd*>   m_nd ; 
        std::map<unsigned, NCSG*> m_csg ; 
        std::map<unsigned, unsigned>  m_mesh2ridx ;
        std::map<unsigned, unsigned>  m_repeat_count ;


};

