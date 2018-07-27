#pragma once

#include <map>
#include <vector>
#include <glm/fwd.hpp>

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"
class NParameters ; 
struct nmat4pair ; 
struct nmat4triple ; 
union nquad ; 


class NPY_API NCSGData
{ 
    public:
        static const unsigned NTRAN ; 
    private:
        static const char* FILENAME ; 
        static const char* PLANES ; 
        static const char* SRC_FACES ; 
        static const char* SRC_VERTS ;
        static const char* IDX ; 
        static const char* TREE_META ; 
        static const char* NODE_META ; 

        enum { NJ = 4, NK = 4, MAX_HEIGHT = 10 };

        static unsigned NumNodes(unsigned height);

        static std::string TxtPath(const char* treedir);
        static bool        ExistsDir(const char* treedir);  // just checks existance of dir
        static bool        ExistsTxt(const char* treedir);  // looks for FILENAME (csg.txt) in the treedir
        static std::string MetaPath(const char* treedir, int idx=-1);
        static bool        ExistsMeta(const char* treedir, int idx=-1);

    public:
        static NParameters* LoadMetadata(const char* treedir, int idx=-1);
        static bool         Exists(const char* treedir);     // compat : pass thru to ExistsDir
    public:
        NCSGData(); 
        void init_buffers(unsigned height);  // maxdepth of node tree
        void setIdx( unsigned index, unsigned soIdx, unsigned lvIdx, unsigned height );
    public:
        unsigned getHeight() const ;
        unsigned getNumNodes() const ;

        NPY<float>*    getNodeBuffer() const ;
        NPY<float>*    getTransformBuffer() const ;
        NPY<float>*    getGTransformBuffer() const ;
        NPY<float>*    getPlaneBuffer() const ;
        NPY<float>*    getSrcVertsBuffer() const ;
        NPY<int>*      getSrcFacesBuffer() const ;
        NPY<unsigned>* getIdxBuffer() const ;
        NParameters*   getMetaParameters(int idx) const ;
        NParameters*   getNodeMetadata(unsigned idx) const ;

    public:
        void getPlanes(std::vector<glm::vec4>& _planes, unsigned idx, unsigned num_plane  ) const ;

    public:
        // from m_nodes 
        unsigned getTypeCode(unsigned idx);
        unsigned getTransformIndex(unsigned idx);
        bool     isComplement(unsigned idx);
        nquad    getQuad(unsigned idx, unsigned j);

    public:
        // from m_transforms
        nmat4pair*   import_transform_pair(unsigned itra);
        nmat4triple* import_transform_triple(unsigned itra);

    public:
        // add to m_gtransforms
        unsigned addUniqueTransform( const nmat4triple* gtransform );
        void dump_gtransforms() const ;
    public:
        std::string smry() const ;
        std::string desc() const ;
        template<typename T> void setMeta(const char* key, T value);
        template<typename T> T getMeta(const char* key, const char* fallback ) const ;

    public:
        void load(const char* treedir);
        void save(const char* treedir) const ;
    private:
        void saveMetadata(const char* treedir, int idx) const  ;
        void saveNodes(const char* treedir) const ;
        void saveNodeMetadata(const char* treedir) const  ;
        void saveTransforms(const char* treedir) const ;
        void savePlanes(const char* treedir) const ;
        void saveIdx(const char* treedir) const ;
        void saveSrcVerts(const char* treedir) const ;
        void saveSrcFaces(const char* treedir) const ;
    private:
        void loadMetadata(const char* treedir);
        void loadNodes(const char* treedir);
        void loadNodeMetadata(const char* treedir);
        void loadTransforms(const char* treedir);
        void loadPlanes(const char* treedir);
        void loadIdx(const char* treedir);
        void loadSrcVerts(const char* treedir);
        void loadSrcFaces(const char* treedir);
    private:
        int            m_verbosity ;  
        NParameters*   m_meta ; 
        NPY<float>*    m_nodes ; 
        NPY<float>*    m_transforms ; 
        NPY<float>*    m_gtransforms ; 
        NPY<float>*    m_planes ;
        NPY<float>*    m_srcverts ;
        NPY<int>*      m_srcfaces ;
        NPY<unsigned>* m_idx ;

        unsigned    m_height ; 
        unsigned    m_num_nodes ; 
        unsigned    m_num_transforms ; 
        unsigned    m_num_planes ;

        unsigned    m_num_srcverts ;
        unsigned    m_num_srcfaces ;

        std::map<unsigned, NParameters*> m_nodemeta ; 

};


