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

        static const char* SRC_PLANES ; 
        static const char* SRC_FACES ; 
        static const char* SRC_VERTS ;
        static const char* SRC_TRANSFORMS ; 
        static const char* SRC_IDX ; 
        static const char* SRC_NODES ; 

        static const char* PLANES ; 
        static const char* NODES ; 
        static const char* IDX ; 

        static const char* TREE_META ; 
        static const char* NODE_META ; 

        enum { NJ = 4, NK = 4, MAX_HEIGHT = 10 };

        static unsigned CompleteTreeHeight( unsigned num_nodes );
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
        NPY<float>*    getPlaneBuffer() const ;
        NPY<float>*    getTransformBuffer() const ;
        NPY<float>*    getGTransformBuffer() const ;
        NPY<unsigned>* getIdxBuffer() const ;

        NPY<float>*    getSrcNodeBuffer() const ;
        NPY<float>*    getSrcPlaneBuffer() const ;
        NPY<float>*    getSrcTransformBuffer() const ;
        NPY<float>*    getSrcVertsBuffer() const ;
        NPY<int>*      getSrcFacesBuffer() const ;
        NPY<unsigned>* getSrcIdxBuffer() const ;

        NParameters*   getMetaParameters(int idx) const ;
        NParameters*   getNodeMetadata(unsigned idx) const ;

    public:
        void getSrcPlanes(std::vector<glm::vec4>& _planes, unsigned idx, unsigned num_plane  ) const ;

    public:
        // from m_srcnodes 
        unsigned getTypeCode(unsigned idx);
        unsigned getTransformIndex(unsigned idx);
        bool     isComplement(unsigned idx);
        nquad    getQuad(unsigned idx, unsigned j);

    public:
        // from m_transforms
        nmat4pair*   import_transform_pair(unsigned itra);
        nmat4triple* import_transform_triple(unsigned itra);

        void prepareForImport();  // m_srctransforms -> m_transforms + prepares m_gtransforms to collect globals during import  
        void prepareForExport();  // create m_nodes ready for exporting from the node tree 
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
        void loadsrc(const char* treedir);
        void save(const char* treedir) const ;
    private:
        void saveSrcNodes(const char* treedir) const ;
        void saveSrcTransforms(const char* treedir) const ;
        void saveSrcPlanes(const char* treedir) const ;
        void saveSrcIdx(const char* treedir) const ;
        void saveSrcVerts(const char* treedir) const ;
        void saveSrcFaces(const char* treedir) const ;
        void saveMetadata(const char* treedir, int idx) const  ;
        void saveNodeMetadata(const char* treedir) const  ;
    private:
        void loadSrcNodes(const char* treedir);
        void loadSrcTransforms(const char* treedir);
        void loadSrcPlanes(const char* treedir);
        void loadSrcIdx(const char* treedir);
        void loadSrcVerts(const char* treedir);
        void loadSrcFaces(const char* treedir);
        void loadMetadata(const char* treedir);
        void loadNodeMetadata(const char* treedir);
    private:
        int            m_verbosity ;  

        NPY<float>*    m_srcnodes ; 
        NPY<float>*    m_srctransforms ; 
        NPY<float>*    m_srcplanes ;
        NPY<float>*    m_srcverts ;
        NPY<int>*      m_srcfaces ;
        NPY<unsigned>* m_srcidx ;

        unsigned    m_height ; 
        unsigned    m_num_nodes ; 
        unsigned    m_num_transforms ; 
        unsigned    m_num_planes ;
        unsigned    m_num_srcverts ;
        unsigned    m_num_srcfaces ;

        NParameters*   m_meta ; 
        std::map<unsigned, NParameters*> m_nodemeta ; 

        NPY<float>*    m_nodes ; 
        NPY<float>*    m_transforms ; 
        NPY<float>*    m_gtransforms ; 
        NPY<float>*    m_planes ; 
        NPY<unsigned>* m_idx ;
};


