#pragma once

#include <map>
#include <vector>
#include <string>
#include <glm/fwd.hpp>

class NPYBase ; 

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"
class NParameters ; 
struct nmat4pair ; 
struct nmat4triple ; 
union nquad ; 


class NPY_API NCSGData
{ 
    public:
        typedef enum 
        { 
           SRC_NODES,
           SRC_IDX, 
           SRC_TRANSFORMS, 
           SRC_PLANES, 
           SRC_FACES, 
           SRC_VERTS,

           NODES,
           TRANSFORMS, 
           GTRANSFORMS, 
           IDX

        }  NCSGData_t ; 

        static NPYBase::Type_t BufferType(NCSGData_t bid);
        static const char*     BufferName(NCSGData_t bid);
        static std::string     BufferPath( const char* treedir, NCSGData_t bid ) ;

        static const char* SRC_NODES_ ; 
        static const char* SRC_IDX_ ; 
        static const char* SRC_TRANSFORMS_ ; 
        static const char* SRC_PLANES_ ; 
        static const char* SRC_FACES_ ; 
        static const char* SRC_VERTS_ ;

        static const char* NODES_ ; 
        static const char* TRANSFORMS_ ; 
        static const char* GTRANSFORMS_ ; 
        static const char* IDX_ ; 


        NPYBase* getBuffer(NCSGData_t bid) const  ; 
        void     setBuffer(NCSGData_t bid, NPYBase* buffer ) ; 

        void saveBuffer(const char* treedir, NCSGData_t bid, bool require=false) const ;
        void loadBuffer(const char* treedir, NCSGData_t bid, bool require=false) ;

    public:

        NPY<float>*    getNodeBuffer() const ;
        NPY<float>*    getTransformBuffer() const ;
        NPY<float>*    getGTransformBuffer() const ;
        NPY<unsigned>* getIdxBuffer() const ;

        NPY<float>*    getSrcNodeBuffer() const ;
        NPY<float>*    getSrcPlaneBuffer() const ;
        NPY<float>*    getSrcTransformBuffer() const ;
        NPY<float>*    getSrcVertsBuffer() const ;
        NPY<int>*      getSrcFacesBuffer() const ;
        NPY<unsigned>* getSrcIdxBuffer() const ;

    public:
        void loadsrc(const char* treedir);
        void savesrc(const char* treedir) const ;

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

    public:
        static const unsigned NTRAN ; 
    private:
        static const char* FILENAME ; 
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
    public:
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

    private:
        int         m_verbosity ;  

        NPYBase*    m_srcnodes ; 
        NPYBase*    m_srctransforms ; 
        NPYBase*    m_srcplanes ;
        NPYBase*    m_srcverts ;
        NPYBase*    m_srcfaces ;
        NPYBase*    m_srcidx ;

        /*
        NPY<float>*    m_srcnodes ; 
        NPY<float>*    m_srctransforms ; 
        NPY<float>*    m_srcplanes ;
        NPY<float>*    m_srcverts ;
        NPY<int>*      m_srcfaces ;
        NPY<unsigned>* m_srcidx ;
        */

        unsigned    m_height ; 
        unsigned    m_num_nodes ; 

        //unsigned    m_num_transforms ; 
        //unsigned    m_num_planes ;
        //unsigned    m_num_srcverts ;
        //unsigned    m_num_srcfaces ;

        NParameters*   m_meta ; 
        std::map<unsigned, NParameters*> m_nodemeta ; 

        /*        
        NPY<float>*    m_nodes ; 
        NPY<float>*    m_transforms ; 
        NPY<float>*    m_gtransforms ; 
        NPY<float>*    m_planes ; 
        NPY<unsigned>* m_idx ;
        */

        NPYBase*  m_nodes ; 
        NPYBase*  m_transforms ; 
        NPYBase*  m_gtransforms ; 
        NPYBase*  m_planes ; 
        NPYBase*  m_idx ;

};


