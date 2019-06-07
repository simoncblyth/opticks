#pragma once

#include <map>
#include <vector>
#include <string>
#include <glm/fwd.hpp>

class NPYBase ; 
class NPYSpecList ; 
class NPYList ; 

template <typename T> class NPY ; 
#include "NPY_API_EXPORT.hh"
struct nmat4triple ; 
union nquad ; 

/**
NCSGData
=========

Are using this class as a testbed for generic buffer handling...
which creates a kinda strange style mix : consider making this 
entirely generic buffer handling code. Moving specifics up into NCSG ? 

Buffers are grouped into two categeories

SrcBuffers
    which tellingly have no GTransform buffer, 
    but do have faces and verts

TransportBuffers
    (is that the right name ?) 




**/

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
           PLANES,
           IDX,
           TRANSFORMS, 
           GTRANSFORMS

        }  NCSGData_t ; 

        static const NPYSpecList* MakeSPECS(); 
        static const NPYSpecList* SPECS ; 

    public:
        unsigned getHeight() const ;
        unsigned getNumNodes() const ;
        NPYList* getNPYList() const ; 

        NPY<float>*    getNodeBuffer() const ;
        NPY<unsigned>* getIdxBuffer() const ;
        NPY<float>*    getTransformBuffer() const ;
        NPY<float>*    getGTransformBuffer() const ;
        NPY<float>*    getPlaneBuffer() const ;

        NPY<float>*    getSrcNodeBuffer() const ;
        NPY<unsigned>* getSrcIdxBuffer() const ;
        NPY<float>*    getSrcTransformBuffer() const ;
        NPY<float>*    getSrcPlaneBuffer() const ;
        NPY<int>*      getSrcFacesBuffer() const ;
        NPY<float>*    getSrcVertsBuffer() const ;

    public:
        void loadsrc(const char* treedir);
        void savesrc(const char* treedir) const ;
    public:
        static bool        Exists(const char* treedir);     // compat : pass thru to ExistsDir
        enum { NTRAN = 3  };
    private:
        static const char* FILENAME ; 
        enum { MAX_HEIGHT = 10 };
        static unsigned CompleteTreeHeight( unsigned num_nodes );
        static unsigned NumNodes(unsigned height);

        static std::string TxtPath(const char* treedir);
        static bool        ExistsTxt(const char* treedir);  // looks for FILENAME (csg.txt) in the treedir

        static bool        ExistsDir(const char* treedir);  // just checks existance of dir
    public:
        NCSGData(); 
        void init_buffers(unsigned height);  // maxdepth of node tree
        void setIdx( unsigned index, unsigned soIdx, unsigned lvIdx, unsigned height );

    public:
        // pure const access to src buffer content 
        void     getSrcPlanes(std::vector<glm::vec4>& _planes, unsigned idx, unsigned num_plane  ) const ;
        unsigned getTypeCode(unsigned idx) const ;
        unsigned getTransformIndex(unsigned idx) const ; 
        bool     isComplement(unsigned idx) const ;
        nquad    getQuad(unsigned idx, unsigned j) const ;

    private:
        void import_src_identity();
    public:  
        unsigned getSrcIndex() const ; 
        unsigned getSrcSOIdx() const ; 
        unsigned getSrcLVIdx() const ; 
        unsigned getSrcHeight() const ; 
    public:
        // from m_transforms
        nmat4triple* import_transform_triple(unsigned itra);
        void prepareForImport();  // m_srctransforms -> m_transforms + prepares m_gtransforms to collect globals during import  
        void prepareForSetup();   
        void prepareForExport();  // create m_nodes ready for exporting from the node tree 
    public:
        unsigned addUniqueTransform( const nmat4triple* gtransform ); // add to m_gtransforms
        void dump_gtransforms() const ;
    public:
        std::string smry() const ;
        std::string desc() const ;
    private:
        int         m_verbosity ;  
        NPYList*    m_npy ; 
        unsigned    m_height ; 
        unsigned    m_num_nodes ; 

    private:
        // from import_src_identity
        unsigned m_src_index ; 
        unsigned m_src_soIdx ; 
        unsigned m_src_lvIdx ; 
        unsigned m_src_height ; 

};


