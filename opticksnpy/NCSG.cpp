#include <cstring>
#include <algorithm>
#include <sstream>

#include "BStr.hh"
#include "BFile.hh"

#include "OpticksCSG.h"

#include "NGLMStream.hpp"
#include "NParameters.hpp"
#include "NPart.h"
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NNode.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"
#include "NTxt.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )

#include "PLOG.hh"

const char* NCSG::FILENAME = "csg.txt" ; 

NCSG::NCSG(const char* treedir, unsigned index) 
   :
   m_index(index),
   m_root(NULL),
   m_treedir(treedir ? strdup(treedir) : NULL),
   m_nodes(NULL),
   m_transforms(NULL),
   m_meta(NULL),

   m_num_nodes(0),
   m_num_transforms(0),
   m_height(-1),
   m_boundary(NULL)
{
}

void NCSG::load()
{
    //std::string metapath = BFile::ChangeExt(m_path, ".json") ;
    std::string metapath = BFile::FormPath(m_treedir, "meta.json") ;
    std::string nodepath = BFile::FormPath(m_treedir, "nodes.npy") ;
    std::string tranpath = BFile::FormPath(m_treedir, "transforms.npy") ;

    m_meta = new NParameters ; 
    m_meta->load_( metapath.c_str() );

    m_nodes = NPY<float>::load(nodepath.c_str());

    m_num_nodes  = m_nodes->getShape(0) ;  
    unsigned nj = m_nodes->getShape(1);
    unsigned nk = m_nodes->getShape(2);
    assert( nj == NJ );
    assert( nk == NK );

    if(BFile::ExistsFile(tranpath.c_str()))
    {
        m_transforms = NPY<float>::load(tranpath.c_str());
        m_num_transforms  = m_transforms->getShape(0) ;  
        unsigned nj = m_transforms->getShape(1);
        unsigned nk = m_transforms->getShape(2);
        assert( nj == NJ );
        assert( nk == NK );
    }


    m_height = -1 ; 
    int h = MAX_HEIGHT ; 
    while(h--) if(TREE_NODES(h) == m_num_nodes) m_height = h ; 
    assert(m_height >= 0); // must be complete binary tree sized 1, 3, 7, 15, 31, ...
}


NCSG::NCSG(nnode* root, unsigned index) 
   :
   m_index(index),
   m_root(root),
   m_treedir(NULL),
   m_nodes(NULL),
   m_num_nodes(0),
   m_height(root->maxdepth()),
   m_boundary(NULL)
{
   m_num_nodes = NumNodes(m_height);
   m_nodes = NPY<float>::make( m_num_nodes, NJ, NK);
   m_nodes->zero();
}



unsigned NCSG::NumNodes(unsigned height)
{
   return TREE_NODES(height);
}
nnode* NCSG::getRoot()
{
    return m_root ; 
}
unsigned NCSG::getHeight()
{
    return m_height ; 
}
unsigned NCSG::getNumNodes()
{
    return m_num_nodes ; 
}
NPY<float>* NCSG::getNodeBuffer()
{
    return m_nodes ; 
}
NParameters* NCSG::getMeta()
{
    return m_meta ; 
}

const char* NCSG::getBoundary()
{
    return m_boundary ; 
}
const char* NCSG::getTreeDir()
{
    return m_treedir ; 
}
unsigned NCSG::getIndex()
{
    return m_index ; 
}

void NCSG::setBoundary(const char* boundary)
{
    m_boundary = boundary ? strdup(boundary) : NULL ; 
}

unsigned NCSG::getTypeCode(unsigned idx)
{
    return m_nodes->getUInt(idx,TYPECODE_J,TYPECODE_K,0u);
}
unsigned NCSG::getTransformIndex(unsigned idx)
{
    return m_nodes->getUInt(idx,RTRANSFORM_J,RTRANSFORM_K,0u);
}





nvec4 NCSG::getQuad(unsigned idx, unsigned j)
{
    nvec4 qj = m_nodes->getVQuad(idx, j) ;
    return qj ;
}

void NCSG::import()
{
    assert(m_nodes);
    LOG(info) << "NCSG::import"
              << " importing buffer into CSG node tree "
              << " num_nodes " << m_num_nodes
              << " height " << m_height 
              ;

    m_root = import_r(0, NULL, 0) ; 
}


//gmat4pair* NCSG::import_transform(unsigned itra)
glm::mat4* NCSG::import_transform(unsigned itra)
{
    if(itra == 0 || m_transforms == NULL) return NULL ; 
    assert( itra - 1 < m_num_transforms );

    glm::mat4* m =  m_transforms->getMat4Ptr(itra - 1);  // itra is a 1-based index, with 0 meaning None

    // input transforms are rotation first then translation :  T*R*v
    //
    // dis-member tr into r and t by inspection and separately  
    // transpose the rotation and negate the translation
    // (see tests/NGLMTest.cc:test_decompose_invert)

    
    glm::mat4& tr = *m ; 
    glm::mat4 ir = glm::transpose(glm::mat4(glm::mat3(tr)));
    glm::mat4 it = glm::translate(glm::mat4(1.f), -glm::vec3(tr[3])) ; 
    glm::mat4 irit = ir*it ;    // <--- inverse of tr 

    //return new gmat4pair(tr,irit) ; 

    return m ;
}

nnode* NCSG::import_r(unsigned idx, nnode* parent, int itransform )
{
    if(idx >= m_num_nodes) return NULL ; 
        
    OpticksCSG_t typecode = (OpticksCSG_t)getTypeCode(idx);      
    nvec4 param = getQuad(idx, 0);

    LOG(info) << "NCSG::import_r " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " itransform " << itransform 
              << " csgname " << CSGName(typecode) 
              << " param.x " << param.x
              << " param.y " << param.y
              << " param.z " << param.z
              << " param.w " << param.w
              ;

    nnode* node = NULL ;   
 
    if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    {
        int ltransform_idx = 0 ; // transforms are only applied to the right, currently 
        int rtransform_idx = getTransformIndex(idx) ; 

        switch(typecode)
        {
           case CSG_UNION:        node = make_nunion_ptr(NULL, NULL )        ; break ; 
           case CSG_INTERSECTION: node = make_nintersection_ptr(NULL, NULL ) ; break ; 
           case CSG_DIFFERENCE:   node = make_ndifference_ptr(NULL, NULL )   ; break ; 
           default:               node = NULL                                 ; break ; 
        }
        assert(node);

        node->parent = parent ; 
        node->transform = import_transform( itransform ) ;

        std::cout << "NCSG::import_r(oper)" 
                  << " idx " << idx
                  << " csgname " << CSGName(typecode) 
                  << " rtransform_idx " << rtransform_idx
                  << " itransform " << itransform
                  << std::endl
                  ;

        if(node->transform) std::cout << " transform " << *node->transform ;
        else                std::cout << " no-transform " ; 
        std::cout << std::endl ; 


        // NB recursive call after the "visit" 
        node->left = import_r(idx*2+1, node, ltransform_idx );     
        node->right = import_r(idx*2+2, node, rtransform_idx );

        // internal node can carry its own transform as well as have a child rtransform
    }
    else 
    {
        switch(typecode)
        {
           case CSG_SPHERE: node = make_nsphere_ptr(param)   ; break ; 
           case CSG_BOX:    node = make_nbox_ptr(param)      ; break ; 
           default:         node = NULL ; break ; 
        }       

        assert(node); 

        // structure of recursive call dictated by need for 
        // the primitive to know parent and its transform here...
        // so the ancestor transforms can be multiplied straightaway 
        // without some 2nd pass

        node->parent = parent ; 
        node->transform = import_transform( itransform ) ;
        node->gtransform = node->global_transform(); 

        if(node->transform) std::cout << " transform " << *node->transform ;
        else                std::cout << " no-transform " ;
        std::cout << std::endl ; 
 
        if(node->gtransform) std::cout << " gtransform " << *node->gtransform ;
        else                std::cout << " no-gtransform " ;
        std::cout << std::endl ; 


    }
    if(node == NULL) LOG(fatal) << "NCSG::import_r"
                                << " TYPECODE NOT IMPLEMENTED " 
                                << " idx " << idx 
                                << " typecode " << typecode
                                << " csgname " << CSGName(typecode)
                                ;
    assert(node); 
    return node ; 
} 





NCSG* NCSG::FromNode(nnode* root, const char* boundary)
{
    NCSG* tree = new NCSG(root);
    tree->setBoundary( boundary );
    tree->export_();
    return tree ; 
}

void NCSG::export_()
{
    assert(m_nodes);
    LOG(debug) << "NCSG::export_ "
              << " exporting CSG node tree into buffer "
              << " num_nodes " << m_num_nodes
              << " height " << m_height 
              ;
    export_r(m_root, 0);
}

void NCSG::export_r(nnode* node, unsigned idx)
{
    assert(idx < m_num_nodes); 
    LOG(trace) << "NCSG::export_r"
              << " idx " << idx 
              << node->desc()
              ;

    npart pt = node->part();

    m_nodes->setPart( pt, idx);  // writes 4 quads to buffer

    if(node->left && node->right)
    {
        export_r(node->left,  2*idx + 1);
        export_r(node->right, 2*idx + 2);
    }  
}




void NCSG::dump(const char* msg)
{
    LOG(info) << msg << " " << desc() ; 
    if(!m_root) return ;
    m_root->dump("NCSG::dump (root)");

    if(m_meta)
    m_meta->dump(); 
}

std::string NCSG::desc()
{
    std::string node_sh = m_nodes ? m_nodes->getShapeString() : "" ;    
    std::string tran_sh = m_transforms ? m_transforms->getShapeString() : "" ;    
    std::stringstream ss ; 
    ss << "NCSG " 
       << " index " << m_index
       << " treedir " << ( m_treedir ? m_treedir : "NULL" ) 
       << " node_sh " << node_sh  
       << " tran_sh " << tran_sh  
       << " boundary " << ( m_boundary ? m_boundary : "NULL" ) 
       << " meta " << m_meta->desc()
       ;
    return ss.str();  
}

int NCSG::Deserialize(const char* base, std::vector<NCSG*>& trees)
{
    assert(trees.size() == 0);
    LOG(info) << base ; 

    std::string txtpath = BFile::FormPath(base, FILENAME) ;
    bool exists = BFile::ExistsFile(txtpath.c_str() ); 

    if(!exists) LOG(fatal) << "NCSG::Deserialize"
                           << " file does not exist " 
                           << txtpath 
                           ;
    assert(exists); 

    NTxt bnd(txtpath.c_str());
    bnd.read();
    //bnd.dump("NCSG::Deserialize");    

    unsigned nbnd = bnd.getNumLines();

    for(unsigned i=0 ; i < nbnd ; i++)
    {
        //std::string path = BFile::FormPath(base, BStr::concat(NULL, i, ".npy"));  
        std::string treedir = BFile::FormPath(base, BStr::itoa(i));  

        NCSG* tree = new NCSG(treedir.c_str(), i);
        tree->setBoundary( bnd.getLine(i) );

        tree->load();    // the buffer (no bbox from user input python)
        tree->import();  // from buffer into CSG node tree 
        tree->export_(); // from CSG node tree back into buffer, with bbox added   

        LOG(info) << "NCSG::Deserialize [" << i << "] " << tree->desc() ; 

        trees.push_back(tree);  
    }
    return 0 ; 
}


