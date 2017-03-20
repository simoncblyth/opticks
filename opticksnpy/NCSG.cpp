#include <cstring>
#include <algorithm>
#include <sstream>

#include "BStr.hh"
#include "BFile.hh"

#include "OpticksCSG.h"
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

NCSG::NCSG(const char* path, unsigned index) 
   :
   m_index(index),
   m_root(NULL),
   m_path(path ? strdup(path) : NULL),
   m_data(NULL),

   m_num_nodes(0),
   m_height(-1),
   m_boundary(NULL)
{
}

void NCSG::load()
{
    m_data = NPY<float>::load(m_path);
    m_num_nodes  = m_data->getShape(0) ;  
    unsigned nj = m_data->getShape(1);
    unsigned nk = m_data->getShape(2);
    assert( nj == NJ );
    assert( nk == NK );

    m_height = -1 ; 
    int h = MAX_HEIGHT ; 
    while(h--) if(TREE_NODES(h) == m_num_nodes) m_height = h ; 
    assert(m_height >= 0); // must be complete binary tree sized 1, 3, 7, 15, 31, ...
}


NCSG::NCSG(nnode* root, unsigned index) 
   :
   m_index(index),
   m_root(root),
   m_path(NULL),
   m_data(NULL),
   m_num_nodes(0),
   m_height(root->maxdepth()),
   m_boundary(NULL)
{
   m_num_nodes = NumNodes(m_height);
   m_data = NPY<float>::make( m_num_nodes, NJ, NK);
   m_data->zero();
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
NPY<float>* NCSG::getBuffer()
{
    return m_data ; 
}
const char* NCSG::getBoundary()
{
    return m_boundary ; 
}
const char* NCSG::getPath()
{
    return m_path ; 
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
    return m_data->getUInt(idx,TYPECODE_J,TYPECODE_K,0u);
}
nvec4 NCSG::getQuad(unsigned idx, unsigned j)
{
    nvec4 qj = m_data->getVQuad(idx, j) ;
    return qj ;
}

void NCSG::import()
{
    assert(m_data);
    LOG(info) << "NCSG::import"
              << " importing buffer into CSG node tree "
              << " num_nodes " << m_num_nodes
              << " height " << m_height 
              ;

    m_root = import_r(0) ; 
}

nnode* NCSG::import_r(unsigned idx)
{
    if(idx >= m_num_nodes) return NULL ; 
        
    OpticksCSG_t typecode = (OpticksCSG_t)getTypeCode(idx);      
    nvec4 param = getQuad(idx, 0);

    LOG(info) << "NCSG::import_r " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " csgname " << CSGName(typecode) 
              << " param.x " << param.x
              << " param.y " << param.y
              << " param.z " << param.z
              << " param.w " << param.w
              ;

    nnode* node = NULL ;   
 
    if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    {
        nnode* left = import_r(idx*2+1); // 0-based binary tree indexing 
        nnode* right = import_r(idx*2+2);

        switch(typecode)
        {
           case CSG_UNION:        node = make_nunion_ptr(left, right )        ; break ; 
           case CSG_INTERSECTION: node = make_nintersection_ptr(left, right ) ; break ; 
           case CSG_DIFFERENCE:   node = make_ndifference_ptr(left, right )   ; break ; 
           default:               node = NULL                                 ; break ; 
        }
    }
    else 
    {
        switch(typecode)
        {
           case CSG_SPHERE: node = make_nsphere_ptr(param)   ; break ; 
           case CSG_BOX:    node = make_nbox_ptr(param)      ; break ; 
           default:         node = NULL ; break ; 
        }       
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
    assert(m_data);
    LOG(info) << "NCSG::export_ "
              << " exporting CSG node tree into buffer "
              << " num_nodes " << m_num_nodes
              << " height " << m_height 
              ;
    export_r(m_root, 0);
}

void NCSG::export_r(nnode* node, unsigned idx)
{
    assert(idx < m_num_nodes); 
    LOG(info) << "NCSG::export_r"
              << " idx " << idx 
              << node->desc()
              ;

    npart pt = node->part();

    m_data->setPart( pt, idx);  // writes 4 quads to buffer

    if(node->left && node->right)
    {
        export_r(node->left,  2*idx + 1);
        export_r(node->right, 2*idx + 2);
    }  
}




void NCSG::dump(const char* msg)
{
    LOG(info) << msg ; 
    if(!m_root) return ;
    m_root->dump("NCSG::dump (root)");
}

std::string NCSG::desc()
{
    std::string sh = m_data ? m_data->getShapeString() : "" ;    
    std::stringstream ss ; 
    ss << "NCSG " 
       << " index " << m_index
       << " path " << ( m_path ? m_path : "NULL" ) 
       << " shape " << sh  
       << " boundary " << m_boundary 
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
    bnd.dump("NCSG::Deserialize");    

    unsigned nbnd = bnd.getNumLines();

    for(unsigned i=0 ; i < nbnd ; i++)
    {
        std::string path = BFile::FormPath(base, BStr::concat(NULL, i, ".npy"));  
        NCSG* tree = new NCSG(path.c_str(), i);
        tree->setBoundary( bnd.getLine(i) );

        tree->load();    // the buffer (no bbox from user input python)
        tree->import();  // from buffer into CSG node tree 
        tree->export_(); // from CSG node tree back into buffer, with bbox added   

        LOG(info) << "NCSG::Deserialize [" << i << "] " << tree->desc() ; 

        trees.push_back(tree);  
    }
    return 0 ; 
}


