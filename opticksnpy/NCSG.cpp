



#include <cstring>
#include <algorithm>
#include <sstream>

#include "BStr.hh"
#include "BFile.hh"

#include "OpticksCSG.h"

#include "NGLMExt.hpp"
#include "NParameters.hpp"
#include "NPart.h"

// primitives
#include "NSphere.hpp"
#include "NZSphere.hpp"
#include "NBox.hpp"
#include "NSlab.hpp"
#include "NPlane.hpp"
#include "NCylinder.hpp"

#include "NNode.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"
#include "NTxt.hpp"

#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )

#include "PLOG.hh"

const char* NCSG::FILENAME = "csg.txt" ; 
const unsigned NCSG::NTRAN = 3 ; 


// ctor : booting via deserialization of directory 
NCSG::NCSG(const char* treedir, unsigned index, int verbosity) 
   :
   m_index(index),
   m_verbosity(verbosity),
   m_root(NULL),
   m_treedir(treedir ? strdup(treedir) : NULL),
   m_nodes(NULL),
   m_transforms(NULL),
   m_gtransforms(NULL),
   m_meta(NULL),

   m_num_nodes(0),
   m_num_transforms(0),
   m_height(-1),
   m_boundary(NULL)
{
}

// ctor : booting from in memory node tree
NCSG::NCSG(nnode* root, unsigned index, int verbosity ) 
   :
   m_index(index),
   m_verbosity(verbosity),
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




void NCSG::load()
{
    std::string metapath = BFile::FormPath(m_treedir, "meta.json") ;
    std::string nodepath = BFile::FormPath(m_treedir, "nodes.npy") ;
    std::string tranpath = BFile::FormPath(m_treedir, "transforms.npy") ;

    m_meta = new NParameters ; 
    m_meta->load_( metapath.c_str() );

    int verbosity = m_meta->get<int>("verbosity", "-1");
    if(verbosity > -1 && verbosity != m_verbosity) 
    {
        LOG(fatal) << "NCSG::load changing verbosity via metadata " 
                   << " treedir " << m_treedir
                   << " old " << m_verbosity 
                   << " new " << verbosity 
                   ; 
        m_verbosity = verbosity ; 
    }


    m_nodes = NPY<float>::load(nodepath.c_str());

    m_num_nodes  = m_nodes->getShape(0) ;  
    unsigned nj = m_nodes->getShape(1);
    unsigned nk = m_nodes->getShape(2);
    assert( nj == NJ );
    assert( nk == NK );

    if(BFile::ExistsFile(tranpath.c_str()))
    {
        NPY<float>* src = NPY<float>::load(tranpath.c_str());
        assert(src); 
        bool valid_src = src->hasShape(-1,4,4) ;
        if(!valid_src) 
            LOG(fatal) << "NCSG::load"
                       << " invalid src transforms "
                       << " tranpath " << tranpath
                       << " src_sh " << src->getShapeString()
                       ;

        assert(valid_src);
        unsigned ni = src->getShape(0) ;

                  
        assert(NTRAN == 2 || NTRAN == 3);
        NPY<float>* transforms = NTRAN == 2 ? NPY<float>::make_paired_transforms(src) : NPY<float>::make_triple_transforms(src) ;
        assert(transforms->hasShape(ni,NTRAN,4,4));

        m_transforms = transforms ; 
        m_gtransforms = NPY<float>::make(0,NTRAN,4,4) ;  // for collecting unique gtransforms

        m_num_transforms  = ni  ;  
    }


    m_height = -1 ; 
    int h = MAX_HEIGHT ; 
    while(h--) if(TREE_NODES(h) == m_num_nodes) m_height = h ; 
    assert(m_height >= 0); // must be complete binary tree sized 1, 3, 7, 15, 31, ...
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
NPY<float>* NCSG::getTransformBuffer()
{
    return m_transforms ; 
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
int NCSG::getVerbosity()
{
    return m_verbosity ; 
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
    return m_nodes->getUInt(idx,TRANSFORM_J,TRANSFORM_K,0u);
}





nquad NCSG::getQuad(unsigned idx, unsigned j)
{
    nquad qj ; 
    qj.f = m_nodes->getVQuad(idx, j) ;
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

    m_root = import_r(0, NULL) ; 

    check();

}


nmat4pair* NCSG::import_transform_pair(unsigned itra)
{
    // itra is a 1-based index, with 0 meaning None

    assert(NTRAN == 2);
    if(itra == 0 || m_transforms == NULL ) return NULL ; 

    unsigned idx = itra - 1 ; 

    assert( idx < m_num_transforms );
    assert(m_transforms->hasShape(-1,NTRAN,4,4));

    nmat4pair* pair = m_transforms->getMat4PairPtr(idx);

    return pair ; 
}


nmat4triple* NCSG::import_transform_triple(unsigned itra)
{
    // itra is a 1-based index, with 0 meaning None

    assert(NTRAN == 3);
    if(itra == 0 || m_transforms == NULL ) return NULL ; 

    unsigned idx = itra - 1 ; 

    assert( idx < m_num_transforms );
    assert(m_transforms->hasShape(-1,NTRAN,4,4));

    nmat4triple* triple = m_transforms->getMat4TriplePtr(idx);

    return triple ; 
}



nnode* NCSG::import_r(unsigned idx, nnode* parent)
{
    if(idx >= m_num_nodes) return NULL ; 
        
    OpticksCSG_t typecode = (OpticksCSG_t)getTypeCode(idx);      
    int transform_idx = getTransformIndex(idx) ; 

    nquad p0 = getQuad(idx, 0);
    nquad p1 = getQuad(idx, 1);
    nquad p2 = getQuad(idx, 2);

    //if(m_verbosity > 2)
    LOG(info) << "NCSG::import_r " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " transform_idx " << transform_idx 
              << " csgname " << CSGName(typecode) 
              << " p0.f.x " << p0.f.x
              << " p0.f.y " << p0.f.y
              << " p0.f.z " << p0.f.z
              << " p0.f.w " << p0.f.w
              << " p0.u.w " << p0.u.w
              << " p1.f.x " << p1.f.x
              << " p1.f.y " << p1.f.y
              << " p1.f.z " << p1.f.z
              << " p1.f.w " << p1.f.w
              << " p2.u.x " << p2.u.x
              ;

    nnode* node = NULL ;   
 
    if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    {
        switch(typecode)
        {
           case CSG_UNION:        node = new nunion(make_union(NULL, NULL )) ; break ; 
           case CSG_INTERSECTION: node = new nintersection(make_intersection(NULL, NULL )) ; break ; 
           case CSG_DIFFERENCE:   node = new ndifference(make_difference(NULL, NULL ))   ; break ; 
           default:               node = NULL                                 ; break ; 
        }
        assert(node);

        node->parent = parent ; 

        node->transform = import_transform_triple( transform_idx ) ;

        if(m_verbosity > 2)
        {
            std::cout << "NCSG::import_r(oper)" 
                      << " idx " << idx
                      << " csgname " << CSGName(typecode) 
                      << " transform_idx " << transform_idx
                      << std::endl
                      ;

            if(node->transform) std::cout << " transform " << *node->transform ;
            else                std::cout << " no-transform " ; 
            std::cout << std::endl ; 
        }

        node->left = import_r(idx*2+1, node );  
        node->right = import_r(idx*2+2, node );
        // recursive calls after "visit" as full ancestry needed for transform collection
        // once reach the primitives
    }
    else 
    {
        switch(typecode)
        {
           case CSG_SPHERE:   node = new nsphere(make_sphere(p0))           ; break ; 
           case CSG_ZSPHERE:  node = new nzsphere(make_zsphere(p0,p1,p2))   ; break ; 
           case CSG_BOX:      node = new nbox(make_box(p0))                 ; break ; 
           case CSG_SLAB:     node = new nslab(make_slab(p0, p1))           ; break ; 
           case CSG_PLANE:    node = new nplane(make_plane(p0))             ; break ; 
           case CSG_CYLINDER: node = new ncylinder(make_cylinder(p0, p1))   ; break ; 
           default:           node = NULL ; break ; 
        }       

        assert(node && "unhandled CSG type"); 

        // structure of recursive call dictated by need for 
        // the primitive to know parent and its transform here...
        // so the ancestor transforms can be multiplied straightaway 
        // without some 2nd pass

        node->parent = parent ; 
        node->transform = import_transform_triple( transform_idx ) ;


        nmat4triple* gtransform = node->global_transform();
        unsigned gtransform_idx = 0 ; 
        if(gtransform)
        {
            NPY<float>* gtmp = NPY<float>::make(1,NTRAN,4,4);
            gtmp->zero();
            gtmp->setMat4Triple(gtransform, 0);
            gtransform_idx = 1 + m_gtransforms->addItemUnique( gtmp, 0 ) ; 
            delete gtmp ; 
        }
        node->gtransform = gtransform ; 
        node->gtransform_idx = gtransform_idx ; // 1-based, 0 for None



    }
    if(node == NULL) LOG(fatal) << "NCSG::import_r"
                                << " TYPECODE NOT IMPLEMENTED " 
                                << " idx " << idx 
                                << " typecode " << typecode
                                << " csgname " << CSGName(typecode)
                                ;
    assert(node); 

    node->idx = idx ; 

    return node ; 
} 




void NCSG::check()
{
    check_r( m_root );


    unsigned ni =  m_gtransforms ? m_gtransforms->getNumItems() : 0  ;

    if(m_verbosity > 1)
    LOG(info) << "NCSG::check"
              << " unique gtransforms " << ni
              ;

    for(unsigned i=0 ; i < ni ; i++)
    {
        //const nmat4pair* u_gtran = m_gtransforms->getMat4PairPtr(i);
        const nmat4triple* u_gtran = m_gtransforms->getMat4TriplePtr(i);
        std::cout 
                  << "[" << std::setw(2) << i << "] " 
                  << *u_gtran 
                  << std::endl ; 
    }
}

void NCSG::check_r(nnode* node)
{

    if(m_verbosity > 2)
    {
        if(node->gtransform)
        {
            std::cout << "NCSG::check_r"
                      << " gtransform_idx " << node->gtransform_idx
                      << std::endl 
                      ;
        }
        if(node->transform)
        {
            std::cout << "NCSG::check_r"
                      << " transform " << *node->transform
                      << std::endl 
                      ;
        }
    }


    if(node->left && node->right)
    {
        check_r(node->left);
        check_r(node->right);
    }
}




NCSG* NCSG::FromNode(nnode* root, const char* boundary)
{
    NCSG* tree = new NCSG(root, 0);
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

    // crucial 2-step here, where m_nodes gets totally rewritten
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
    std::string node_sh = m_nodes ? m_nodes->getShapeString() : "-" ;    
    std::string tran_sh = m_transforms ? m_transforms->getShapeString() : "-" ;    
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

int NCSG::Deserialize(const char* base, std::vector<NCSG*>& trees, int verbosity )
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
        std::string treedir = BFile::FormPath(base, BStr::itoa(i));  

        NCSG* tree = new NCSG(treedir.c_str(), i, verbosity );
        tree->setBoundary( bnd.getLine(i) );

        tree->load();    // m_nodes, the user input serialization buffer (no bbox from user input python)
        tree->import();  // input m_nodes buffer into CSG nnode tree 

        tree->export_(); // from CSG nnode tree back into *same* buffer, with bbox added   

        LOG(info) << "NCSG::Deserialize [" << i << "] " << tree->desc() ; 

        trees.push_back(tree);  
    }
    return 0 ; 
}


