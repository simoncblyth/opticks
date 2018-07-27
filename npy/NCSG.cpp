#include <cstring>
#include <algorithm>
#include <sstream>

#include "SSys.hh"

#include "BStr.hh"
#include "BFile.hh"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NParameters.hpp"

//#include "NPart.h"

#include "NTrianglesNPY.hpp"
#include "NPolygonizer.hpp"

#include "NPrimitives.hpp"

#include "NSceneConfig.hpp"
#include "NScan.hpp"
#include "NNode.hpp"
#include "NNodePoints.hpp"
#include "NNodeUncoincide.hpp"
#include "NNodeNudger.hpp"
#include "NBBox.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"
#include "NCSGData.hpp"
#include "NTxt.hpp"



#include "PLOG.hh"

const float NCSG::SURFACE_EPSILON = 1e-5f ; 


// ctor : booting via deserialization of directory 
NCSG::NCSG(const char* treedir) 
   :
   m_treedir(treedir ? strdup(treedir) : NULL),
   m_index(0),
   m_surface_epsilon(SURFACE_EPSILON),
   m_verbosity(0),
   m_usedglobally(true),   // changed to true : June 2018, see notes/issues/subtree_instances_missing_transform.rst
   m_root(NULL),
   m_points(NULL),
   m_uncoincide(NULL),
   m_nudger(NULL),
   m_csgdata(new NCSGData),
   m_boundary(NULL),
   m_config(NULL),
   m_gpuoffset(0,0,0),
   m_container(0),
   m_containerscale(2.f),
   m_tris(NULL),
   m_soIdx(0),
   m_lvIdx(0)
{
}

// ctor : booting from in memory node tree : cannot be const because of the nudger 
NCSG::NCSG(nnode* root ) 
   :
   m_treedir(NULL),
   m_index(0),
   m_surface_epsilon(SURFACE_EPSILON),
   m_verbosity(root->verbosity),
   m_usedglobally(true),   // changed to true : June 2018, see notes/issues/subtree_instances_missing_transform.rst
   m_root(root),
   m_points(NULL),
   m_uncoincide(make_uncoincide()),
   m_nudger(make_nudger()),
   m_csgdata(new NCSGData),
   m_boundary(NULL),
   m_config(NULL),
   m_gpuoffset(0,0,0),
   m_container(0),
   m_containerscale(2.f),
   m_tris(NULL),
   m_soIdx(0),
   m_lvIdx(0)
{
   setBoundary( root->boundary );  // boundary spec
   m_csgdata->init_buffers(root->maxdepth()) ;  
}

NCSGData* NCSG::getCSGData() const 
{
   return m_csgdata ; 
}

NNodeUncoincide* NCSG::make_uncoincide() const 
{
    return NULL ;  
    //return new NNodeUncoincide(m_root, m_surface_epsilon, m_root->verbosity);
}
NNodeNudger* NCSG::get_nudger() const 
{
    return m_nudger ; 
}
NNodeNudger* NCSG::make_nudger() const 
{
   // when test running from nnode there is no metadata or treedir
   // LOG(info) << soname() << " treeNameIdx " << getTreeNameIdx() ; 

    NNodeNudger* nudger = new NNodeNudger(m_root, m_surface_epsilon, m_root->verbosity);
    return nudger ; 
}


std::string NCSG::TestVolumeName(const char* shapename, const char* suffix, int idx) // static
{
    std::stringstream ss ; 
    ss << shapename << "_" << suffix ; 
    if(idx > -1) ss << idx ; 
    ss << "_" ; 
    return ss.str();
}

std::string NCSG::getTestPVName() const 
{
    OpticksCSG_t type = getRootType() ;
    const char* shapename = CSGName(type); 
    unsigned idx = getIndex();
    return TestVolumeName( shapename, "pv", idx);     
}
std::string NCSG::getTestLVName() const 
{
    OpticksCSG_t type = getRootType() ;
    const char* shapename = CSGName(type); 
    unsigned idx = getIndex();
    return TestVolumeName( shapename, "lv", idx);     
}


std::string NCSG::lvname() const {    return m_csgdata->getMeta<std::string>("lvname","-") ; }
std::string NCSG::soname() const {    return m_csgdata->getMeta<std::string>("soname","-") ; }
int         NCSG::treeindex() const { return m_csgdata->getMeta<int>("treeindex","-1") ; }
int         NCSG::depth() const {     return m_csgdata->getMeta<int>("depth","-1") ; }
int         NCSG::nchild() const {    return m_csgdata->getMeta<int>("nchild","-1") ; }
bool        NCSG::isSkip() const {        return m_csgdata->getMeta<int>("skip","0") == 1 ; }
bool        NCSG::is_uncoincide() const { return m_csgdata->getMeta<int>("uncoincide","1") == 1 ; }
int         NCSG::getEmit() const {       return m_csgdata->getMeta<int>("emit","0") ;  }

bool NCSG::isEmit() const 
{  
    int emit = getEmit() ;
    return emit == 1 || emit == -1 ;
}
void NCSG::setEmit(int emit) // used by --testauto
{
    m_csgdata->setMeta<int>("emit", emit);
}

void NCSG::setEmitConfig(const char* emitconfig)
{
    m_csgdata->setMeta<std::string>("emitconfig", emitconfig );
}
const char* NCSG::getEmitConfig() const 
{ 
    std::string ec = m_csgdata->getMeta<std::string>("emitconfig","") ;
    return ec.empty() ? NULL : strdup(ec.c_str()) ; 
}

std::string NCSG::meta() const 
{
    std::stringstream ss ; 
    ss << " treeindex " << treeindex()
       << " depth " << depth()
       << " nchild " << nchild()
       << " lvname " << lvname() 
       << " soname " << soname() 
       << " isSkip " << isSkip()
       << " is_uncoincide " << is_uncoincide()
       << " emit " << getEmit()
       ;

    return ss.str();
}

std::string NCSG::smry() const 
{
    std::stringstream ss ; 
    ss 
       << " ht " << std::setw(2) << getHeight() 
       << " nn " << std::setw(4) << getNumNodes()
       << " tri " << std::setw(6) << getNumTriangles()
       << " tmsg " << ( m_tris ? m_tris->getMessage() : "NULL-tris" ) 
       << " iug " << m_usedglobally 
       << m_csgdata->smry() 
      ;

    return ss.str();
}

NParameters* NCSG::getNodeMetadata(unsigned idx) const 
{
    return m_csgdata->getNodeMetadata(idx); 
}

void NCSG::increaseVerbosity(int verbosity)
{
    if(verbosity > -1) 
    {
        if(verbosity > m_verbosity)
        {
            LOG(debug) << "NCSG::increaseVerbosity" 
                       << " treedir " << m_treedir
                       << " old " << m_verbosity 
                       << " new " << verbosity 
                       ; 
        }
        else
        {
            LOG(debug) << "NCSG::increaseVerbosity IGNORING REQUEST TO DECREASE verbosity " 
                       << " treedir " << m_treedir
                       << " current verbosity " << m_verbosity 
                       << " requested : " << verbosity 
                       ; 
 
        }
    }
}


void NCSG::postload()
{
    m_container       = m_csgdata->getMeta<int>("container", "-1");
    m_containerscale  = m_csgdata->getMeta<float>("containerscale", "2.");

    std::string gpuoffset = m_csgdata->getMeta<std::string>("gpuoffset", "0,0,0" );
    m_gpuoffset = gvec3(gpuoffset);  

    int verbosity     = m_csgdata->getMeta<int>("verbosity", "0");
    increaseVerbosity(verbosity);
}

void NCSG::save(const char* treedir_ ) const 
{
    bool same_dir = m_treedir && strcmp( treedir_, m_treedir) == 0  ;
    if( same_dir ) LOG(fatal) << "saving back into the same dir as loaded from is not allowed " ; 
    assert( !same_dir) ; 

    assert( treedir_ ) ; 
    LOG(info) << " treedir_ " << treedir_ ; 
    m_csgdata->save(treedir_) ;  
}

void NCSG::loadsrc()
{
    if(m_index % 100 == 0 && m_verbosity > 0)
    {
        LOG(info) << "NCSG::load " 
                  << " index " << m_index
                  << " treedir " << m_treedir 
                  ; 
    }

    assert(m_csgdata);
    m_csgdata->loadsrc( m_treedir ) ; 
    postload();

    LOG(debug) << "NCSG::load DONE " ; 
}




nnode* NCSG::getRoot() const 
{
    return m_root ; 
}
OpticksCSG_t NCSG::getRootType() const 
{
    assert(m_root);
    return m_root->type ; 
}

unsigned NCSG::getHeight() const 
{
    return m_csgdata->getHeight(); 
}
unsigned NCSG::getNumNodes() const 
{
    return m_csgdata->getNumNodes() ; 
}



NPY<float>* NCSG::getNodeBuffer() const 
{
    return m_csgdata->getNodeBuffer() ; 
}
NPY<float>* NCSG::getTransformBuffer() const 
{
    return m_csgdata->getTransformBuffer() ; 
}
NPY<float>* NCSG::getGTransformBuffer() const
{
    return m_csgdata->getGTransformBuffer() ; 
}



NPY<float>* NCSG::getSrcTransformBuffer() const 
{
    return m_csgdata->getSrcTransformBuffer() ; 
}
NPY<float>* NCSG::getSrcNodeBuffer() const 
{
    return m_csgdata->getSrcNodeBuffer() ; 
}
NPY<float>* NCSG::getSrcPlaneBuffer() const 
{
    return m_csgdata->getSrcPlaneBuffer() ; 
}
NPY<unsigned>* NCSG::getSrcIdxBuffer() const 
{
    return m_csgdata->getSrcIdxBuffer() ; 
}



NParameters* NCSG::getMetaParameters(int idx) const
{
    return m_csgdata->getMetaParameters(idx) ; 
}




const char* NCSG::getBoundary() const 
{
    return m_boundary ; 
}
const char* NCSG::getTreeDir() const 
{
    return m_treedir ; 
}

const char* NCSG::getTreeName() const 
{
    std::string name = BFile::Name(m_treedir ? m_treedir : "-1") ; 
    return strdup(name.c_str());
}
int NCSG::getTreeNameIdx() const 
{
    const char* name = getTreeName();
    return BStr::atoi(name, -1);
}



unsigned NCSG::getIndex() const 
{
    return m_index ; 
}
int NCSG::getVerbosity() const 
{
    return m_verbosity ; 
}
bool NCSG::isContainer() const 
{
    return m_container > 0  ; 
}
float NCSG::getContainerScale() const 
{
    return m_containerscale  ; 
}
bool NCSG::isUsedGlobally() const 
{
    return m_usedglobally ; 
}







void NCSG::setIndex(unsigned index)
{
    m_index = index ; 
}
void NCSG::setVerbosity(int verbosity)
{
    m_verbosity = verbosity ; 
}
void NCSG::setIsUsedGlobally(bool usedglobally )
{
    m_usedglobally = usedglobally ; 
}


void NCSG::setBoundary(const char* boundary)
{
    m_boundary = boundary ? strdup(boundary) : NULL ; 
}

void NCSG::setConfig(const NSceneConfig* config)
{
    m_config = config ; 
}



/** 
NCSG::import : from complete binary tree buffer into nnode tree
------------------------------------------------------------------
**/

void NCSG::import()
{
    if(m_verbosity > 1)
        LOG(info) << "NCSG::import START" 
                  << " verbosity " << m_verbosity
                  << " treedir " << m_treedir
                  << " smry " << smry()
                  ; 

    if(m_verbosity > 0)
    {
        LOG(info) << "NCSG::import"
                  << " importing buffer into CSG node tree "
                  << " num_nodes " << getNumNodes()
                  << " height " << getHeight()
                  ;
    }

    m_csgdata->prepareForImport() ;  // from m_srctransforms to m_transforms, and get m_gtransforms ready to collect

    m_root = import_r(0, NULL) ;  

    m_root->set_treedir(m_treedir) ; 
    m_root->set_treeidx(getTreeNameIdx()) ; 

    postimport();

    if(m_verbosity > 5) check();  // recursive transform dumping 

    if(m_verbosity > 1) LOG(info) << "NCSG::import DONE " ; 
}


void NCSG::postimport()
{
    m_nudger = make_nudger() ; 

    //m_uncoincide = make_uncoincide(); 
    //m_uncoincide->uncoincide();

    //postimport_autoscan();
}


/**
NCSG::import_r
----------------

Importing : constructs the node tree from src buffers 
loaded by loadsrc ( which were written by analytic/csg.py ) 

On import the gtransforms (**for primitives only**) are constructed 
by multiplication down the tree, and uniquely collected into m_gtransforms 
with the 1-based gtransforms_idx being set on the node.

**/

nnode* NCSG::import_r(unsigned idx, nnode* parent)
{
    if(idx >= getNumNodes()) return NULL ; 
    
    // from srcnodes     
    OpticksCSG_t typecode = (OpticksCSG_t)m_csgdata->getTypeCode(idx);      
    int transform_idx = m_csgdata->getTransformIndex(idx) ; 
    bool complement = m_csgdata->isComplement(idx) ; 

    LOG(debug) << "NCSG::import_r"
              << " idx " << idx
              << " transform_idx " << transform_idx
              << " complement " << complement 
              ;

    nnode* node = NULL ;   
 
    if(typecode == CSG_UNION || typecode == CSG_INTERSECTION || typecode == CSG_DIFFERENCE)
    {
        node = import_operator( idx, typecode ) ; 

        node->parent = parent ; 
        node->idx = idx ; 
        node->complement = complement ; 

        node->transform = m_csgdata->import_transform_triple( transform_idx ) ;  // from m_transforms, expecting (-1,3,4,4)

        node->left = import_r(idx*2+1, node );  
        node->right = import_r(idx*2+2, node );

        node->left->other = node->right ;   // used by NOpenMesh 
        node->right->other = node->left ; 

        // recursive calls after "visit" as full ancestry needed for transform collection once reach primitives
    }
    else 
    {
        node = import_primitive( idx, typecode ); 

        node->parent = parent ;                // <-- parent hookup needed prior to gtransform collection 
        node->idx = idx ; 
        node->complement = complement ; 

        node->transform = m_csgdata->import_transform_triple( transform_idx ) ;  // from m_transforms, expecting (-1,3,4,4)

        const nmat4triple* gtransform = node->global_transform();   

        // see opticks/notes/issues/subtree_instances_missing_transform.rst
        //if(gtransform == NULL && m_usedglobally)
        if(gtransform == NULL )  // move to giving all primitives a gtransform 
        {
            gtransform = nmat4triple::make_identity() ;
        }

        unsigned gtransform_idx = gtransform ? addUniqueTransform(gtransform) : 0 ; 

        node->gtransform = gtransform ; 
        node->gtransform_idx = gtransform_idx ; // 1-based, 0 for None

        LOG(error) << "import_r node->gtransform_idx " << node->gtransform_idx ; 
    }
    assert(node); 

    NParameters* nodemeta = getNodeMetadata(idx);
    if(nodemeta) node->meta = nodemeta ; 

    // Avoiding duplication between the operator and primitive branches 
    // in the above is not sufficient reason to put things here, so very late.
    return node ; 
} 

nnode* NCSG::import_operator( unsigned idx, OpticksCSG_t typecode )
{
    if(m_verbosity > 2)
    {
    LOG(info) << "NCSG::import_operator " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " csgname " << CSGName(typecode) 
              ;
    }

    nnode* node = NULL ;   
    switch(typecode)
    {
       case CSG_UNION:        node = new nunion(nunion::make_union(NULL, NULL )) ; break ; 
       case CSG_INTERSECTION: node = new nintersection(nintersection::make_intersection(NULL, NULL )) ; break ; 
       case CSG_DIFFERENCE:   node = new ndifference(ndifference::make_difference(NULL, NULL ))   ; break ; 
       default:               node = NULL                                 ; break ; 
    }
    assert(node);
    return node ; 
}

nnode* NCSG::import_primitive( unsigned idx, OpticksCSG_t typecode )
{
    // from srcnodes buffer
    nquad p0 = m_csgdata->getQuad(idx, 0);
    nquad p1 = m_csgdata->getQuad(idx, 1);
    nquad p2 = m_csgdata->getQuad(idx, 2);
    nquad p3 = m_csgdata->getQuad(idx, 3);

    if(m_verbosity > 2)
    {
    LOG(info) << "NCSG::import_primitive  " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " csgname " << CSGName(typecode) 
              ;
    }

    nnode* node = NULL ;   
    switch(typecode)
    {
       case CSG_SPHERE:         node = new nsphere(make_sphere(p0))           ; break ; 
       case CSG_ZSPHERE:        node = new nzsphere(make_zsphere(p0,p1,p2))   ; break ; 
       case CSG_BOX:            node = new nbox(make_box(p0))                 ; break ; 
       case CSG_BOX3:           node = new nbox(make_box3(p0))                ; break ; 
       case CSG_SLAB:           node = new nslab(make_slab(p0, p1))           ; break ; 
       case CSG_PLANE:          node = new nplane(make_plane(p0))             ; break ; 
       case CSG_CYLINDER:       node = new ncylinder(make_cylinder(p0, p1))   ; break ; 
       case CSG_DISC:           node = new ndisc(make_disc(p0, p1))           ; break ; 
       case CSG_CONE:           node = new ncone(make_cone(p0))               ; break ; 
       case CSG_TORUS:          node = new ntorus(make_torus(p0))             ; break ; 
       case CSG_CUBIC:          node = new ncubic(make_cubic(p0,p1))          ; break ; 
       case CSG_HYPERBOLOID:    node = new nhyperboloid(make_hyperboloid(p0)) ; break ; 
       case CSG_TRAPEZOID:  
       case CSG_SEGMENT:  
       case CSG_CONVEXPOLYHEDRON:  
                                node = new nconvexpolyhedron(make_convexpolyhedron(p0,p1,p2,p3))   ; break ; 

       case CSG_ELLIPSOID: assert(0 && "ellipsoid should be zsphere at this level" )   ; break ; 
       default:           node = NULL ; break ; 
    }       


    if(node == NULL) 
    {
            LOG(fatal) << "NCSG::import_primitive"
                       << " TYPECODE NOT IMPLEMENTED " 
                       << " idx " << idx 
                       << " typecode " << typecode
                       << " csgname " << CSGName(typecode)
                       ;
    } 

    assert(node); 

    if(CSGHasPlanes(typecode)) 
    {
        import_srcplanes( node );
        import_srcvertsfaces( node );
    }

    if(m_verbosity > 3)
    {
    LOG(info) << "NCSG::import_primitive  " 
              << " idx " << idx 
              << " typecode " << typecode 
              << " csgname " << CSGName(typecode) 
              << " DONE " 
              ;
    } 
    return node ; 
}


void NCSG::import_srcvertsfaces(nnode* node)
{
    assert( node->has_planes() );
    
    NPY<float>* srcverts = m_csgdata->getSrcVertsBuffer() ; 
    NPY<int>*   srcfaces = m_csgdata->getSrcFacesBuffer() ; 

    if(!srcverts || !srcfaces) 
    {
        LOG(debug) << "NCSG::import_srcvertsfaces no srcverts  srcfaces " ; 
        return ; 
    }

    nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
    assert(cpol);

    std::vector<glm::vec3> _verts ;  
    std::vector<glm::ivec4> _faces ;  

    srcverts->copyTo(_verts);
    srcfaces->copyTo(_faces);

    cpol->set_srcvertsfaces(_verts, _faces);     
}

void NCSG::import_srcplanes(nnode* node)
{
    assert( node->has_planes() );

    nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
    assert(cpol);

    unsigned iplane = node->planeIdx() ;   // 1-based idx ?
    unsigned num_plane = node->planeNum() ;
    unsigned idx = iplane - 1 ;     

    if(m_verbosity > 3)
    {
    LOG(info) << "NCSG::import_planes"
              << " iplane " << iplane
              << " num_plane " << num_plane
              ;
    }

    assert( node->planes.size() == 0u );

    std::vector<glm::vec4> _planes ;  
    m_csgdata->getSrcPlanes(_planes, idx, num_plane ); 
    assert( _planes.size() == num_plane ) ; 

    cpol->set_planes(_planes);     
    assert( cpol->planes.size() == num_plane );
}


void NCSG::export_planes(nnode* node)
{
    if(!node->has_planes()) return ;

    nconvexpolyhedron* cpol = dynamic_cast<nconvexpolyhedron*>(node);
    assert(cpol);

    assert(node->planes.size() > 0);  

    unsigned planeNum = node->planes.size() ;

    NPY<float>* _planes = m_csgdata->getPlaneBuffer() ; 

    unsigned planeIdx0 = _planes->getNumItems(); 

    node->setPlaneIdx( planeIdx0 );
    node->setPlaneNum( planeNum );
    assert( planeNum > 3); 

    for(unsigned i=0 ; i < planeNum ; i++)
    {
        const glm::vec4& pln = node->planes[i] ; 
        _planes->add(pln); 
    } 
    unsigned planeIdxN = _planes->getNumItems(); 
    assert( planeIdxN == planeIdx0 + planeNum );
}


unsigned NCSG::addUniqueTransform( const nmat4triple* gtransform_ )
{
    bool no_offset = m_gpuoffset.x == 0.f && m_gpuoffset.y == 0.f && m_gpuoffset.z == 0.f ;

    bool reverse = true ; // <-- apply transfrom at root of transform hierarchy (rather than leaf)

    const nmat4triple* gtransform = no_offset ? gtransform_ : gtransform_->make_translated(m_gpuoffset, reverse, "NCSG::addUniqueTransform" ) ;

    /*
    std::cout << "NCSG::addUniqueTransform"
              << " orig " << *gtransform_
              << " tlated " << *gtransform
              << " gpuoffset " << m_gpuoffset 
              << std::endl 
              ;
    */
    return m_csgdata->addUniqueTransform( gtransform );   // add to m_gtransforms

}


void NCSG::check()
{
    check_r( m_root );

    m_csgdata->dump_gtransforms(); 

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


void NCSG::export_()
{
    m_csgdata->prepareForExport() ;  //  create node buffer 
    NPY<float>* _nodes = m_csgdata->getNodeBuffer() ; 
    assert(_nodes);

    LOG(debug) << "NCSG::export_ "
              << " exporting CSG node tree into nodes buffer "
              << " num_nodes " << getNumNodes()
              << " height " << getHeight()
              ;
    export_idx(); 
    export_r(m_root, 0);
}


void NCSG::export_idx() 
{
    m_csgdata->setIdx( m_index, m_soIdx, m_lvIdx, getHeight() ); 
}


void NCSG::export_r(nnode* node, unsigned idx)
{
    export_node( node, idx) ; 

    if(node->left && node->right)
    {
        export_r(node->left,  2*idx + 1);
        export_r(node->right, 2*idx + 2);
    }  
}

/**
NCSG::export_node
-------------------

export_gtransform
   collects gtransform into the tran buffer and sets gtransform_idx 
   on the node tree 

export_planes
   collects convex polyhedron planes into plan buffer


NB the part writing into part buffer has to be after 
these as idxs get into that buffer

**/



void NCSG::export_node(nnode* node, unsigned idx)
{
    assert(idx < getNumNodes() ); 
    LOG(trace) << "NCSG::export_node"
              << " idx " << idx 
              << node->desc()
              ;

    export_gtransform(node);
    export_planes(node);
  
    // crucial 2-step here, where m_nodes gets totally rewritten
    npart pt = node->part();  // node->type node->gtransform_idx node->complement written into pt 

    pt.check_bb_zero(node->type); 

    NPY<float>* _nodes = m_csgdata->getNodeBuffer(); 
    _nodes->setPart( pt, idx);  // writes 4 quads to buffer
}

/**
NCSG::export_gtransform
-------------------------

On GPU only gtransform_idx on primitives are used
(there being no multiplying up the tree). 
Any gtransform_idx on operator nodes are ignored.

see notes/issues/OKX4Test_partBuffer_difference.rst

using 0 (meaning None) for identity 

**/
void NCSG::export_gtransform(nnode* node)
{
    if(node->is_primitive())
    {
        const nmat4triple* gtransform = node->global_transform();   
        if( gtransform == NULL ) gtransform = nmat4triple::make_identity() ;
        node->gtransform_idx = addUniqueTransform(gtransform) ; // to m_gtransforms

        LOG(error) << " node->gtransform_idx " << node->gtransform_idx ; 

    }
}


void NCSG::dump(const char* msg)
{
    LOG(info) << msg  ; 
    std::cout << brief() << std::endl ; 

    if(!m_root) return ;

    unsigned nsp = getNumSurfacePoints();
    if( nsp > 0)
    {
        nbbox bbsp = bbox_surface_points();
        std::cout << " bbsp " << bbsp.desc() << std::endl ; 
    }

    m_root->dump("NCSG::dump");   

    NParameters* _meta = m_csgdata->getMetaParameters(-1) ;
    if(_meta) _meta->dump(); 

}

void NCSG::dump_surface_points(const char* msg, unsigned dmax) const 
{
    if(!m_points) return ;
    m_points->dump(msg, dmax );
}





std::string NCSG::brief() const 
{
    std::stringstream ss ; 
    ss << " NCSG " 
       << " ix " << std::setw(4) << m_index
       << " surfpoints " << std::setw(4) << getNumSurfacePoints() 
       << " so " << std::setw(40) << std::left << soname()
       << " lv " << std::setw(40) << std::left << lvname()
       ;

    return ss.str();  
}

std::string NCSG::desc()
{
    std::stringstream ss ; 
    ss << "NCSG " 
       << " index " << m_index
       << " treedir " << ( m_treedir ? m_treedir : "NULL" ) 
       << " boundary " << ( m_boundary ? m_boundary : "NULL" ) 
       << m_csgdata->desc()
       ;
    return ss.str();  
}

 


NCSG* NCSG::LoadCSG(const char* treedir, const char* gltfconfig)
{
    if(!NCSGData::Exists(treedir))
    {
         LOG(warning) << "NCSG::LoadCSG no such dir OR does not contain tree " << treedir ;
         return NULL ; 
    }
    NSceneConfig* config = new NSceneConfig(gltfconfig) ; 

    int verbosity = SSys::getenvint("VERBOSITY", 1) ;
    if(verbosity != config->verbosity) 
    {
        LOG(info) << "NCSG::LoadCSG"
                  << " setting verbosity from envvar " 
                  << verbosity 
                  ;   
         config->verbosity = verbosity ; 
    }


    NCSG* csg = NCSG::LoadTree(treedir, config);
    assert(csg);
    return csg ; 
}


NCSG* NCSG::LoadTree(const char* treedir, const NSceneConfig* config  )
{
    if(!NCSGData::Exists(treedir) )
    {
         LOG(warning) << "NCSG::LoadTree no such treedir OR does not contain tree " 
                      << " treedir: " << treedir 
                      ;

         return NULL ; 
    }
 

    NCSG* tree = new NCSG(treedir) ; 

    tree->setConfig(config);
    tree->setVerbosity(config->verbosity);
    tree->setIsUsedGlobally(true);

    tree->loadsrc();  // populate the src* buffers 
    tree->import();   // complete binary tree m_nodes buffer -> node tree
    tree->export_();  // node tree -> complete binary tree m_nodes buffer

    if(config->verbosity > 1) tree->dump("NCSG::LoadTree");

    if(config->polygonize)
    {
       tree->polygonize();
    }

    tree->collect_surface_points();

    return tree ; 

}


NCSG* NCSG::FromNode(nnode* root, const NSceneConfig* config, unsigned soIdx, unsigned lvIdx )
{
    nnode::Set_parent_links_r(root, NULL);

    root->set_treeidx(lvIdx) ;  // without this no nudging is done

    NCSG* tree = new NCSG(root);

    tree->setConfig(config);
    tree->setSOIdx(soIdx); 
    tree->setLVIdx(lvIdx); 

    /*
    LOG(error) 
               << " soIdx " << std::setw(3) << soIdx  
               << " lvIdx " << std::setw(3) << lvIdx  
               << " "  
               << tree->get_nudger()->brief() 
               ;
    */

    tree->export_();        // node tree -> complete binary tree m_nodes buffer
    assert( tree->getGTransformBuffer() );

    tree->collect_surface_points();

    return tree ; 
}
 

NTrianglesNPY* NCSG::polygonize()
{
    if(m_tris == NULL)
    {
        if(m_verbosity > 1)
        {
        LOG(info) << "NCSG::polygonize START"
                  << " verbosity " << m_verbosity 
                  << " treedir " << m_treedir
                  ; 
        }

        NPolygonizer pg(this);
        m_tris = pg.polygonize();

        if(m_verbosity > 1)
        {
        LOG(info) << "NCSG::polygonize DONE" 
                  << " verbosity " << m_verbosity 
                  << " treedir " << m_treedir
                  ; 
        }

    }
    return m_tris ; 
}

NTrianglesNPY* NCSG::getTris() const 
{
    return m_tris ; 
}

unsigned NCSG::getNumTriangles() const 
{
    return m_tris ? m_tris->getNumTriangles() : 0 ; 
}



glm::uvec4 NCSG::collect_surface_points() 
{
    //LOG(info) << "NCSG::collect_surface_points START " << brief() ; 

    if(!m_points) 
    {
        //LOG(info) << "NCSG::collect_surface_points points ctor " ; 
        m_points = new NNodePoints(m_root, m_config );
    } 

    glm::uvec4 tots = m_points->collect_surface_points();


    //LOG(info) << "NCSG::collect_surface_points DONE " << m_points->desc()  ; 
    return tots ; 
}





nbbox NCSG::bbox_analytic() const 
{
    assert(m_root);
    return m_root->bbox();
}

nbbox NCSG::bbox_surface_points() const 
{
    assert(m_points); 
    return m_points->bbox_surface_points();
}


const std::vector<glm::vec3>& NCSG::getSurfacePoints() const 
{
    assert(m_points); 
    return m_points->getCompositePoints();
}
unsigned NCSG::getNumSurfacePoints() const 
{
    return m_points ? m_points->getNumCompositePoints() : 0 ;
}
float NCSG::getSurfaceEpsilon() const 
{
    return m_points ? m_points->getEpsilon() : -1.f ;
}





void NCSG::adjustToFit( const nbbox& container, float scale, float delta ) const 
{
    LOG(debug) << "NCSG::adjustToFit START " ; 

    nnode* root = getRoot();

    nbbox root_bb = root->bbox();
 
    nnode::AdjustToFit(root, container, scale, delta );         

    LOG(debug) << "NCSG::updateContainer DONE"
              << " root_bb " << root_bb.desc()
              << " container " << container.desc()
              ;
}


void NCSG::setSOIdx(unsigned soIdx)
{
    m_soIdx = soIdx ; 
}
void NCSG::setLVIdx(unsigned lvIdx)
{
    m_lvIdx = lvIdx ; 
}
unsigned NCSG::getSOIdx() const 
{
    return m_soIdx ; 
}
unsigned NCSG::getLVIdx() const 
{
    return m_lvIdx ; 
}







unsigned NCSG::get_num_coincidence() const 
{
   assert(m_nudger);
   return m_nudger->get_num_coincidence() ; 
}
std::string NCSG::desc_coincidence() const 
{
   assert(m_nudger);
   return m_nudger->desc_coincidence() ; 
}

std::string NCSG::get_type_mask_string() const 
{
   assert(m_root);
   return m_root->get_type_mask_string() ;
}
unsigned NCSG::get_type_mask() const 
{
   assert(m_root);
   return m_root->get_type_mask() ;
}
unsigned NCSG::get_oper_mask() const 
{
   assert(m_root);
   return m_root->get_oper_mask() ;
}
unsigned NCSG::get_prim_mask() const 
{
   assert(m_root);
   return m_root->get_prim_mask() ;
}


void NCSG::postimport_autoscan()
{

   float mmstep = 0.1f ; 
    NScan scan(*m_root, m_verbosity);
    unsigned nzero = scan.autoscan(mmstep);
    const std::string& msg = scan.get_message();

    bool even_crossings = nzero % 2 == 0 ; 

    if( !msg.empty() )
    {
        LOG(warning) << "NCSG::postimport_autoscan"
                     << " autoscan message " << msg 
                     ;
    }


    if( !even_crossings )
    {
        LOG(warning) << "NCSG::postimport_autoscan"
                     << " autoscan odd crossings "
                     << " nzero " << nzero 
                     ;
    }

    std::cout 
         << "NCSG::postimport_autoscan"
         << " nzero " << std::setw(4) << nzero 
         << " NScanTest " << std::left << std::setw(40) << getTreeDir()  << std::right
         << " soname " << std::setw(40) << soname()  
         << " tag " << std::setw(10) << m_root->tag()
         << " nprim " << std::setw(4) << m_root->get_num_prim()
         << " typ " << std::setw(20)  << m_root->get_type_mask_string()
         << " msg " << scan.get_message()
         << std::endl 
         ;
}


