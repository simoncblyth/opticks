#include <map>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <climits>


#include "BStr.hh"
#include "OpticksCSG.h"

// npy-

#include "NBBox.hpp"
#include "NGLMExt.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NPart.hpp"
#include "NCSG.hpp"
#include "NQuad.hpp"
#include "NNode.hpp"
#include "NPlane.hpp"
#include "GLMFormat.hpp"

#include "GVector.hh"
#include "GItemList.hh"
#include "GBndLib.hh"
#include "GParts.hh"
#include "GMatrix.hh"

#include "PLOG.hh"


const char* GParts::CONTAINING_MATERIAL = "CONTAINING_MATERIAL" ;  
const char* GParts::SENSOR_SURFACE = "SENSOR_SURFACE" ;  

GParts* GParts::combine(GParts* onesub, unsigned verbosity)
{
    // for consistency: need to combine even when only one sub
    std::vector<GParts*> subs ; 
    subs.push_back(onesub); 
    return GParts::combine(subs, verbosity );
}

GParts* GParts::combine(std::vector<GParts*> subs, unsigned verbosity)
{
    // Concatenate vector of GParts instances into a single GParts instance
    if(verbosity > 1)
    LOG(info) << "GParts::combine " << subs.size() ; 

    GParts* parts = new GParts(); 

    GBndLib* bndlib = NULL ; 
    unsigned analytic_version = 0 ;  
    OpticksCSG_t primflag = CSG_ZERO ; 


    for(unsigned int i=0 ; i < subs.size() ; i++)
    {
        GParts* sp = subs[i];

        OpticksCSG_t pf = sp->getPrimFlag();


        if(primflag == CSG_ZERO) 
            primflag = pf ; 
        else
            assert(pf == primflag && "GParts::combine requires all GParts instances to have the same primFlag (either CSG_FLAGNODETREE or legacy CSG_FLAGPARTLIST)" );


        unsigned av = sp->getAnalyticVersion();

        if(analytic_version == 0)
            analytic_version = av ;
        else
            assert(av == analytic_version && "GParts::combine requires all GParts instances to have the same analytic_version " );   


        parts->add(sp, verbosity );

        if(!bndlib) bndlib = sp->getBndLib(); 
    } 

    if(bndlib) parts->setBndLib(bndlib);
    parts->setAnalyticVersion(analytic_version);
    parts->setPrimFlag(primflag);

    return parts ; 
}


GParts* GParts::make(const npart& pt, const char* spec)
{
    // Serialize the npart shape (BOX, SPHERE or PRISM) into a (1,4,4) parts buffer. 
    // Then instanciate a GParts instance to hold the parts buffer 
    // together with the boundary spec.

    NPY<float>* partBuf = NPY<float>::make(1, NJ, NK );
    partBuf->zero();

    NPY<float>* tranBuf = NPY<float>::make(0, NTRAN, 4, 4 );
    tranBuf->zero();

    NPY<float>* planBuf = NPY<float>::make(0, 4 );
    planBuf->zero();


    partBuf->setPart( pt, 0u );

    GParts* gpt = new GParts(partBuf,tranBuf,planBuf,spec) ;

    unsigned typecode = gpt->getTypeCode(0u);
    assert(typecode == CSG_BOX || typecode == CSG_SPHERE || typecode == CSG_PRISM);

    return gpt ; 
}

GParts* GParts::make(OpticksCSG_t csgflag, glm::vec4& param, const char* spec)
{
    float size = param.w ;  
    //-------   
    // FIX: this is wrong for most solids 
    //gbbox bb(gfloat3(-size), gfloat3(size));  
     
    nbbox bb = make_bbox(glm::vec3(-size), glm::vec3(size));  


    if(csgflag == CSG_ZSPHERE)
    {
        assert( 0 && "TODO: geometry specifics should live in nzsphere etc.. not here " );
        bb.min.z = param.x*param.w ; 
        bb.max.z = param.y*param.w ; 
    } 
    //-------

    NPY<float>* partBuf = NPY<float>::make(1, NJ, NK );
    partBuf->zero();

    NPY<float>* tranBuf = NPY<float>::make(0, NTRAN, 4, 4 );
    tranBuf->zero();

    NPY<float>* planBuf = NPY<float>::make(0, 4);
    planBuf->zero();



    assert(BBMIN_K == 0 );
    assert(BBMAX_K == 0 );

    unsigned int i = 0u ; 
    partBuf->setQuad( i, PARAM_J, param.x, param.y, param.z, param.w );
    partBuf->setQuad( i, BBMIN_J, bb.min.x, bb.min.y, bb.min.z , 0.f );
    partBuf->setQuad( i, BBMAX_J, bb.max.x, bb.max.y, bb.max.z , 0.f );

    // TODO: go via an npart instance

    GParts* pt = new GParts(partBuf, tranBuf, planBuf, spec) ;
    pt->setTypeCode(0u, csgflag);

    return pt ; 
} 

const int GParts::NTRAN = 3 ; 


GParts* GParts::make( NCSG* tree, const char* spec, unsigned verbosity )
{
    assert(spec);

    bool usedglobally = tree->isUsedGlobally() ;   // see opticks/notes/issues/subtree_instances_missing_transform.rst
    //bool usedglobally = true ; 

    NPY<float>* tree_tranbuf = tree->getGTransformBuffer() ;
    NPY<float>* tree_planbuf = tree->getPlaneBuffer() ;
    assert( tree_tranbuf );

    NPY<float>* nodebuf = tree->getNodeBuffer();       // serialized binary tree
    NPY<float>* tranbuf = usedglobally                 ? tree_tranbuf->clone() : tree_tranbuf ; 
    NPY<float>* planbuf = usedglobally && tree_planbuf ? tree_planbuf->clone() : tree_planbuf ;  

    // if any convexpolyhedron eg Trapezoids are usedglobally (ie non-instanced), will need:
    //
    //   1. clone the PlaneBuffer above
    //   2. transform the planes with the global transform, do this in applyPlacementTransform 
    //

    if(verbosity > 1)
    LOG(info) << "GParts::make(NCSG)"
              << " tree " << std::setw(5) << tree->getIndex()
              << " usedglobally " << std::setw(1) << usedglobally 
              << " nodebuf " << ( nodebuf ? nodebuf->getShapeString() : "NULL" ) 
              << " tranbuf " << ( tranbuf ? tranbuf->getShapeString() : "NULL" ) 
              << " planbuf " << ( planbuf ? planbuf->getShapeString() : "NULL" ) 
              ;

    if(!tranbuf) 
    {
       LOG(fatal) << "GParts::make NO GTransformBuffer " ; 
       assert(0);

       tranbuf = NPY<float>::make(0,NTRAN,4,4) ;
       tranbuf->zero();
    } 
    assert( tranbuf && tranbuf->hasShape(-1,NTRAN,4,4));

    if(!planbuf) 
    {
       planbuf = NPY<float>::make(0,4) ;
       planbuf->zero();
    } 
    assert( planbuf && planbuf->hasShape(-1,4));

    nnode* root = tree->getRoot(); 
    // hmm maybe should not use the nnode ? ie operate fully from the persistable buffers ?

    assert(nodebuf && root) ; 

    unsigned ni = nodebuf->getShape(0);
    assert( nodebuf->hasItemShape(NJ, NK) && ni > 0 );

    bool type_ok = root && root->type < CSG_UNDEFINED ;
    if(!type_ok)
        LOG(fatal) << "GParts::make"
                   << " bad type " << root->type
                   << " name " << CSGName(root->type) 
                   << " YOU MAY JUST NEED TO RECOMPILE " 
                   ;

    assert(type_ok);

    LOG(debug) << "GParts::make NCSG "
              << " treedir " << tree->getTreeDir()
              << " node_sh " << nodebuf->getShapeString()
              << " tran_sh " << tranbuf->getShapeString() 
              << " spec " << spec 
              << " type " << root->csgname()
              ; 

    // GParts originally intended to handle lists of parts each of which 
    // must have an associated boundary spec. When holding CSG trees there 
    // is really only a need for a single common boundary, but for
    // now enable reuse of the old GParts by duplicating the spec 
    // for every node of the tree

    const char* reldir = "" ;  // empty reldir avoids defaulting to GItemList  

    GItemList* lspec = GItemList::Repeat("GParts", spec, ni, reldir) ; 

    GParts* pts = new GParts(nodebuf, tranbuf, planbuf, lspec) ;

    //pts->setTypeCode(0u, root->type);   //no need, slot 0 is the root node where the type came from
    return pts ; 
}

GParts::GParts(GBndLib* bndlib) 
      :
      m_part_buffer(NPY<float>::make(0, NJ, NK )),
      m_tran_buffer(NPY<float>::make(0, NTRAN, 4, 4 )),
      m_plan_buffer(NPY<float>::make(0, 4)),
      m_bndspec(new GItemList("GParts","")),   // empty reldir allows GParts.txt to be written directly at eg GPartsAnalytic/0/GParts.txt
      m_bndlib(bndlib),
      m_name(NULL),
      m_prim_buffer(NULL),
      m_closed(false),
      m_loaded(false),
      m_verbosity(0),
      m_analytic_version(0),
      m_primflag(CSG_FLAGNODETREE),
      m_medium(NULL)
{
      m_part_buffer->zero();
      m_tran_buffer->zero();
      m_plan_buffer->zero();
 
      init() ; 
}
GParts::GParts(NPY<float>* partBuf,  NPY<float>* tranBuf, NPY<float>* planBuf, const char* spec, GBndLib* bndlib) 
      :
      m_part_buffer(partBuf ? partBuf : NPY<float>::make(0, NJ, NK )),
      m_tran_buffer(tranBuf ? tranBuf : NPY<float>::make(0, NTRAN, 4, 4 )),
      m_plan_buffer(planBuf ? planBuf : NPY<float>::make(0, 4)),
      m_bndspec(new GItemList("GParts","")),   // empty reldir allows GParts.txt to be written directly at eg GPartsAnalytic/0/GParts.txt
      m_bndlib(bndlib),
      m_name(NULL),
      m_prim_buffer(NULL),
      m_closed(false),
      m_loaded(false),
      m_verbosity(0),
      m_analytic_version(0),
      m_primflag(CSG_FLAGNODETREE),
      m_medium(NULL)
{
      m_bndspec->add(spec);
      init() ; 
}
GParts::GParts(NPY<float>* partBuf,  NPY<float>* tranBuf, NPY<float>* planBuf, GItemList* spec, GBndLib* bndlib) 
      :
      m_part_buffer(partBuf ? partBuf : NPY<float>::make(0, NJ, NK )),
      m_tran_buffer(tranBuf ? tranBuf : NPY<float>::make(0, NTRAN, 4, 4 )),
      m_plan_buffer(planBuf ? planBuf : NPY<float>::make(0, 4)),
      m_bndspec(spec),
      m_bndlib(bndlib),
      m_name(NULL),
      m_prim_buffer(NULL),
      m_closed(false),
      m_loaded(false),
      m_verbosity(0),
      m_analytic_version(0),
      m_primflag(CSG_FLAGNODETREE),
      m_medium(NULL)
{
     
      const std::string& reldir = spec->getRelDir() ;
      bool empty_rel = reldir.empty() ;

      //bool gpmt_0 = strcmp(reldir.c_str(), "GPmt/0")== 0 ; 

      bool is_gpmt = !empty_rel && reldir.find("GPmt/") == 0 ; 

      if(is_gpmt)
      {
          LOG(info) << "GParts::GParts detected is_gpmt " << reldir ; 
      } 
      else
      {
          if(!empty_rel)
             LOG(warning) << "GParts::GParts"
                          << " EXPECTING EMPTY RelDir FOR NON GPmt GParts [" << reldir << "]"
                          ;
          assert( empty_rel );
          //  WHY ?
          //  RELDIR IS GItemList ctor argument which 
          //  GPmt::loadFromCache plants the relative PmtPath in ? 
      }
 
      init() ; 
}


void GParts::init()
{
    unsigned npart = m_part_buffer ? m_part_buffer->getNumItems() : 0 ;
    unsigned nspec = m_bndspec ? m_bndspec->getNumItems() : 0  ;

    bool match = npart == nspec ; 
    if(!match) 
    LOG(fatal) << "GParts::init"
               << " parts/spec MISMATCH "  
               << " npart " << npart 
               << " nspec " << nspec
               ;

    assert(match);

}





void GParts::setPrimFlag(OpticksCSG_t primflag)
{
    assert(primflag == CSG_FLAGNODETREE || primflag == CSG_FLAGPARTLIST || primflag == CSG_FLAGINVISIBLE );
    m_primflag = primflag ; 
}
bool GParts::isPartList()  // LEGACY ANALYTIC, NOT LONG TO LIVE ? ACTUALLY ITS FASTER SO BETTER TO KEEP ALIVE
{
    return m_primflag == CSG_FLAGPARTLIST ;
}
bool GParts::isNodeTree()  // ALMOST ALWAYS THIS ONE NOWADAYS
{
    return m_primflag == CSG_FLAGNODETREE ;
}
bool GParts::isInvisible()
{
    return m_primflag == CSG_FLAGINVISIBLE ;
}

const char* GParts::getPrimFlagString() const 
{
    return CSGName(m_primflag); 
}


void GParts::setInvisible()
{
    setPrimFlag(CSG_FLAGINVISIBLE);
}
void GParts::setPartList()
{
    setPrimFlag(CSG_FLAGPARTLIST);
}
void GParts::setNodeTree()
{
    setPrimFlag(CSG_FLAGNODETREE);
}







OpticksCSG_t GParts::getPrimFlag()
{
    return m_primflag ;
}

void GParts::BufferTags(std::vector<std::string>& tags) // static
{
    tags.push_back("part");
    tags.push_back("tran");
    tags.push_back("plan");
   // tags.push_back("prim");
}

const char* GParts::BufferName(const char* tag) // static
{
    return BStr::concat(tag, "Buffer.npy", NULL) ;
}

void GParts::save(const char* dir)
{
    if(!dir) return ; 

    // resource organization handled by GGeoLib, that invokes this

    LOG(info) << "GParts::save dir " << dir ; 

    if(!isClosed())
    {
        LOG(info) << "GParts::save pre-save closing, for primBuf   " ; 
        close();
    }    

    std::vector<std::string> tags ; 
    BufferTags(tags);

    for(unsigned i=0 ; i < tags.size() ; i++)
    {
        const char* tag = tags[i].c_str();
        const char* name = BufferName(tag);
        NPY<float>* buf = getBuffer(tag);
        if(buf)
        {
            unsigned num_items = buf->getShape(0);
            if(num_items > 0)
            { 
                buf->save(dir, name);     
            }
        }
    } 
    if(m_prim_buffer) m_prim_buffer->save(dir, BufferName("prim"));    

    if(m_bndspec) m_bndspec->save(dir); 
}


NPY<float>* GParts::LoadBuffer(const char* dir, const char* tag) // static
{
    const char* name = BufferName(tag) ;
    bool quietly = true ; 
    NPY<float>* buf = NPY<float>::load(dir, name, quietly ) ;
    return buf ; 
}

GParts* GParts::Load(const char* dir) // static
{
    LOG(debug) << "GParts::Load dir " << dir ; 

    NPY<float>* partBuf = LoadBuffer(dir, "part");
    NPY<float>* tranBuf = LoadBuffer(dir, "tran");
    NPY<float>* planBuf = LoadBuffer(dir, "plan");

    // hmm what is appropriate for spec and bndlib these ? 
    //
    // bndlib has to be externally set, its a global thing 
    // that is only needed by registerBoundaries
    //
    // spec is internal ... it needs to be saved with the GParts
    //    

    const char* reldir = "" ;   // empty, signally inplace itemlist persisting
    GItemList* bndspec = GItemList::load(dir, "GParts", reldir ) ; 
    GBndLib*  bndlib = NULL ; 
    GParts* parts = new GParts(partBuf,  tranBuf, planBuf, bndspec, bndlib) ;
    
    NPY<int>* primBuf = NPY<int>::load(dir, BufferName("prim") );
    parts->setPrimBuffer(primBuf);
    parts->setLoaded();

    return parts  ; 
}



void GParts::setName(const char* name)
{
    m_name = name ? strdup(name) : NULL  ; 
}
const char* GParts::getName()
{
    return m_name ; 
}

void GParts::setVerbosity(unsigned verbosity)
{
    m_verbosity = verbosity ; 
}




unsigned GParts::getAnalyticVersion()
{
    return m_analytic_version ; 
}
void GParts::setAnalyticVersion(unsigned version)
{
    m_analytic_version = version ; 
}
 



bool GParts::isClosed()
{
    return m_closed ; 
}
bool GParts::isLoaded()
{
    return m_loaded ; 
}

void GParts::setLoaded(bool loaded)
{
    m_loaded = loaded ; 
}



unsigned int GParts::getPrimNumParts(unsigned int prim_index)
{
   // DOES NOT WORK POSTCACHE
    return m_parts_per_prim.count(prim_index)==1 ? m_parts_per_prim[prim_index] : 0 ; 
}


void GParts::setBndSpec(GItemList* bndspec)
{
    m_bndspec = bndspec ;
}
GItemList* GParts::getBndSpec()
{
    return m_bndspec ; 
}
void GParts::setBndLib(GBndLib* bndlib)
{
    m_bndlib = bndlib ; 
}
GBndLib* GParts::getBndLib()
{
    return m_bndlib ; 
}



void GParts::setPrimBuffer(NPY<int>* buf )
{
    m_prim_buffer = buf ; 
}
void GParts::setPartBuffer(NPY<float>* buf )
{
    m_part_buffer = buf ; 
}
void GParts::setTranBuffer(NPY<float>* buf)
{
    m_tran_buffer = buf ; 
}
void GParts::setPlanBuffer(NPY<float>* buf)
{
    m_plan_buffer = buf ; 
}


NPY<int>* GParts::getPrimBuffer()
{
    return m_prim_buffer ; 
}
NPY<float>* GParts::getPartBuffer()
{
    return m_part_buffer ; 
}
NPY<float>* GParts::getTranBuffer()
{
    return m_tran_buffer ; 
}
NPY<float>* GParts::getPlanBuffer()
{
    return m_plan_buffer ; 
}

NPY<float>* GParts::getBuffer(const char* tag) const 
{
    if(strcmp(tag,"part")==0) return m_part_buffer ; 
    if(strcmp(tag,"tran")==0) return m_tran_buffer ; 
    if(strcmp(tag,"plan")==0) return m_plan_buffer ; 
 //   if(strcmp(tag,"prim")==0) return m_prim_buffer ; 
    return NULL ; 
}





unsigned int GParts::getNumParts()
{
    // for combo GParts this is total of all prim
    if(!m_part_buffer)
    {
        LOG(error) << "GParts::getNumParts NULL part_buffer" ; 
        return 0 ; 
    }

    assert(m_part_buffer->getNumItems() == m_bndspec->getNumItems() );
    return m_part_buffer->getNumItems() ;
}


void GParts::applyPlacementTransform(GMatrix<float>* gtransform, unsigned verbosity )
{
   // gets invoked from GMergedMesh::mergeVolumeAnalytic

    const float* data = static_cast<float*>(gtransform->getPointer());

    if(verbosity > 2)
    nmat4triple::dump(data, "GParts::applyPlacementTransform gtransform:" ); 

    glm::mat4 placement = glm::make_mat4( data ) ;  

    assert(m_tran_buffer->hasShape(-1,3,4,4));

    unsigned ni = m_tran_buffer->getNumItems();

    if(verbosity > 2)
    LOG(info) << "GParts::applyPlacementTransform"
              << " tran_buffer " << m_tran_buffer->getShapeString()
              << " ni " << ni
              ;


    bool reversed = true ; // means apply transform at root end, not leaf end 

    if(verbosity > 2)
    nmat4triple::dump(m_tran_buffer,"GParts::applyPlacementTransform before");

    for(unsigned i=0 ; i < ni ; i++)
    {
        nmat4triple* tvq = m_tran_buffer->getMat4TriplePtr(i) ;

        const nmat4triple* ntvq = nmat4triple::make_transformed( tvq, placement, reversed, "GParts::applyPlacementTransform" );
                              //  ^^^^^^^^^^^^^^^^^^^^^^^ SUSPECT DOUBLE NEGATIVE RE REVERSED  ^^^^^^^

        m_tran_buffer->setMat4Triple( ntvq, i ); 
    }

    if(verbosity > 2)
    nmat4triple::dump(m_tran_buffer,"GParts::applyPlacementTransform after");


    assert(m_plan_buffer->hasShape(-1,4));
    unsigned num_plane = m_plan_buffer->getNumItems();

    // Formerly all geometry that required planes (eg trapezoids) 
    // was part of instanced solids... so this was not needed.
    // BUT for debugging it is useful to be able to operate in global mode
    // whilst testing small subsets of geometry
    //
    //if(num_plane > 0 ) assert(0 && "plane placement not implemented" );

    if(num_plane > 0) 
    {
        if(verbosity > 3)
        m_plan_buffer->dump("planes_before_transform");

        nglmext::transform_planes( m_plan_buffer, placement );

        if(verbosity > 3)
        m_plan_buffer->dump("planes_after_transform");
    }

}


void GParts::add(GParts* other, unsigned verbosity )
{
    if(getBndLib() == NULL)
    {
        setBndLib(other->getBndLib()); 
    }
    else
    {
        assert(getBndLib() == other->getBndLib());
    }

    unsigned int n0 = getNumParts(); // before adding

    m_bndspec->add(other->getBndSpec());

    NPY<float>* other_part_buffer = other->getPartBuffer() ;
    NPY<float>* other_tran_buffer = other->getTranBuffer() ;
    NPY<float>* other_plan_buffer = other->getPlanBuffer() ;

    m_part_buffer->add(other_part_buffer);
    m_tran_buffer->add(other_tran_buffer);
    m_plan_buffer->add(other_plan_buffer);

    unsigned num_part_add = other_part_buffer->getNumItems() ;
    unsigned num_tran_add = other_tran_buffer->getNumItems() ;
    unsigned num_plan_add = other_plan_buffer->getNumItems() ;

    m_part_per_add.push_back(num_part_add); 
    m_tran_per_add.push_back(num_tran_add);
    m_plan_per_add.push_back(num_plan_add);

    unsigned int n1 = getNumParts(); // after adding

    for(unsigned int p=n0 ; p < n1 ; p++)  // update indices for parts added
    {
        setIndex(p, p);
    }



    if(verbosity > 2)
    LOG(info) 
              << " n0 " << std::setw(3) << n0  
              << " n1 " << std::setw(3) << n1
              << " num_part_add " << std::setw(3) <<  num_part_add
              << " num_tran_add " << std::setw(3) << num_tran_add
              << " num_plan_add " << std::setw(3) << num_plan_add
              << " other_part_buffer  " << other_part_buffer->getShapeString()
              << " other_tran_buffer  " << other_tran_buffer->getShapeString()
              << " other_plan_buffer  " << other_plan_buffer->getShapeString()
              ;  
    
}

void GParts::setContainingMaterial(const char* material)
{
    // for flexibility persisted GParts should leave the outer containing material
    // set to a default marker name such as "CONTAINING_MATERIAL", 
    // to allow the GParts to be placed within other geometry

    if(m_medium)
       LOG(fatal) << "setContainingMaterial called already " << m_medium 
       ;

    assert( m_medium == NULL && "GParts::setContainingMaterial WAS CALLED ALREADY " );
    m_medium = strdup(material); 

    unsigned field    = 0 ; 
    const char* from  = GParts::CONTAINING_MATERIAL ;
    const char* to    = material  ;
    const char* delim = "/" ;     

    m_bndspec->replaceField(field, from, to, delim );

    // all field zero *from* occurences are replaced with *to* 
}

void GParts::setSensorSurface(const char* surface)
{
    m_bndspec->replaceField(1, GParts::SENSOR_SURFACE, surface ) ; 
    m_bndspec->replaceField(2, GParts::SENSOR_SURFACE, surface ) ; 
}


void GParts::close()
{
    if(isClosed()) LOG(fatal) << "closed already " ;
    assert(!isClosed()); 
    m_closed = true ; 


    //if(m_verbosity > 1)
    LOG(info) << "GParts::close START "
              << " verbosity " << m_verbosity 
               ; 

    registerBoundaries();

    if(!m_loaded)
    {
        makePrimBuffer(); 
    }

    if(m_verbosity > 1)
    dumpPrimBuffer(); 

    //if(m_verbosity > 1)
    LOG(info) << "GParts::close DONE " 
              << " verbosity " << m_verbosity 
              ; 
}

void GParts::registerBoundaries() // convert boundary spec names into integer codes using bndlib
{
   assert(m_bndlib); 
   unsigned int nbnd = m_bndspec->getNumKeys() ; 
   assert( getNumParts() == nbnd );

   if(m_verbosity > 0)
   LOG(info) << "GParts::registerBoundaries " 
             << " verbosity " << m_verbosity
             << " nbnd " << nbnd 
             << " NumParts " << getNumParts() 
             ;

   for(unsigned int i=0 ; i < nbnd ; i++)
   {
       const char* spec = m_bndspec->getKey(i);
       unsigned int boundary = m_bndlib->addBoundary(spec);
       setBoundary(i, boundary);

       if(m_verbosity > 1)
       LOG(info) << "GParts::registerBoundaries " 
                 << " i " << std::setw(3) << i 
                 << " " << std::setw(30) << spec
                 << " --> "
                 << std::setw(4) << boundary 
                 << " " << std::setw(30) << m_bndlib->shortname(boundary)
                 ;

   } 
}


void GParts::reconstructPartsPerPrim()
{
/*
The "classic" partlist formed in python with opticks/ana/pmt/analytic.py  (pmt-ecd)
uses the nodeindex entry in the partlist buffer to identify which parts 
correspond to each solid eg PYREX,VACUUM,CATHODE,BOTTOM,DYNODE. 

Hence by counting parts keyed by the nodeIndex the below reconstructs 
the number of parts for each primitive.

In orther words "parts" are associated to their containing "prim" 
via the nodeIndex property.

For the CSG nodeTree things are simpler as each NCSG tree 
directly corresponds to a 1 GVolume and 1 GParts that
are added separtately, see GGeoTest::loadCSG.
*/

    assert(isPartList());
    m_parts_per_prim.clear();

    unsigned int nmin(INT_MAX) ; 
    unsigned int nmax(0) ; 

    unsigned numParts = getNumParts() ; 

    LOG(info) << "GParts::reconstructPartsPerPrim"
              << " numParts " << numParts 
              ;
 
    // count parts for each nodeindex
    for(unsigned int i=0; i < numParts ; i++)
    {
        unsigned int nodeIndex = getNodeIndex(i);
        unsigned typ = getTypeCode(i);
        std::string  typName = CSGName((OpticksCSG_t)typ);
 
        LOG(info) << "GParts::makePrimBuffer"
                   << " i " << std::setw(3) << i  
                   << " nodeIndex " << std::setw(3) << nodeIndex
                   << " typ " << std::setw(3) << typ 
                   << " typName " << typName 
                   ;  
                     
        m_parts_per_prim[nodeIndex] += 1 ; 

        if(nodeIndex < nmin) nmin = nodeIndex ; 
        if(nodeIndex > nmax) nmax = nodeIndex ; 
    }

    unsigned int num_prim = m_parts_per_prim.size() ;

    //assert(nmax - nmin == num_solids - 1);  // expect contiguous node indices
    if(nmax - nmin != num_prim - 1)
    {
        LOG(warning) << "GParts::reconstructPartsPerPrim  non-contiguous node indices"
                     << " nmin " << nmin 
                     << " nmax " << nmax
                     << " num_prim " << num_prim
                     << " part_per_add.size " << m_part_per_add.size()
                     << " tran_per_add.size " << m_tran_per_add.size()
                     ; 
    }
}



void GParts::makePrimBuffer()
{
    /*
    Derives prim buffer from the part buffer
    
    Primbuffer acts as an "index" providing cross
    referencing associating a primitive via
    offsets to the parts/nodes, transforms and planes
    relevant to the primitive.

    prim/part/tran/plan buffers are used GPU side in cu/intersect_analytic.cu.
    

    Hmm looks impossible to do this prim Buffer derivation
    postcache, as it is relying on the offsets collected 
    at each concatentation. So will need to load the primBuf 

    */

    unsigned int num_prim = 0 ; 

    if(m_verbosity > 0)
    LOG(info) << "GParts::makePrimBuffer"
              << " verbosity " << m_verbosity
              << " isPartList " << isPartList()
              << " isNodeTree " << isNodeTree()
              << " parts_per_prim.size " << m_parts_per_prim.size()
              << " part_per_add.size " << m_part_per_add.size()
              << " tran_per_add.size " << m_tran_per_add.size()
              << " plan_per_add.size " << m_plan_per_add.size()
              ; 

    if(isPartList())
    {
        reconstructPartsPerPrim();
        num_prim = m_parts_per_prim.size() ;
    } 
    else if(isNodeTree() )
    {
        num_prim = m_part_per_add.size() ;
        assert( m_part_per_add.size() == num_prim );
        assert( m_tran_per_add.size() == num_prim );
        assert( m_plan_per_add.size() == num_prim );
    }
    else
    {
        assert(0);
    }


    if(m_verbosity > 2)
    LOG(info) << "GParts::makePrimBuffer"
              << " verbosity " << m_verbosity
              << " num_prim " << num_prim
              << " parts_per_prim.size " << m_parts_per_prim.size()
              << " part_per_add.size " << m_part_per_add.size()
              << " tran_per_add.size " << m_tran_per_add.size()
              << " plan_per_add.size " << m_plan_per_add.size()
              ; 

    nivec4* priminfo = new nivec4[num_prim] ;

    unsigned part_offset = 0 ; 
    unsigned tran_offset = 0 ; 
    unsigned plan_offset = 0 ; 

    if(isNodeTree())
    {
        unsigned n = 0 ; 
        for(unsigned i=0 ; i < num_prim ; i++)
        {
            unsigned int tran_for_prim = m_tran_per_add[i] ; 
            unsigned int plan_for_prim = m_plan_per_add[i] ; 

            //unsigned int parts_for_prim = m_parts_per_prim[i] ; 
            unsigned int parts_for_prim = m_part_per_add[i] ; 

            nivec4& pri = *(priminfo+n) ;

            pri.x = part_offset ; 
            pri.y = m_primflag == CSG_FLAGPARTLIST ? -parts_for_prim : parts_for_prim ;
            pri.z = tran_offset ; 
            pri.w = plan_offset ; 

            if(m_verbosity > 2)
            LOG(info) << "GParts::makePrimBuffer(nodeTree) priminfo " << pri.desc() ;       

            part_offset += parts_for_prim ; 
            tran_offset += tran_for_prim ; 
            plan_offset += plan_for_prim ; 

            n++ ; 
        }
    }
    else if(isPartList())
    {
        unsigned n = 0 ; 
        typedef std::map<unsigned int, unsigned int> UU ; 
        for(UU::const_iterator it=m_parts_per_prim.begin() ; it != m_parts_per_prim.end() ; it++)
        {
            //unsigned int node_index = it->first ; 
            unsigned int parts_for_prim = it->second ; 

            nivec4& pri = *(priminfo+n) ;

            pri.x = part_offset ; 
            pri.y = m_primflag == CSG_FLAGPARTLIST ? -parts_for_prim : parts_for_prim ;
            pri.z = 0 ; 
            //pri.w = m_primflag ; 
            pri.w = 0 ; 

            if(m_verbosity > 2)
            LOG(info) << "GParts::makePrimBuffer(partList) priminfo " << pri.desc() ;       

            part_offset += parts_for_prim ; 
            n++ ; 
        }
    }


    NPY<int>* buf = NPY<int>::make( num_prim, 4 );
    buf->setData((int*)priminfo);
    delete [] priminfo ; 

    setPrimBuffer(buf);
}



void GParts::dumpPrim(unsigned primIdx)
{
    // following access pattern of oxrap/cu/intersect_analytic.cu::intersect

    NPY<int>*    primBuffer = getPrimBuffer();
    NPY<float>*  partBuffer = getPartBuffer();
    //NPY<float>*  planBuffer = getPlanBuffer();

    if(!primBuffer) return ; 
    if(!partBuffer) return ; 

    glm::ivec4 prim = primBuffer->getQuadI(primIdx) ;

    int partOffset = prim.x ; 
    int numParts_   = prim.y ; 
    int tranOffset = prim.z ; 
    int planOffset = prim.w ; 

    unsigned numParts = abs(numParts_) ;
    unsigned primFlag = numParts_ < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE  ; 

    unsigned num_zeros = 0 ; 
    unsigned num_nonzeros = 0 ; 

    for(unsigned p=0 ; p < numParts ; p++)
    {
        unsigned int partIdx = partOffset + p ;

        nquad q0, q1, q2, q3 ;

        q0.f = partBuffer->getVQuad(partIdx,0);  
        q1.f = partBuffer->getVQuad(partIdx,1);  
        q2.f = partBuffer->getVQuad(partIdx,2);  
        q3.f = partBuffer->getVQuad(partIdx,3);  

        unsigned typecode = q2.u.w ;
        assert(TYPECODE_J == 2 && TYPECODE_K == 3);

        bool iszero = typecode == 0 ; 
        if(iszero) num_zeros++ ; 
        else num_nonzeros++ ; 

        if(!iszero)
        LOG(info) << " p " << std::setw(3) << p 
                  << " partIdx " << std::setw(3) << partIdx
                  << " typecode " << typecode
                  << " CSGName " << CSGName((OpticksCSG_t)typecode)
                  ;

    }

    LOG(info) << " primIdx "    << std::setw(3) << primIdx 
              << " partOffset " << std::setw(3) << partOffset 
              << " tranOffset " << std::setw(3) << tranOffset 
              << " planOffset " << std::setw(3) << planOffset 
              << " numParts_ "  << std::setw(3) << numParts_
              << " numParts "   << std::setw(3) << numParts
              << " num_zeros "   << std::setw(5) << num_zeros
              << " num_nonzeros " << std::setw(5) << num_nonzeros
              << " primFlag "   << std::setw(5) << primFlag 
              << " CSGName "  << CSGName((OpticksCSG_t)primFlag) 
              << " prim "       << gformat(prim)
              ;




}


void GParts::dumpPrimBuffer(const char* msg)
{
    NPY<int>*    primBuffer = getPrimBuffer();
    NPY<float>*  partBuffer = getPartBuffer();
    if(!primBuffer) return ; 
    if(!partBuffer) return ; 

    if(m_verbosity > 2 )
    LOG(info) 
        << msg 
        << " verbosity " << m_verbosity
        << " primBuffer " << primBuffer->getShapeString() 
        << " partBuffer " << partBuffer->getShapeString() 
        ; 

    assert( primBuffer->hasItemShape(4,0) && primBuffer->getNumItems() > 0  );
    assert( partBuffer->hasItemShape(4,4) && partBuffer->getNumItems() > 0 );

    if(m_verbosity > 3)
    {
        for(unsigned primIdx=0 ; primIdx < primBuffer->getNumItems() ; primIdx++) dumpPrim(primIdx);
    }
}


void GParts::dumpPrimInfo(const char* msg, unsigned lim )
{
    unsigned numPrim = getNumPrim() ;
    unsigned ulim = std::min( numPrim, lim ) ; 

    LOG(info) << msg 
              << " (part_offset, parts_for_prim, tran_offset, plan_offset) "
              << " numPrim: " << numPrim 
              << " ulim: " << ulim 
              ;

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        if( ulim != 0 &&  i > ulim && i < numPrim - ulim ) continue ;    

        nivec4 pri = getPrimInfo(i);
        LOG(info) << pri.desc() ;
    }
}


void GParts::Summary(const char* msg, unsigned lim)
{
    LOG(info) << msg 
              << " num_parts " << getNumParts() 
              << " num_prim " << getNumPrim()
              ;
 
    typedef std::map<unsigned int, unsigned int> UU ; 
    for(UU::const_iterator it=m_parts_per_prim.begin() ; it!=m_parts_per_prim.end() ; it++)
    {
        unsigned int prim_index = it->first ; 
        unsigned int nparts = it->second ; 
        unsigned int nparts2 = getPrimNumParts(prim_index) ; 
        printf("%2u : %2u \n", prim_index, nparts );
        assert( nparts == nparts2 );
    }

    unsigned numParts = getNumParts() ;
    unsigned ulim = std::min( numParts, lim ) ; 

    for(unsigned i=0 ; i < numParts ; i++)
    {
        if( ulim != 0 &&  i > ulim && i < numParts - ulim ) continue ; 
   
        std::string bn = getBoundaryName(i);
        printf(" part %2u : node %2u type %2u boundary [%3u] %s  \n", i, getNodeIndex(i), getTypeCode(i), getBoundary(i), bn.c_str() ); 
    }
}



std::string GParts::desc()
{
    std::stringstream ss ; 
    ss 
       << " GParts "
       << " primflag " << std::setw(20) << getPrimFlagString()
       << " numParts " << std::setw(4) << getNumParts()
       << " numPrim " << std::setw(4) << getNumPrim()
       ;

    return ss.str(); 
}


unsigned int GParts::getNumPrim()
{
    return m_prim_buffer ? m_prim_buffer->getShape(0) : 0 ; 
}
const char* GParts::getTypeName(unsigned int part_index)
{
    unsigned int code = getTypeCode(part_index);
    return CSGName((OpticksCSG_t)code);
}
     
float* GParts::getValues(unsigned int i, unsigned int j, unsigned int k)
{
    float* data = m_part_buffer->getValues();
    float* ptr = data + i*NJ*NK+j*NJ+k ;
    return ptr ; 
}
     
gfloat3 GParts::getGfloat3(unsigned int i, unsigned int j, unsigned int k)
{
    float* ptr = getValues(i,j,k);
    return gfloat3( *ptr, *(ptr+1), *(ptr+2) ); 
}

nivec4 GParts::getPrimInfo(unsigned int iprim)
{
    int* data = m_prim_buffer->getValues();
    int* ptr = data + iprim*SK  ;

    nivec4 pri = make_nivec4( *ptr, *(ptr+1), *(ptr+2), *(ptr+3) );
    return pri ;  
}
nbbox GParts::getBBox(unsigned int i)
{
   gfloat3 min = getGfloat3(i, BBMIN_J, BBMIN_K );  
   gfloat3 max = getGfloat3(i, BBMAX_J, BBMAX_K );  
   //gbbox bb(min, max) ; 

   nbbox bb = make_bbox(min.x, min.y, min.z, max.x, max.y, max.z);  
   return bb ; 
}

void GParts::enlargeBBoxAll(float epsilon)
{
   for(unsigned int part=0 ; part < getNumParts() ; part++) enlargeBBox(part, epsilon);
}

void GParts::enlargeBBox(unsigned int part, float epsilon)
{
    float* pmin = getValues(part,BBMIN_J,BBMIN_K);
    float* pmax = getValues(part,BBMAX_J,BBMAX_K);

    glm::vec3 min = glm::make_vec3(pmin) - glm::vec3(epsilon);
    glm::vec3 max = glm::make_vec3(pmax) + glm::vec3(epsilon);
 
    *(pmin+0 ) = min.x ; 
    *(pmin+1 ) = min.y ; 
    *(pmin+2 ) = min.z ; 

    *(pmax+0 ) = max.x ; 
    *(pmax+1 ) = max.y ; 
    *(pmax+2 ) = max.z ; 

    LOG(debug) << "GParts::enlargeBBox"
              << " part " << part 
              << " epsilon " << epsilon
              << " min " << gformat(min) 
              << " max " << gformat(max)
              ; 

}


unsigned int GParts::getUInt(unsigned int i, unsigned int j, unsigned int k)
{
    assert(i < getNumParts() );
    unsigned int l=0u ; 
    return m_part_buffer->getUInt(i,j,k,l);
}
void GParts::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int value)
{
    assert(i < getNumParts() );
    unsigned int l=0u ; 
    m_part_buffer->setUInt(i,j,k,l, value);
}



unsigned int GParts::getNodeIndex(unsigned int part)
{
    return getUInt(part, NODEINDEX_J, NODEINDEX_K);
}
unsigned int GParts::getTypeCode(unsigned int part)
{
    return getUInt(part, TYPECODE_J, TYPECODE_K);
}
unsigned int GParts::getIndex(unsigned int part)
{
    return getUInt(part, INDEX_J, INDEX_K);
}
unsigned int GParts::getBoundary(unsigned int part)
{
    return getUInt(part, BOUNDARY_J, BOUNDARY_K);
}




void GParts::setNodeIndex(unsigned int part, unsigned int nodeindex)
{
    setUInt(part, NODEINDEX_J, NODEINDEX_K, nodeindex);
}
void GParts::setTypeCode(unsigned int part, unsigned int typecode)
{
    setUInt(part, TYPECODE_J, TYPECODE_K, typecode);
}
void GParts::setIndex(unsigned int part, unsigned int index)
{
    setUInt(part, INDEX_J, INDEX_K, index);
}
void GParts::setBoundary(unsigned int part, unsigned int boundary)
{
    setUInt(part, BOUNDARY_J, BOUNDARY_K, boundary);
}





void GParts::setBoundaryAll(unsigned int boundary)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setBoundary(i, boundary);
}
void GParts::setNodeIndexAll(unsigned int nodeindex)
{
    for(unsigned int i=0 ; i < getNumParts() ; i++) setNodeIndex(i, nodeindex);
}


std::string GParts::getBoundaryName(unsigned int part)
{
    unsigned int boundary = getBoundary(part);
    std::string name = m_bndlib ? m_bndlib->shortname(boundary) : "" ;
    return name ;
}



void GParts::fulldump(const char* msg, unsigned lim)
{
    LOG(info) << msg 
              << " lim " << lim 
              ; 

    dump(msg, lim);
    Summary(msg);

    NPY<float>* partBuf = getPartBuffer();
    NPY<int>*   primBuf = getPrimBuffer(); 

    partBuf->dump("partBuf");
    primBuf->dump("primBuf:partOffset/numParts/primIndex/0");
}

void GParts::dump(const char* msg, unsigned lim)
{


    LOG(info) << msg
              << " lim " << lim 
              << " pbuf " << m_part_buffer->getShapeString()
              ; 

    dumpPrimInfo(msg, lim);

    NPY<float>* buf = m_part_buffer ; 
    assert(buf);
    assert(buf->getDimensions() == 3);

    unsigned ni = buf->getShape(0) ;
    unsigned nj = buf->getShape(1) ;
    unsigned nk = buf->getShape(2) ;

    unsigned ulim = std::min( ni, lim ) ; 

    LOG(info) << "GParts::dump"
              << " ni " << ni 
              << " lim " << lim
              << " ulim " << ulim
              ; 

    assert( nj == NJ );
    assert( nk == NK );

    float* data = buf->getValues();

    uif_t uif ; 

    for(unsigned i=0; i < ni; i++)
    {   
       if( ulim != 0 &&  i > ulim && i < ni - ulim ) continue ;    

       unsigned int tc = getTypeCode(i);
       unsigned int id = getIndex(i);
       unsigned int bnd = getBoundary(i);
       std::string  bn = getBoundaryName(i);

       std::string csg = CSGName((OpticksCSG_t)tc);

       const char*  tn = getTypeName(i);

       for(unsigned int j=0 ; j < NJ ; j++)
       {   
          for(unsigned int k=0 ; k < NK ; k++) 
          {   
              uif.f = data[i*NJ*NK+j*NJ+k] ;
              if( j == TYPECODE_J && k == TYPECODE_K )
              {
                  assert( uif.u == tc );
                  printf(" %10u (%s) TYPECODE ", uif.u, tn );
              } 
              else if( j == INDEX_J && k == INDEX_K)
              {
                  assert( uif.u == id );
                  printf(" %6u <-INDEX   " , uif.u );
              }
              else if( j == BOUNDARY_J && k == BOUNDARY_K)
              {
                  assert( uif.u == bnd );
                  printf(" %6u <-bnd  ", uif.u );
              }
              else if( j == NODEINDEX_J && k == NODEINDEX_K)
                  printf(" %10d (nodeIndex) ", uif.i );
              else
                  printf(" %10.4f ", uif.f );
          }   

          if( j == BOUNDARY_J ) printf(" bn %s ", bn.c_str() );
          printf("\n");
       }   
       printf("\n");
    }   
}


