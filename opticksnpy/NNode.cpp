
#include <cstdio>
#include <cassert>
#include <cmath>
#include <sstream>
#include <iomanip>


#include "NGLM.hpp"
#include "NGLMExt.hpp"

#include "NCSG.hpp"
#include "NNode.hpp"
#include "NPart.hpp"
#include "NQuad.hpp"
#include "Nuv.hpp"
#include "NBBox.hpp"

// primitives
#include "NSphere.hpp"
#include "NZSphere.hpp"
#include "NBox.hpp"
#include "NSlab.hpp"
#include "NPlane.hpp"
#include "NCylinder.hpp"
#include "NDisc.hpp"
#include "NCone.hpp"
#include "NConvexPolyhedron.hpp"



#include "PLOG.hh"


const char* nnode::FRAME_MODEL_ = "FRAME_MODEL" ;
const char* nnode::FRAME_LOCAL_ = "FRAME_LOCAL" ;
const char* nnode::FRAME_GLOBAL_ = "FRAME_GLOBAL" ;

const char* nnode::FrameType(NNodeFrameType fr)
{
    const char* s = NULL ;
    switch(fr)
    {
        case FRAME_MODEL: s = FRAME_MODEL_ ; break ; 
        case FRAME_LOCAL: s = FRAME_LOCAL_ ; break ; 
        case FRAME_GLOBAL: s = FRAME_GLOBAL_ ; break ; 
    }
    return s ;
}


float nnode::operator()(float,float,float) const 
{
    assert(0 && "nnode::operator() needs override ");
    return 0.f ; 
} 


float nnode::sdf_(const glm::vec3& , NNodeFrameType ) const 
{
    assert(0 && "nnode::sdf_ needs override ");
    return 0.f ; 
}


glm::vec3 nnode::par_pos_(const nuv& , NNodeFrameType ) const 
{
    assert(0 && "nnode::par_pos_ needs override ");
    return glm::vec3(0);
}




bool nnode::has_planes()
{
    return CSGHasPlanes(type) ;
}

unsigned nnode::planeIdx()
{
    return param.u.x ; 
}
unsigned nnode::planeNum()
{
    return param.u.y ; 
}





std::string nnode::desc() const 
{
    std::stringstream ss ; 
    ss  
        << " [" << std::setw(3) << idx << "] "
        << CSGTag(type) 
        << " "
        << ( complement ? "!" : " " )
        << ( label ? label : " " )
        ;     
    return ss.str();
}


void nnode::pdump(const char* msg) const 
{
    LOG(info) << msg << " verbosity " << verbosity ; 
    assert(0 && "override nnode::pdump in primitives" );
}


void nnode::nudge(unsigned /*uv_surf*/, float /*delta*/ ) 
{
    assert(0 && "override nnode::nudge in primitives" );
}



void nnode::dump(const char* msg) const 
{
    bool prim = is_primitive();

    std::cout 
          << std::setw(10) << msg << " " 
          << desc() 
          << ( prim ? " PRIM " : " OPER " )
          << " v:" << std::setw(1) << verbosity 
          ; 

    if(prim)
    {
        nbbox bb = bbox();
        std::cout << " bb " << bb.desc() << std::endl ; 
    }
    else
    {
        std::cout << std::endl ; 

        left->dump("left");
        right->dump("right");
    }

/*
    if(transform)
    {
        //std::cout << "transform: " << glm::to_string( *transform ) << std::endl ; 
        std::cout << "transform: " << *transform  << std::endl ; 
    } 
*/

}


bool nnode::is_primitive() const 
{
    return left == NULL && right == NULL ; 
}
bool nnode::is_bileaf() const 
{
    return !is_primitive() && left->is_primitive() && right->is_primitive() ; 
}




void nnode::Init( nnode& n , OpticksCSG_t type, nnode* left, nnode* right )
{
    n.idx = 0 ; 
    n.type = type ; 

    n.left = left ; 
    n.right = right ; 
    n.parent = NULL ; 
    n.other  = NULL ;   // used by NOpenMesh 
    n.label = NULL ; 

    n.transform = NULL ; 
    n.gtransform = NULL ; 
    n.gtransform_idx = 0 ; 
    n.complement = false ; 
    n.verbosity = 0 ; 

    n.param.u  = {0u,0u,0u,0u};
    n.param1.u = {0u,0u,0u,0u};
    n.param2.u = {0u,0u,0u,0u};
    n.param3.u = {0u,0u,0u,0u};

}




const char* nnode::csgname()
{
   return CSGName(type);
}
unsigned nnode::maxdepth()
{
    return _maxdepth(0);
}
unsigned nnode::_maxdepth(unsigned depth)  // recursive 
{
    return left && right ? nmaxu( left->_maxdepth(depth+1), right->_maxdepth(depth+1)) : depth ;  
}


nmat4triple* nnode::global_transform()
{
    return global_transform(this);
}

nmat4triple* nnode::global_transform(nnode* n)
{
    std::vector<nmat4triple*> tvq ; 
    while(n)
    {
        if(n->transform) tvq.push_back(n->transform);
        n = n->parent ; 
    }

    // tvq transforms are in reverse hierarchical order from leaf back up to root
    // these are the CSG nodes
    bool reverse = true ; 
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq, reverse) ; 
}

void nnode::update_gtransforms()
{
    update_gtransforms_r(this);
}
void nnode::update_gtransforms_r(nnode* node)
{
    // NB this downwards traversal doesnt need parent links, 
    // but the ancestor collection of global_transforms that it uses does...
    //
    node->gtransform = node->global_transform();

    if(node->left && node->right)
    {
        update_gtransforms_r(node->left);
        update_gtransforms_r(node->right);
    }
}



glm::vec3 nnode::apply_gtransform(const glm::vec4& v_) const 
{
    glm::vec4 v(v_) ; 
    if(gtransform) v = gtransform->t * v ; 
    return glm::vec3(v) ; 
}


glm::vec3 nnode::gseeddir()   // override in shapes if needed
{
    glm::vec4 dir(1,1,1,0); 
    return apply_gtransform(dir);
}


npart nnode::part()
{
    // this is invoked by NCSG::export_r to totally re-write the nodes buffer 
    // BUT: is it being used by partlist approach, am assuming not by not setting bbox


    npart pt ; 
    pt.zero();
    pt.setParam(  param );
    pt.setParam1( param1 );
    pt.setParam2( param2 );
    pt.setParam3( param3 );

    pt.setTypeCode( type );
    pt.setGTransform( gtransform_idx, complement );

    // gtransform_idx is index into a buffer of the distinct compound transforms for the tree

    if(npart::VERSION == 0u)
    {
        nbbox bb = bbox();
        pt.setBBox( bb );  
    }

    return pt ; 
}


void nnode::composite_bbox( nbbox& bb ) const 
{
    assert( left && right );

    nbbox l_bb = left->bbox();
    nbbox r_bb = right->bbox();


    if(left->is_primitive())
    {
        bool l_invert_is_complement = l_bb.invert == left->complement ;

        if(!l_invert_is_complement)
           LOG(fatal) << "nnode::composite_bbox"
                      << " l_invert_is_complement FAIL "
                      << " l_bb  " << l_bb.desc()
                      << " left  " << left->desc()
                      ;
                     

        assert( l_invert_is_complement );
    }

    if(right->is_primitive())
    {

        bool r_invert_is_complement = r_bb.invert == right->complement ;

        if(!r_invert_is_complement)
           LOG(fatal) << "nnode::composite_bbox"
                      << " r_invert_is_complement FAIL "
                      << " r_bb  " << r_bb.desc()
                      << " right  " << right->desc()
                      ;
                     

        assert( r_invert_is_complement );
    }
    
    nbbox::CombineCSG(bb, l_bb, r_bb, type, verbosity  );

    if(verbosity > 0)
    std::cout << "nnode::composite_bbox "
              << " left " << left->desc()
              << " right " << right->desc()
              << " bb " << bb.desc()
              << std::endl 
              ;

}  

nbbox nnode::bbox() const 
{
    if(verbosity > 0)
    LOG(info) << "nnode::bbox " << desc() ; 

    nbbox bb = make_bbox() ; 
    if(is_primitive())
    { 
        assert(0 && "nnode::bbox() needs to be overridden for all primitives" );
    } 
    else 
    {
        composite_bbox(bb);
    }
    return bb ; 
}



void nnode::Scan( std::vector<float>& sd, const nnode& node, const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& tt, bool dump )
{
    LOG(info) << "nnode::Scan" ;
    std::cout 
        << " origin " << origin 
        << " direction " << direction
        << " range " << tt
        << std::endl ; 

    sd.clear(); 

    for(float t=tt.x ; t <= tt.y ; t+= tt.z)
    {
        glm::vec3 p = origin + t * direction ;  
        float sd_ = node(p.x,p.y,p.z) ;        
        sd.push_back(sd_);

        if(dump)
        std::cout
                 << " t " <<  std::fixed << std::setprecision(4) << std::setw(10) << t  
                 << " x " <<  std::fixed << std::setprecision(4) << std::setw(10) << p.x 
                 << " y " <<  std::fixed << std::setprecision(4) << std::setw(10) << p.y 
                 << " z " <<  std::fixed << std::setprecision(4) << std::setw(10) << p.z 
                 << " : " <<  std::fixed << std::setprecision(4) << std::setw(10) <<  sd_
                 << std::endl ; 
    }
}



nnode* nnode::load(const char* treedir, int verbosity)
{
    bool usedglobally = false ; 
    bool polygonize = false ; 

    NCSG* tree = NCSG::LoadTree(treedir, usedglobally, verbosity, polygonize );
    nnode* root = tree->getRoot();
    return root ; 
}



/**
To translate or rotate a surface modeled as an SDF, you can apply the inverse
transformation to the point before evaluating the SDF.

**/


float nunion::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fminf( l, r );
}
float nintersection::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, r );
}
float ndifference::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, -r);    // difference is intersection with complement, complement negates signed distance function
}


void nnode::Tests(std::vector<nnode*>& nodes )
{
    nsphere* a = new nsphere(make_sphere(0.f,0.f,-50.f,100.f));
    nsphere* b = new nsphere(make_sphere(0.f,0.f, 50.f,100.f));
    nbox*    c = new nbox(make_box(0.f,0.f,0.f,200.f));

    nunion* u = new nunion(make_union( a, b ));
    nintersection* i = new nintersection(make_intersection( a, b )); 
    ndifference* d1 = new ndifference(make_difference( a, b )); 
    ndifference* d2 = new ndifference(make_difference( b, a )); 
    nunion* u2 = new nunion(make_union( d1, d2 ));

    nodes.push_back( (nnode*)a );
    nodes.push_back( (nnode*)b );
    nodes.push_back( (nnode*)u );
    nodes.push_back( (nnode*)i );
    nodes.push_back( (nnode*)d1 );
    nodes.push_back( (nnode*)d2 );
    nodes.push_back( (nnode*)u2 );

    nodes.push_back( (nnode*)c );


    float radius = 200.f ; 
    float inscribe = 1.3f*radius/sqrt(3.f) ; 

    nsphere* sp = new nsphere(make_sphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_box(0.f,0.f,0.f, inscribe ));
    nunion*  u_sp_bx = new nunion(make_union( sp, bx ));
    nintersection*  i_sp_bx = new nintersection(make_intersection( sp, bx ));
    ndifference*    d_sp_bx = new ndifference(make_difference( sp, bx ));
    ndifference*    d_bx_sp = new ndifference(make_difference( bx, sp ));

    nodes.push_back( (nnode*)u_sp_bx );
    nodes.push_back( (nnode*)i_sp_bx );
    nodes.push_back( (nnode*)d_sp_bx );
    nodes.push_back( (nnode*)d_bx_sp );


}





void nnode::AdjustToFit(nnode* root, const nbbox& container, float scale) 
{
    if( root->type == CSG_BOX )
    {
        nbox* box = dynamic_cast<nbox*>(root)  ;
        assert(box) ; 
        box->adjustToFit(container, scale );
    }
    else if( root->type == CSG_SPHERE )
    {
        nsphere* sph = dynamic_cast<nsphere*>(root)  ;
        assert(sph) ; 
        sph->adjustToFit(container, scale );
    }
    else
    {
        LOG(fatal) << "nnode::AdjustToFit"
                   << " auto-containement only implemented for BOX and SPHERE"
                   << " root: " << root->desc()
                   ; 
        assert(0 && "nnode::AdjustToFit" ); 
    }
}




std::function<float(float,float,float)> nnode::sdf() const 
{
    //  return node types operator() callable as function
    //  somehow the object parameter member data goes along for the ride

    const nnode* node = this ; 
    std::function<float(float,float,float)> f ; 

    switch(node->type)   
    {
        case CSG_UNION:          { nunion* n        = (nunion*)node         ; f = *n ; } break ;
        case CSG_INTERSECTION:   { nintersection* n = (nintersection*)node  ; f = *n ; } break ;
        case CSG_DIFFERENCE:     { ndifference* n   = (ndifference*)node    ; f = *n ; } break ;
        case CSG_SPHERE:         { nsphere* n       = (nsphere*)node        ; f = *n ; } break ;
        case CSG_ZSPHERE:        { nzsphere* n      = (nzsphere*)node       ; f = *n ; } break ;
        case CSG_BOX:            { nbox* n          = (nbox*)node           ; f = *n ; } break ;
        case CSG_BOX3:           { nbox* n          = (nbox*)node           ; f = *n ; } break ;
        case CSG_SLAB:           { nslab* n         = (nslab*)node          ; f = *n ; } break ; 
        case CSG_PLANE:          { nplane* n        = (nplane*)node         ; f = *n ; } break ; 
        case CSG_CYLINDER:       { ncylinder* n     = (ncylinder*)node      ; f = *n ; } break ; 
        case CSG_DISC:           { ndisc* n         = (ndisc*)node          ; f = *n ; } break ; 
        case CSG_CONE:           { ncone* n         = (ncone*)node          ; f = *n ; } break ; 
        case CSG_CONVEXPOLYHEDRON:{ nconvexpolyhedron* n = (nconvexpolyhedron*)node ; f = *n ; } break ; 
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
    return f ;
}

int nnode::par_euler() const 
{
    assert(0 && "this need to be overridden");
    return 0 ; 
}

unsigned nnode::par_nsurf() const 
{
    assert(0 && "this need to be overridden");
    return 0 ; 
}
unsigned nnode::par_nvertices(unsigned , unsigned ) const 
{
    assert(0 && "this need to be overridden");
    return 0 ; 
}

glm::vec3 nnode::par_pos(const nuv& uv) const  // override in shapes 
{
    unsigned surf = uv.s();
    assert(0 && "this need to be overridden");
    assert( surf < par_nsurf());
    glm::vec3 pos ;
    return pos ;  
}




void nnode::collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs, int verbosity)
{
    std::vector<nnode*> prim ; 
    collect_prim(prim);    // recursive collection of list of all primitives in tree
    unsigned nprim = prim.size();

    if(verbosity > 0)
    LOG(info) << "nnode::collect_prim_centers"
              << " verbosity " << verbosity
              << " nprim " << nprim 
              ;

    for(unsigned i=0 ; i < nprim ; i++)
    {
        nnode* p = prim[i] ; 

        if(verbosity > 1 )
        LOG(info) << "nnode::collect_prim_centers"
                  << " i " << i 
                  << " type " << p->type 
                  << " name " << CSGName(p->type) 
                  ;


        switch(p->type)
        {
            case CSG_SPHERE: 
               {  
                   nsphere* n = (nsphere*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  

            case CSG_ZSPHERE: 
               {  
                   nzsphere* n = (nzsphere*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  
          
            case CSG_BOX: 
            case CSG_BOX3: 
               {  
                   nbox* n = (nbox*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  

            case CSG_SLAB: 
               {  
                   nslab* n = (nslab*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  

            case CSG_PLANE: 
               {  
                   nplane* n = (nplane*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  
 
            case CSG_CYLINDER: 
               {  
                   ncylinder* n = (ncylinder*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  

            case CSG_DISC: 
               {  
                   ndisc* n = (ndisc*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  

            case CSG_CONE: 
               {  
                   ncone* n = (ncone*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  

            case CSG_CONVEXPOLYHEDRON: 
               {  
                   nconvexpolyhedron* n = (nconvexpolyhedron*)p ;
                   centers.push_back(n->gseedcenter()); 
                   dirs.push_back(n->gseeddir());
               }
               break ;  
 
            default:
               {
                   LOG(fatal) << "nnode::collect_prim_centers unhanded shape type " << p->type << " name " << CSGName(p->type) ;
                   assert(0) ;
               }
        }
    }
}


void nnode::collect_prim(std::vector<nnode*>& prim)
{
    collect_prim_r(prim, this);   
}

void nnode::collect_prim_r(std::vector<nnode*>& prim, nnode* node)
{
    bool internal = node->left && node->right ; 
    if(!internal)
    {
        prim.push_back(node);
    }
    else
    {
        collect_prim_r(prim, node->left);
        collect_prim_r(prim, node->right);
    }
}


void nnode::dump_prim( const char* msg)
{
    std::vector<nnode*> prim ;
    collect_prim(prim);   
    unsigned nprim = prim.size();
    LOG(info) << msg << " nprim " << nprim ; 
    for(unsigned i=0 ; i < nprim ; i++)
    {
        nnode* p = prim[i] ; 
        switch(p->type)
        {
            case CSG_SPHERE: ((nsphere*)p)->pdump("sp") ; break ; 
            case CSG_BOX   :    ((nbox*)p)->pdump("bx") ; break ; 
            case CSG_SLAB   :  ((nslab*)p)->pdump("sl") ; break ; 
            default:
            {
                   LOG(fatal) << "nnode::dump_prim unhanded shape type " << p->type << " name " << CSGName(p->type) ;
                   assert(0) ;
            }
        }
    }
}




bool nnode::can_uncoincide(const nnode* a, const nnode* b) const 
{
    return ( a && b && a->type == CSG_BOX3 && b->type == CSG_BOX3 ) ;
}

unsigned nnode::uncoincide()
{
    // canonically invoked for bileaf from NCSG::import_r

    assert( is_bileaf() && "nnode::uncoincide expects left and right to be primitives" );
  
    float epsilon = 1e-5f ; 
    unsigned level = 1 ; 
    int margin = 1 ; 

    std::vector<nuv> coincident ;
    
    nnode* a = NULL ; 
    nnode* b = NULL ; 

    // hmm theres an implicit assumption here that all complements have 
    // already fed down to the leaves
    assert(!complement); 

    if( type == CSG_DIFFERENCE )  // left - right 
    {
        a = left ; 
        b = right ; 
    }
    else if( type == CSG_INTERSECTION && !left->complement &&  right->complement)  // left * !right  ->   left - right
    { 
        a = left ; 
        b = right ; 
    }
    else if( type == CSG_INTERSECTION &&  left->complement && !right->complement)  // !left * right  ->  right - left 
    {
        a = right ; 
        b = left ; 
    }

    if( a && b )
    {
        if(!can_uncoincide(a, b))
        {
            LOG(debug) << "nnode::uncoincide detects bileaf A-B subtraction, but must skip as not implemented for: "
                         << " A " << a->csgname()
                         << " B " << b->csgname()
                          ;
            return 0 ; 
        }


        a->getCoincident( coincident, b, epsilon, level, margin, FRAME_LOCAL );

        unsigned ncoin = coincident.size() ;
        if(ncoin > 0)
        {
            LOG(info) << "nnode::uncoincide   START " ; 

            a->verbosity = 4 ; 
            b->verbosity = 4 ; 

            a->pdump("A");
            b->pdump("B");

            assert( ncoin == 1);
            nuv uv = coincident[0] ; 

            //float delta = 5*epsilon ; 
            float delta = 1 ;  // crazy large delta, so can see it  

            std::cout << "nnode::uncoincide" 
                      << " ncoin " << ncoin 
                      << " uv " << uv.desc()
                      << " epsilon " << epsilon
                      << " delta " << delta
                      << std::endl 
                      ;

            b->nudge( uv.s(),  delta ); 

            b->pdump("B(nudged)");


            LOG(info) << "nnode::uncoincide   DONE " ; 
        } 
    }
    return coincident.size() ;
}




void nnode::dumpSurfacePointsAll(const char* msg, NNodeFrameType fr) const 
{
    LOG(info) << msg << " nnode::dumpSurfacePointsAll " ; 

    unsigned level = 1 ;  // +---+---+
    int margin = 1 ;      // o---*---o

    std::cout 
              << FrameType(fr)
              << " verbosity " << verbosity     
              << " uvlevel " << level     
              << " uvmargin " << margin
              << std::endl ;     

    std::vector<glm::vec3> points ; 
    getSurfacePointsAll( points, level, margin, fr ); 

    for(unsigned i=0 ; i < points.size() ; i++) 
          std::cout
               << " i " << std::setw(4) << i 
               << " " 
               << glm::to_string(points[i]) 
               << std::endl
               ; 
}



void nnode::getSurfacePointsAll(std::vector<glm::vec3>& surf, unsigned level, int margin, NNodeFrameType fr) const 
{
     unsigned ns = par_nsurf();
     for(unsigned s = 0 ; s < ns ; s++)
     {    
         getSurfacePoints(surf, s, level, margin, fr) ;
     }
}

void nnode::getSurfacePoints(std::vector<glm::vec3>& surf, int s, unsigned level, int margin, NNodeFrameType fr) const 
{
    /*

     level 1  
          (1 << 1) = 2 divisions in u and v,  +1 for endpost -> 3 points in u and v 

     margin 1
          skip the start and end of uv range 
          3 points - 2*margin ->   1 point   at mid uv of the face 

          +---+---+
          |   |   |
          +---*---+
          |   |   |
          +---+---+
    */
    int nu = 1 << level ; 
    int nv = 1 << level ; 
    assert( (nv - margin) > 0 );
    assert( (nu - margin) > 0 );

    int ndiv = nu + 1 - 2*margin ;
    unsigned expect = ndiv*ndiv  ;  
    unsigned n0 = surf.size();

    for (int v = margin; v <= (nv-margin) ; v++)
    {
        for (int u = margin; u <= (nu-margin) ; u++) 
        {
            nuv uv = make_uv(s,u,v,nu,nv);
            glm::vec3 pos = par_pos_(uv, fr );
            surf.push_back(pos) ;
        }
    }
    unsigned n1 = surf.size();
    assert( n1 - n0 == expect );
}


void nnode::getCoincidentSurfacePoints(std::vector<nuv>& surf, int s, unsigned level, int margin, const nnode* other, float epsilon, NNodeFrameType fr) const 
{
    // Checking the disposition of parametric points of parametric surface s of this node 
    // with respect to the implicit distance to the surface of the other node...  
    //
    // Parametric uv coordinates of this node are collected into *surf* 
    // when their positions are within epsilon of the surface of the other node.
    //
    // The frame of this nodes paramtric positions and the frame of the other nodes 
    // signed distance to surface are specified by the fr arguement, typically FRAME_LOCAL
    // is appropriate.
    //


    int nu = 1 << level ; 
    int nv = 1 << level ; 
    assert( (nv - margin) > 0 );
    assert( (nu - margin) > 0 );

    for (int v = margin; v <= (nv-margin)  ; v++)
    {
        for (int u = margin; u <= (nu-margin) ; u++) 
        {
            nuv uv = make_uv(s,u,v,nu,nv);

            glm::vec3 pos = par_pos_(uv, fr);

            float other_sdf = other->sdf_(pos, fr );

            if( fabs(other_sdf) < epsilon )  
            {
                surf.push_back(uv) ;
            }
        }
    }
}

void nnode::getCoincident(std::vector<nuv>& coincident, const nnode* other, float epsilon, unsigned level, int margin, NNodeFrameType fr) const 
{
     unsigned ns = par_nsurf();
     for(unsigned s = 0 ; s < ns ; s++)
     {    
         getCoincidentSurfacePoints(coincident, s, level, margin, other, epsilon, fr) ;
     }
}



