
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
#include "NBBox.hpp"

// primitives
#include "NSphere.hpp"
#include "NZSphere.hpp"
#include "NBox.hpp"
#include "NSlab.hpp"
#include "NPlane.hpp"
#include "NCylinder.hpp"
#include "NCone.hpp"
#include "NConvexPolyhedron.hpp"


#include "PLOG.hh"


float nnode::operator()(float,float,float) const 
{
    return 0.f ; 
} 

std::string nnode::desc()
{
    std::stringstream ss ; 
    ss  << " nnode "
        << std::setw(3) << type 
        << ( complement ? "!" : "" )
        << std::setw(15) << CSGName(type) 
        ;     
    return ss.str();
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


void nnode::dump(const char* msg)
{
    printf("(%s)%s\n",csgname(), msg);
    if(left && right)
    {
        left->dump("left");
        right->dump("right");
    }

    if(transform)
    {
        //std::cout << "transform: " << glm::to_string( *transform ) << std::endl ; 
        std::cout << "transform: " << *transform  << std::endl ; 
    } 

}

void nnode::Init( nnode& n , OpticksCSG_t type, nnode* left, nnode* right )
{
    n.idx = 0 ; 
    n.type = type ; 

    n.left = left ; 
    n.right = right ; 
    n.parent = NULL ; 
    n.label = NULL ; 

    n.transform = NULL ; 
    n.gtransform = NULL ; 
    n.gtransform_idx = 0 ; 
    n.complement = false ; 

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
    return tvq.size() == 0 ? NULL : nmat4triple::product(tvq) ; 
}



void nnode::update_gtransforms()
{
    update_gtransforms_r(this);
}
void nnode::update_gtransforms_r(nnode* node)
{
    // NB this traversal doesnt need parent links, but global_transforms does...
    node->gtransform = node->global_transform();

    if(node->left && node->right)
    {
        update_gtransforms_r(node->left);
        update_gtransforms_r(node->right);
    }
}



glm::vec3 nnode::apply_gtransform(const glm::vec4& v_)
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


nbbox nnode::bbox() const 
{
   // needs to be overridden for primitives
    nbbox bb = make_bbox() ; 
    if(left && right)
    {
        bb.include( left->bbox() );
        bb.include( right->bbox() );
    }
    return bb ; 
}



void nnode::Scan( const nnode& node, const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& tt )
{
    LOG(info) << "nnode::Scan" ;
    std::cout 
        << " origin " << origin 
        << " direction " << direction
        << " range " << tt
        << std::endl ; 

    for(float t=tt.x ; t <= tt.y ; t+= tt.z)
    {
        glm::vec3 p = origin + t * direction ;  
        std::cout
                 << " t " <<  std::fixed << std::setprecision(4) << std::setw(10) << t  
                 << " x " <<  std::fixed << std::setprecision(4) << std::setw(10) << p.x 
                 << " y " <<  std::fixed << std::setprecision(4) << std::setw(10) << p.y 
                 << " z " <<  std::fixed << std::setprecision(4) << std::setw(10) << p.z 
                 << " : " <<  std::fixed << std::setprecision(4) << std::setw(10) <<  node(p.x,p.y,p.z) 
                 << std::endl ; 
    }
}



nnode* nnode::load(const char* treedir, unsigned verbosity)
{
    NCSG* tree = NCSG::LoadTree(treedir, verbosity );
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
    return fminf(l, r);
}
float nintersection::operator()(float x, float y, float z) const 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, r);
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




std::function<float(float,float,float)> nnode::sdf()
{
    nnode* node = this ; 
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
        case CSG_CONE:           { ncone* n         = (ncone*)node          ; f = *n ; } break ; 
        case CSG_CONVEXPOLYHEDRON:{ nconvexpolyhedron* n = (nconvexpolyhedron*)node ; f = *n ; } break ; 
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
    return f ;
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


void nnode::dump_prim( const char* msg, int verbosity )
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
            case CSG_SPHERE: ((nsphere*)p)->pdump("sp",verbosity) ; break ; 
            case CSG_BOX   :    ((nbox*)p)->pdump("bx",verbosity) ; break ; 
            case CSG_SLAB   :  ((nslab*)p)->pdump("sl",verbosity) ; break ; 
            default:
            {
                   LOG(fatal) << "nnode::dump_prim unhanded shape type " << p->type << " name " << CSGName(p->type) ;
                   assert(0) ;
            }
        }
    }
}




