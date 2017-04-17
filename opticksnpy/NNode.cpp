
#include <cstdio>
#include <cassert>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "NGLM.hpp"
#include "NGLMExt.hpp"

#include "NNode.hpp"
#include "NPart.hpp"
#include "NQuad.hpp"
#include "NBBox.hpp"

// primitives
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NSlab.hpp"
#include "NPlane.hpp"
#include "NCylinder.hpp"


#include "PLOG.hh"


//double nnode::operator()(double,double,double) 
float nnode::operator()(float,float,float) 
{
    return 0.f ; 
} 

std::string nnode::desc()
{
    std::stringstream ss ; 
    ss  << " nnode "
        << std::setw(3) << type 
        << std::setw(15) << CSGName(type) 
        ;     
    return ss.str();
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

    n.param.f = {0.f, 0.f, 0.f, 0.f };
    n.param1.f = {0.f, 0.f, 0.f, 0.f };
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






npart nnode::part()
{
    // this is invoked by NCSG::export_r to totally re-write the nodes buffer 
    // BUT: is it being used by partlist approach, am assuming not by not setting bbox


    npart pt ; 
    pt.zero();
    pt.setParam( param );
    pt.setParam1( param1 );

    pt.setTypeCode( type );
    pt.setGTransform( gtransform_idx );
    // gtransform_idx is index into a buffer of the distinct compound transforms for the tree

    if(npart::VERSION == 0u)
    {
        nbbox bb = bbox();
        pt.setBBox( bb );  
    }

    return pt ; 
}


nbbox nnode::bbox()
{
   // needs to be overridden for primitives
    nbbox bb = make_nbbox() ; 
    if(left && right)
    {
        bb.include( left->bbox() );
        bb.include( right->bbox() );
    }
    return bb ; 
}



/**
To translate or rotate a surface modeled as an SDF, you can apply the inverse
transformation to the point before evaluating the SDF.

**/


float nunion::operator()(float x, float y, float z) 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fminf(l, r);
}
float nintersection::operator()(float x, float y, float z) 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, r);
}
float ndifference::operator()(float x, float y, float z) 
{
    assert( left && right );
    float l = (*left)(x, y, z) ;
    float r = (*right)(x, y, z) ;
    return fmaxf( l, -r);    // difference is intersection with complement, complement negates signed distance function
}


void nnode::Tests(std::vector<nnode*>& nodes )
{
    nsphere* a = new nsphere(make_nsphere(0.f,0.f,-50.f,100.f));
    nsphere* b = new nsphere(make_nsphere(0.f,0.f, 50.f,100.f));
    nbox*    c = new nbox(make_nbox(0.f,0.f,0.f,200.f));

    nunion* u = new nunion(make_nunion( a, b ));
    nintersection* i = new nintersection(make_nintersection( a, b )); 
    ndifference* d1 = new ndifference(make_ndifference( a, b )); 
    ndifference* d2 = new ndifference(make_ndifference( b, a )); 
    nunion* u2 = new nunion(make_nunion( d1, d2 ));

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

    nsphere* sp = new nsphere(make_nsphere(0.f,0.f,0.f,radius));
    nbox*    bx = new nbox(make_nbox(0.f,0.f,0.f, inscribe ));
    nunion*  u_sp_bx = new nunion(make_nunion( sp, bx ));
    nintersection*  i_sp_bx = new nintersection(make_nintersection( sp, bx ));
    ndifference*    d_sp_bx = new ndifference(make_ndifference( sp, bx ));
    ndifference*    d_bx_sp = new ndifference(make_ndifference( bx, sp ));

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
        case CSG_UNION:
            {
                nunion* n = (nunion*)node ; 
                f = *n ;
            }
            break ;
        case CSG_INTERSECTION:
            {
                nintersection* n = (nintersection*)node ; 
                f = *n ;
            }
            break ;
        case CSG_DIFFERENCE:
            {
                ndifference* n = (ndifference*)node ; 
                f = *n ;
            }
            break ;
        case CSG_SPHERE:
            {
                nsphere* n = (nsphere*)node ; 
                f = *n ;
            }
            break ;
        case CSG_BOX:
            {
                nbox* n = (nbox*)node ;  
                f = *n ;
            }
            break ;
        case CSG_SLAB:
            {
                nslab* n = (nslab*)node ;  
                f = *n ;
            }
            break ;
        case CSG_PLANE:
            {
                nplane* n = (nplane*)node ;  
                f = *n ;
            }
            break ;
        case CSG_CYLINDER:
            {
                ncylinder* n = (ncylinder*)node ;  
                f = *n ;
            }
            break ;
        default:
            LOG(fatal) << "Need to add upcasting for type: " << node->type << " name " << CSGName(node->type) ;  
            assert(0);
    }
    return f ;
}



void nnode::collect_prim_centers(std::vector<glm::vec3>& centers, std::vector<glm::vec3>& dirs )
{
    std::vector<nnode*> prim ; 
    collect_prim(prim); 

    unsigned npr = prim.size();
    for(unsigned i=0 ; i < npr ; i++)
    {
        nnode* p = prim[i] ; 
        switch(p->type)
        {
            case CSG_SPHERE: 
               {  
                   nsphere* n = (nsphere*)p ;
                   centers.push_back(n->gcenter()); 
                   glm::vec4 dir(1,1,1,0); 
                   if(n->gtransform) dir = n->gtransform->t * dir ; 
                   dirs.push_back( glm::vec3(dir));
               }
               break ;  
          
            case CSG_BOX: 
               {  
                   nbox* n = (nbox*)p ;
                   centers.push_back(n->gcenter()); 
                   
                   glm::vec4 dir(1,1,1,0); 
                   if(n->gtransform) dir = n->gtransform->t * dir ; 
                   dirs.push_back( glm::vec3(dir));
               }
               break ;  

            case CSG_SLAB: 
               {  
                   nslab* s = (nslab*)p ;
                   centers.push_back(s->gcenter()); 
                   glm::vec4 dir(s->n,0); 
                   if(s->gtransform) dir = s->gtransform->t * dir ; 

                   dirs.push_back( glm::vec3(dir));
               }
               break ;  

            case CSG_PLANE: 
               {  
                   nplane* n = (nplane*)p ;
                   centers.push_back(n->gcenter()); 
                   glm::vec4 dir(n->n,0); 
                   if(n->gtransform) dir = n->gtransform->t * dir ; 

                   dirs.push_back( glm::vec3(dir));
               }
               break ;  
 
            case CSG_CYLINDER: 
               {  
                   ncylinder* n = (ncylinder*)p ;
                   centers.push_back(n->gcenter()); 
                   glm::vec4 dir(0,0,1,0); 
                   if(n->gtransform) dir = n->gtransform->t * dir ; 

                   dirs.push_back( glm::vec3(dir));
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




