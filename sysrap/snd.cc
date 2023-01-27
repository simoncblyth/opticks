
#include <iostream>

#include "spa.h"
#include "sbb.h"
#include "sxf.h"

#include "OpticksCSG.h"
#include "scsg.hh"
#include "snd.hh"

scsg* snd::POOL = nullptr  ; 
void snd::SetPOOL( scsg* pool ){ POOL = pool ; }  // static 

NPFold* snd::Serialize(){ return POOL ? POOL->serialize() : nullptr ; }  // static 
void    snd::Import(const NPFold* fold){ assert(POOL) ; POOL->import(fold) ; } // static 
std::string snd::Desc(){  return POOL ? POOL->desc() : "? NO POOL ?" ; } // static 


const snd* snd::GetNode( int idx){ return POOL ? POOL->getND(idx) : nullptr ; } // static
const spa* snd::GetParam(int idx){ return POOL ? POOL->getPA(idx) : nullptr ; } // static
const sxf* snd::GetXForm(int idx){ return POOL ? POOL->getXF(idx) : nullptr ; } // static
const sbb* snd::GetAABB( int idx){ return POOL ? POOL->getBB(idx) : nullptr ; } // static 

int snd::GetNodeXForm(int idx){ return POOL ? POOL->getNDXF(idx) : -1 ; } // static  

snd* snd::GetNode_(int idx){  return POOL ? POOL->getND_(idx) : nullptr ; } // static
spa* snd::GetParam_(int idx){ return POOL ? POOL->getPA_(idx) : nullptr ; } // static
sxf* snd::GetXForm_(int idx){ return POOL ? POOL->getXF_(idx) : nullptr ; } // static
sbb* snd::GetAABB_(int idx){  return POOL ? POOL->getBB_(idx) : nullptr ; } // static 

std::string snd::Desc(      int idx){ return POOL ? POOL->descND(idx) : "-" ; } // static
std::string snd::DescParam( int idx){ return POOL ? POOL->descPA(idx) : "-" ; } // static
std::string snd::DescXForm( int idx){ return POOL ? POOL->descXF(idx) : "-" ; } // static
std::string snd::DescAABB(  int idx){ return POOL ? POOL->descBB(idx) : "-" ; } // static


int snd::Add(const snd& nd) // static
{
    assert( POOL && "snd::Add MUST SET snd::SetPOOL to scsg instance first" ); 
    return POOL->addND(nd); 
}

void snd::init()
{
    index = -1 ; 
    depth = -1 ; 
    sibdex = -1 ; 
    parent = -1 ; 

    num_child = -1 ; 
    first_child = -1 ; 
    next_sibling = -1 ; 
    lvid = -1 ;

    typecode = -1  ; 
    param = -1 ; 
    aabb = -1 ; 
    xform = -1 ; 
}

std::string snd::brief() const 
{
    int w = 5 ; 
    std::stringstream ss ; 
    ss 
       << " ix:" << std::setw(w) << index
       << " dp:" << std::setw(w) << depth
       << " sx:" << std::setw(w) << sibdex
       << " pt:" << std::setw(w) << parent
       << "    "
       << " nc:" << std::setw(w) << num_child 
       << " fc:" << std::setw(w) << first_child
       << " ns:" << std::setw(w) << next_sibling
       << " lv:" << std::setw(w) << lvid
       << "    "
       << " tc:" << std::setw(w) << typecode 
       << " pa:" << std::setw(w) << param 
       << " bb:" << std::setw(w) << aabb
       << " xf:" << std::setw(w) << xform
       << "    "
       << CSG::Tag(typecode) 
       ; 
    std::string str = ss.str(); 
    return str ; 
}







void snd::setTypecode(int _tc )
{
    init(); 
    typecode = _tc ; 
    num_child = 0 ;  // maybe changed later
}
void snd::setParam( double x, double y, double z, double w, double z1, double z2 )
{
    assert( POOL && "snd::setParam MUST SET snd::SetPOOL to scsg instance first" ); 
    spa o = { x, y, z, w, z1, z2 } ; 

    param = POOL->addPA(o) ; 
}
void snd::setAABB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    assert( POOL && "snd::setAABB MUST SET snd::SetPOOL to scsg instance first" ); 
    sbb o = {x0, y0, z0, x1, y1, z1} ; 

    aabb = POOL->addBB(o) ; 
}
void snd::setXForm(const glm::tmat4x4<double>& t )
{
    assert( POOL && "snd::setXForm MUST SET snd::SetPOOL to scsg instance first" ); 
    sxf o ; 
    o.mat = t ; 
    xform = POOL->addXF(o) ; 
}

std::string snd::desc() const 
{
    std::stringstream ss ; 
    ss 
       << brief() << std::endl  
       << DescParam(param) << std::endl  
       << DescAABB(aabb) << std::endl 
       << DescXForm(xform) << std::endl 
       ; 

    const snd& nd = *this ; 
    int ch = nd.first_child ; 
    int count = 0 ; 

    while( ch > -1 )
    {
        ss << Desc(ch) << std::endl ; 
        const snd& child = POOL->node[ch] ; 
        assert( child.parent == nd.index );  
        count += 1 ;         
        ch = child.next_sibling ;
    }

    bool expect_child = count == nd.num_child ; 

    if(!expect_child) 
    {
        ss << std::endl << " FAIL count " << count << " num_child " << num_child << std::endl; 
    }
    assert(expect_child); 

    std::string str = ss.str(); 
    return str ; 
}

std::ostream& operator<<(std::ostream& os, const snd& v)  
{
    os << v.desc() ;  
    return os; 
}

/**
snd::Boolean
--------------

**/

int snd::Boolean( int op, int l, int r ) // static 
{
    assert( l > -1 && r > -1 );

    snd nd = {} ;
    nd.setTypecode( op ); 
    nd.num_child = 2 ; 
    nd.first_child = l ;

    snd& ln = POOL->node[l] ; 
    snd& rn = POOL->node[r] ; 

    ln.next_sibling = r ; 
    rn.next_sibling = -1 ; 

    int idx = Add(nd) ; 

    ln.parent = idx ; 
    rn.parent = idx ; 

    return idx ; 
}

int snd::Cylinder(double radius, double z1, double z2) // static
{
    assert( z2 > z1 );  
    snd nd = {} ;
    nd.setTypecode(CSG_CYLINDER); 
    nd.setParam( 0.f, 0.f, 0.f, radius, z1, z2)  ;   
    nd.setAABB( -radius, -radius, z1, +radius, +radius, z2 );   
    return Add(nd) ; 
}

int snd::Cone(double r1, double z1, double r2, double z2)  // static
{   
    assert( z2 > z1 );
    double rmax = fmax(r1, r2) ; 
    snd nd = {} ;
    nd.setTypecode(CSG_CONE) ;
    nd.setParam( r1, z1, r2, z2, 0., 0. ) ;
    nd.setAABB( -rmax, -rmax, z1, rmax, rmax, z2 );
    return Add(nd) ;
}

int snd::Sphere(double radius)  // static
{
    assert( radius > zero ); 
    snd nd = {} ;
    nd.setTypecode(CSG_SPHERE) ; 
    nd.setParam( zero, zero, zero, radius, zero, zero );  
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  );  
    return Add(nd) ;
}

int snd::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd nd = {} ;
    nd.setTypecode(CSG_ZSPHERE) ; 
    nd.setParam( zero, zero, zero, radius, z1, z2 );  
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  );  
    return Add(nd) ;
}

int snd::Box3(double fullside)  // static 
{
    return Box3(fullside, fullside, fullside); 
}
int snd::Box3(double fx, double fy, double fz )  // static 
{
    assert( fx > 0. );  
    assert( fy > 0. );  
    assert( fz > 0. );  

    snd nd = {} ;
    nd.setTypecode(CSG_BOX3) ; 
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f );  
    nd.setAABB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    return Add(nd) ; 
}

int snd::Zero(double  x,  double y,  double z,  double w,  double z1, double z2)
{
    snd nd = {} ;
    nd.setTypecode(CSG_ZERO) ; 
    nd.setParam( x, y, z, w, z1, z2 );  
    return Add(nd) ; 
}


