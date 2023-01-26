


#include "spa.h"
#include "sbb.h"
#include "sxf.h"

#include "scsg.hh"
#include "snd.hh"


scsg* snd::POOL = nullptr  ; 

void snd::SetPOOL( scsg* pool )
{
    POOL = pool ; 
}

int snd::Add(const snd& nd) // static
{
    assert( POOL && "snd::Add MUST SET snd::SetPOOL to scsg instance first" ); 
    return POOL->addNode(nd); 
}

void snd::init()
{
    tc = -1 ;
    fc = -1 ; 
    nc = -1 ;

    pa = -1 ; 
    bb = -1 ;
    xf = -1 ; 
}

void snd::setTypecode( unsigned _tc )
{
    init(); 
    tc = _tc ; 
}
void snd::setParam( double x, double y, double z, double w, double z1, double z2 )
{
    assert( POOL && "snd::setParam MUST SET snd::SetPOOL to scsg instance first" ); 
    spa o = { x, y, z, w, z1, z2 } ; 
    pa = POOL->addParam(o) ; 
}
void snd::setAABB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    assert( POOL && "snd::setAABB MUST SET snd::SetPOOL to scsg instance first" ); 
    sbb o = {x0, y0, z0, x1, y1, z1} ; 
    bb = POOL->addAABB(o) ; 
}
void snd::setXForm(const glm::tmat4x4<double>& t )
{
    assert( POOL && "snd::setXForm MUST SET snd::SetPOOL to scsg instance first" ); 
    sxf o ; 
    o.mat = t ; 
    xf = POOL->addXForm(o) ; 
}


std::string snd::brief() const 
{
    int w = 3 ; 
    std::stringstream ss ; 
    ss 
       << " tc:" << std::setw(w) << tc 
       << " fc:" << std::setw(w) << fc 
       << " nc:" << std::setw(w) << nc 
       << " pa:" << std::setw(w) << pa 
       << " bb:" << std::setw(w) << bb 
       << " xf:" << std::setw(w) << xf
       << " "    << CSG::Tag(tc) 
       ; 
    std::string str = ss.str(); 
    return str ; 
}

std::string snd::desc() const 
{
    std::stringstream ss ; 
    ss 
       << brief() << std::endl  
       << POOL->descPA(pa) << std::endl  
       << POOL->descBB(bb) << std::endl 
       << POOL->descXF(xf) << std::endl 
       ; 

    for(int i=0 ; i < nc ; i++) ss << POOL->descND(fc+i) << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}

std::ostream& operator<<(std::ostream& os, const snd& v)  
{
    os << v.desc() ;  
    return os; 
}

snd snd::Sphere(double radius)  // static
{
    assert( radius > zero ); 
    snd nd = {} ;
    nd.setTypecode(CSG_SPHERE) ; 
    nd.setParam( zero, zero, zero, radius, zero, zero );  
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  );  
    return nd ;
}

snd snd::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd nd = {} ;
    nd.setTypecode(CSG_ZSPHERE) ; 
    nd.setParam( zero, zero, zero, radius, z1, z2 );  
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  );  
    return nd ;
}

snd snd::Box3(double fullside)  // static 
{
    return Box3(fullside, fullside, fullside); 
}
snd snd::Box3(double fx, double fy, double fz )  // static 
{
    assert( fx > 0. );  
    assert( fy > 0. );  
    assert( fz > 0. );  

    snd nd = {} ;
    nd.setTypecode(CSG_BOX3) ; 
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f );  
    nd.setAABB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    return nd ; 
}

snd snd::Boolean( OpticksCSG_t op, int l, int r ) // static 
{
    assert( l > -1 && r > -1 );
    assert( l+1 == r );  

    snd nd = {} ;
    nd.setTypecode( op ); 
    nd.nc = 2 ; 
    nd.fc = l ;
 
    return nd ; 
}


