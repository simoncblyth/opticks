


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


const snd* snd::GetND(int idx){ return POOL ? POOL->getND(idx) : nullptr ; } // static
const spa* snd::GetPA(int idx){ return POOL ? POOL->getPA(idx) : nullptr ; } // static
const sxf* snd::GetXF(int idx){ return POOL ? POOL->getXF(idx) : nullptr ; } // static
const sbb* snd::GetBB(int idx){ return POOL ? POOL->getBB(idx) : nullptr ; } // static 

int snd::GetNDXF(int idx){ return POOL ? POOL->getNDXF(idx) : -1 ; } // static  

snd* snd::GetND_(int idx){ return POOL ? POOL->getND_(idx) : nullptr ; } // static
spa* snd::GetPA_(int idx){ return POOL ? POOL->getPA_(idx) : nullptr ; } // static
sxf* snd::GetXF_(int idx){ return POOL ? POOL->getXF_(idx) : nullptr ; } // static
sbb* snd::GetBB_(int idx){ return POOL ? POOL->getBB_(idx) : nullptr ; } // static 

std::string snd::DescND( int idx){ return POOL ? POOL->descND(idx) : "-" ; } // static
std::string snd::DescPA( int idx){ return POOL ? POOL->descPA(idx) : "-" ; } // static
std::string snd::DescXF( int idx){ return POOL ? POOL->descXF(idx) : "-" ; } // static
std::string snd::DescBB( int idx){ return POOL ? POOL->descBB(idx) : "-" ; } // static


int snd::Add(const snd& nd) // static
{
    assert( POOL && "snd::Add MUST SET snd::SetPOOL to scsg instance first" ); 
    return POOL->addND(nd); 
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

void snd::setTC(int _tc )
{
    init(); 
    tc = _tc ; 
}
void snd::setPA( double x, double y, double z, double w, double z1, double z2 )
{
    assert( POOL && "snd::setPA MUST SET snd::SetPOOL to scsg instance first" ); 
    spa o = { x, y, z, w, z1, z2 } ; 
    pa = POOL->addPA(o) ; 
}
void snd::setBB( double x0, double y0, double z0, double x1, double y1, double z1 )
{
    assert( POOL && "snd::setBB MUST SET snd::SetPOOL to scsg instance first" ); 
    sbb o = {x0, y0, z0, x1, y1, z1} ; 
    bb = POOL->addBB(o) ; 
}
void snd::setXF(const glm::tmat4x4<double>& t )
{
    assert( POOL && "snd::setXF MUST SET snd::SetPOOL to scsg instance first" ); 
    sxf o ; 
    o.mat = t ; 
    xf = POOL->addXF(o) ; 
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
       << DescPA(pa) << std::endl  
       << DescBB(bb) << std::endl 
       << DescXF(xf) << std::endl 
       ; 

    for(int i=0 ; i < nc ; i++) ss << DescND(fc+i) << std::endl ; 
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
    nd.setTC(CSG_SPHERE) ; 
    nd.setPA( zero, zero, zero, radius, zero, zero );  
    nd.setBB(  -radius, -radius, -radius,  radius, radius, radius  );  
    return nd ;
}

snd snd::ZSphere(double radius, double z1, double z2)  // static
{
    assert( radius > zero ); 
    assert( z2 > z1 );  
    snd nd = {} ;
    nd.setTC(CSG_ZSPHERE) ; 
    nd.setPA( zero, zero, zero, radius, z1, z2 );  
    nd.setBB(  -radius, -radius, z1,  radius, radius, z2  );  
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
    nd.setTC(CSG_BOX3) ; 
    nd.setPA( fx, fy, fz, 0.f, 0.f, 0.f );  
    nd.setBB( -fx*0.5 , -fy*0.5, -fz*0.5, fx*0.5 , fy*0.5, fz*0.5 );   
    return nd ; 
}

snd snd::Boolean( int op, int l, int r ) // static 
{
    assert( l > -1 && r > -1 );
    assert( l+1 == r );  

    snd nd = {} ;
    nd.setTC( op ); 
    nd.nc = 2 ; 
    nd.fc = l ;
 
    return nd ; 
}


