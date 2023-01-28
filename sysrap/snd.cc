
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
void snd::SetNodeXForm(int idx, const glm::tmat4x4<double>& tr )
{
    snd* nd = GetNode_(idx); 
    nd->setXForm(tr); 
}


/**
snd::SetLVID
--------------

This gets invoked from the snd roots only, 
as its called for each root from U4Tree::initSolid

**/
void snd::SetLVID(int idx, int lvid)  // label node tree 
{
    snd* nd = GetNode_(idx); 
    nd->setLVID(lvid);   

    int chk = nd->checktree(); 
    if( chk != 0 )
    { 
        std::cerr 
           << "snd::SetLVID" 
           << " idx " << idx 
           << " lvid " << lvid 
           << " checktree " << chk 
           << std::endl 
           //<< " POOL.desc " 
           //<< POOL->desc() 
           ; 
    }
    //assert( chk == 0 ); 
}



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

    // HMM: what about changing transform ? 
    // Currently just adds another and updates the ref, "leaking" the old. 
}


void snd::setLVID(int lvid_)
{
    setLVID_r(lvid_, 0); 
}

void snd::setLVID_r(int lvid_, int d )
{
    lvid = lvid_ ;  
    depth = d ;     

    int ch = first_child ; 
    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 
        child.setLVID_r(lvid, depth+1 ); 
        ch = child.next_sibling ;
    }
}

int snd::checktree() const 
{
    int chk_D = checktree_r('D', 0); 
    int chk_P = checktree_r('P', 0); 
    int chk = chk_D + chk_P ; 

    if( chk > 0 ) 
    {
        std::cerr 
            << "snd::checktree"
            << " chk_D " << chk_D
            << " chk_P " << chk_P
            << brief()
            << std::endl
            ;
    }

    return chk ; 
}
int snd::checktree_r(char code,  int d ) const 
{
    int chk = 0 ; 
    int ch = first_child ; 

    if( code == 'D' ) // check expected depth
    {
        if(d != depth) chk += 1 ; 
    }
    else if( code == 'P' ) // check for non-roots without parent set 
    {
        if( depth > 0 && parent < 0 ) chk += 1 ; 
    }


    while( ch > -1 )
    {
        snd& child = POOL->node[ch] ; 

        chk += child.checktree_r(code,  d + 1 );  

        ch = child.next_sibling ;
    }
    return chk ; 
}




double snd::zmin() const 
{
    assert( CSG::CanZNudge(typecode) ); 
    assert( param > -1 ); 
    const spa& pa = POOL->param[param] ; 
    return pa.zmin(); 
}

double snd::zmax() const 
{
    assert( CSG::CanZNudge(typecode) ); 
    assert( param > -1 ); 
    const spa& pa = POOL->param[param] ; 
    return pa.zmax(); 
}

void snd::check_z() const 
{
    assert( CSG::CanZNudge(typecode) ); 
    assert( param > -1 ); 
    assert( aabb > -1 ); 

    const spa& pa = POOL->param[param] ; 
    const sbb& bb = POOL->aabb[aabb] ; 

    assert( pa.zmin() == bb.zmin() ); 
    assert( pa.zmax() == bb.zmax() ); 
}


/**
snd::decrease_zmin
-------------------

   bb.z1 +--------+ pa.z2 
         |        |
         |        |
         |________|
   bb.z0 +~~~~~~~~+ pa.z1

**/

void snd::decrease_zmin( double dz )
{
    check_z(); 

    spa& pa = POOL->param[param] ; 
    sbb& bb = POOL->aabb[aabb] ; 

    pa.decrease_zmin(dz); 
    bb.decrease_zmin(dz); 
}

/**
snd::increase_zmax
-------------------

::

   bb.z1 +~~~~~~~~+ pa.z2
         +--------+       
         |        |
         |        |
         |        |
   bb.z0 +--------+ pa.z1

**/

void snd::increase_zmax( double dz )
{
    check_z(); 

    spa& pa = POOL->param[param] ; 
    sbb& bb = POOL->aabb[aabb] ; 

    pa.increase_zmax(dz) ; 
    bb.increase_zmax(dz) ; 
}

/**
snd::ZDesc
-----------

   +----+
   |    |
   +----+
   |    |
   +----+
   |    |
   +----+

**/

std::string snd::ZDesc(const std::vector<int>& prims) // static
{
    std::stringstream ss ; 
    ss << "snd::ZDesc" ; 
    ss << " prims(" ;
    for(unsigned i=0 ; i < prims.size() ; i++) ss << prims[i] << " " ; 
    ss << ") " ;
    ss << std::endl ;  

    for(unsigned i=0 ; i < prims.size() ; i++)
    {
        int _a = prims[i];
        snd& a = POOL->node[_a] ; 
        ss << std::setw(3) << _a 
           << ":" 
           << " " << std::setw(10) << a.zmin() 
           << " " << std::setw(10) << a.zmax()
           << std::endl 
           ; 
    }
    std::string str = ss.str(); 
    return str ; 
}

/**
snd::ZNudgeEnds
-----------------

CAUTION: changes geometry, only appropriate 
for subtracted consituents eg inners 

**/

void snd::ZNudgeEnds(const std::vector<int>& prims) // static
{
    std::cout 
       << std::endl
       << "snd::ZNudgeEnds "
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;

    /*
    for(unsigned i=1 ; i < prims.size() ; i++)
    {
        int _a = prims[i-1]; 
        int _b = prims[i]; 

        snd& a = POOL->node[_a] ; 
        snd& b = POOL->node[_b] ; 
         
        a.check_z(); 
        b.check_z();
    }
    */
}

void snd::ZNudgeJoints(const std::vector<int>& prims) // static
{
    std::cout 
       << std::endl
       << "snd::ZNudgeJoints "
       << std::endl
       << ZDesc(prims)
       << std::endl
       ;
}




std::string snd::desc() const 
{
    std::stringstream ss ; 
    ss 
       << "[snd::desc" << std::endl
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

        bool consistent_parent_index = child.parent == nd.index ; 

        if(!consistent_parent_index) ss 
            << "snd::desc "
            << " FAIL consistent_parent_index "
            << " ch " << ch 
            << " count " << count 
            << " child.parent " << child.parent
            << " nd.index " << nd.index
            << " nd.lvid "  << nd.lvid
            << " child.index " << child.index
            << " child.lvid "  << child.lvid
            << std::endl 
            ;

        //assert(consistent_parent_index);  
        count += 1 ;         
        ch = child.next_sibling ;
    }

    bool expect_child = count == nd.num_child ; 

    if(!expect_child) 
    {
        ss << std::endl << " FAIL count " << count << " num_child " << num_child << std::endl; 
    }
    assert(expect_child); 
    ss << "]snd::desc" << std::endl ; 
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

    snd* ln = POOL->getND_(l) ; 
    snd* rn = POOL->getND_(r) ; 

    ln->next_sibling = r ; 
    ln->sibdex = 0 ; 

    rn->next_sibling = -1 ; 
    rn->sibdex = 1 ; 

    int idx = Add(nd) ; 

    std::cout << "snd::Boolean change l " << l << " ln.parent " << ln->parent << " to " << idx << std::endl ; 
    std::cout << "snd::Boolean change r " << r << " rn.parent " << rn->parent << " to " << idx << std::endl ; 

    ln->parent = idx ; 
    rn->parent = idx ; 

    return idx ; 
}

int snd::Compound(int type, const std::vector<int>& prims )
{
    assert( type == CSG_CONTIGUOUS || type == CSG_DISCONTIGUOUS ); 

    int num_prim = prims.size(); 
    assert( num_prim > 0 ); 

    snd nd = {} ;
    nd.setTypecode( type ); 
    nd.num_child = num_prim ; 
    nd.first_child = prims[0] ;
    int idx = Add(nd) ; 

    for(int i=0 ; i < num_prim ; i++)
    {
        int i_sib = prims[i]; 
        int p_sib = i > 0 ? prims[i-1] : -1 ; 

        snd& i_child = POOL->node[i_sib] ; 
        i_child.sibdex = i ; 
        i_child.parent = idx ; 
        i_child.next_sibling = -1 ; 

        // other than for last i = num_prim-1 
        // the next_sibling gets reset by prior "reach back" below  

        if(i > 0)
        {
            assert( p_sib > -1 ); 
            snd& p_child = POOL->node[p_sib] ; 
            p_child.next_sibling = i_sib ; 
        }
    }
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


