/**
CSGQueryTest
==============

DUMP=2 NUM=210 CSGQueryTest A
   dump miss

**/

#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"

#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"


struct CSGQueryTest 
{ 
    static const int VERBOSE ; 

    const CSGFoundry* fd ; 
    const CSGQuery* q ; 
    CSGDraw* d ; 
    int gsid ; 

    static const char* DUMP ; 
    int dump ;  
    bool dump_hit ; 
    bool dump_miss ; 
    bool dump_dist ; 
 
    const char* name ; 
    float3 ray_origin ; 
    float3 ray_direction ; 
    float tmin ; 
    int num ; 


    std::vector<quad4>* isects ;
    CSGQueryTest(); 

    void create_isects(); 
    std::string config( 
        const char* _name, 
        const char* _ray_origin, 
        const char* _ray_direction, 
        const char* _tmin, 
        const char* _num 
        ); 

    void operator()( char mode); 
    void save() const ; 

    void intersect(int idx, float3* mod_ray_origin=nullptr , float3* mod_ray_direction=nullptr); 
    void distance(int idx, float3* mod_ray_origin=nullptr ); 

    void OneIntersection(); 
    void OneDistance(); 
    void Load(); 
    void XScan(); 
    void LookForSeam(); 
    void PhiScan(); 
    void AxisScan(); 
    void PacmanPhiLine0();
    void PacmanPhiLine1();
    void PacmanPhiLine2();
}; 

const char* CSGQueryTest::DUMP=" ( 0:no 1:hit 2:miss 3:hit+miss ) " ; 
const int CSGQueryTest::VERBOSE = SSys::getenvint("VERBOSE", 0 ); 


CSGQueryTest::CSGQueryTest()
    :
    fd(CSGFoundry::LoadGeom()),
    q(new CSGQuery(fd)),
    d(new CSGDraw(q)),
    gsid(0),
    dump(SSys::getenvint("DUMP",0)),
    dump_hit(  (dump & 1) != 0 ),
    dump_miss( (dump & 2) != 0 ),
    dump_dist( (dump & 4) != 0 ),
    name("noname"),
    isects(nullptr)
{
    if(VERBOSE > 0 )
    {
        LOG(info) << " GEOM " << fd->geom ; 
        d->draw("CSGQueryTest");
    }
}

void CSGQueryTest::operator()( char mode)
{
    if(VERBOSE > 0 ) LOG(info) << " mode " << mode ; 
    switch(mode)
    {
        case 'O': OneIntersection() ; break ; 
        case 'D': OneDistance()     ; break ; 
        case 'L': Load()            ; break ; 
        case 'S': LookForSeam()     ; break ; 
        case 'X': XScan()           ; break ; 
        case 'P': PhiScan()         ; break ; 
        case 'A': AxisScan()        ; break ; 
        case '0': PacmanPhiLine0()  ; break ; 
        case '1': PacmanPhiLine1()  ; break ; 
        case '2': PacmanPhiLine2()  ; break ; 
        default: assert(0 && "mode unhandled" ) ; break ; 
    }
}

void CSGQueryTest::create_isects()
{
    assert( num > 0 ); 
    isects = new std::vector<quad4>(num) ; 

    if( num == 1 )  // when only one : dump everything 
    {
        dump_hit = true ; 
        dump_miss = true ; 
        dump_dist = true ; 
    }

}

std::string CSGQueryTest::config( 
    const char* _name, 
    const char* _ray_origin, 
    const char* _ray_direction, 
    const char* _tmin,
    const char* _num
    )
{
    name = strdup(_name); 

    qvals(ray_origin   , "ORI",  _ray_origin );  
    qvals(ray_direction, "DIR",  _ray_direction );  
    qvals(tmin         , "TMIN", _tmin ); 
    qvals(num          , "NUM",  _num ); 

    ray_direction = normalize(ray_direction); 

    create_isects(); 

    std::stringstream ss ; 
    ss 
       << " name " << name << std::endl 
       << " dump " << dump 
       
       << " dump_hit " << dump_hit 
       << " dump_miss " << dump_miss 
       << DUMP 
       << std::endl 
       << " ORI ray_origin " << ray_origin << std::endl  
       << " DIR ray_direction " << ray_direction  << std::endl
       << " TMIN tmin " << tmin << std::endl 
       << " GSID gsid " << gsid << std::endl 
       << " NUM num " << num << std::endl 
       ;

    std::string s = ss.str(); 

    if(VERBOSE > 0 ) LOG(info) << std::endl << s ; 
    return s ; 
}

void CSGQueryTest::intersect(int idx, float3* mod_ray_origin, float3* mod_ray_direction )
{
    assert( idx < num ); 
    quad4& isect = (*isects)[idx] ; 

    bool valid_intersect = q->intersect(isect, tmin, 
                                   mod_ray_origin    ? *mod_ray_origin    : ray_origin, 
                                   mod_ray_direction ? *mod_ray_direction : ray_direction, 
                                   gsid ); 

    bool do_dump = ( dump_hit && valid_intersect ) || ( dump_miss && !valid_intersect );  
    if(do_dump) std::cout << CSGQuery::Desc( isect, name, &valid_intersect ) << std::endl ; 
}

void CSGQueryTest::distance(int idx, float3* mod_ray_origin )
{
    assert( idx < num ); 
    quad4& isect = (*isects)[idx] ; 

    q->distance( isect, mod_ray_origin ? *mod_ray_origin : ray_origin ); 

    if(dump_dist) std::cout << CSGQuery::Desc( isect, name, nullptr ) << std::endl ; 
}


void CSGQueryTest::save() const 
{
    NP::Write("/tmp", "CSGQueryTest.npy",  (float*)isects->data(), isects->size(), 4, 4 ); 
}

void CSGQueryTest::XScan()
{
    config("XScan", "0,0,0", "0,0,1", "0", "10" );  

    float3 ray_direction_0 = ray_direction ; 
    float3 ray_direction_1 = ray_direction ; 
    ray_direction_0.z =  1.f ; 
    ray_direction_1.z = -1.f ; 

    ray_direction_0 = normalize(ray_direction_0 ); 
    ray_direction_1 = normalize(ray_direction_1 ); 
 
    for(int i=0 ; i < num ; i++)
    {
        ray_origin.x = float(i - num/2) ;  
        intersect( 2*i+0, nullptr, &ray_direction_0 ); 
        intersect( 2*i+1, nullptr, &ray_direction_1 ); 
    }
}

void CSGQueryTest::PhiScan()
{
    config("PhiScan", "0,0,0", "1,0,0", "0", "12" );  

    for(int i=0 ; i < num ; i++)
    {
         double fphi = double(i)/double(num-1) ; 
         double fphi_deg = fphi*360. ; 
         double phi = fphi*2.*M_PIf ;
 
         ray_direction.x = std::cos( phi );
         ray_direction.y = std::sin( phi ); 
         ray_direction.z = 0.f ; 
 
         std::cout << " i " << std::setw(3) << i << " fphi " << fphi << " fphi_deg " << fphi_deg << std::endl ; 
         intersect(i);             
    }
}

void CSGQueryTest::AxisScan()
{
    config("AxisScan", "-200,0,0", "1,0,0", "0", "100" ); 
    for( int i=0 ; i < num ; i++)
    {
       ray_origin.z = float(i-num/2) ; 
       intersect(i); 
    }
}
void CSGQueryTest::LookForSeam()
{
    config("LookForSeam", "-200,0,10", "1,0,0", "0", "1" ); 
    intersect(0); 
}
void CSGQueryTest::PacmanPhiLine0()
{
    config("PacmanPhiLine0", "0,0,0", "1,1,0", "0", "1" ); 
    intersect(0); 
}
void CSGQueryTest::PacmanPhiLine1()
{
    config("PacmanPhiLine1", "0,0,0", "1,-1,0", "0", "1" ); 
    intersect(0); 
}
void CSGQueryTest::PacmanPhiLine2()
{
    config("PacmanPhiLine1", "1,1,0", "1,1,0", "0", "1" ); 
    intersect(0); 
}
void CSGQueryTest::OneIntersection()
{
    config("One", "-150,0,0", "1,0,0", "0", "1" ); 
    intersect(0); 
}
void CSGQueryTest::OneDistance()
{
    config("Dist", "-150,0,0", "1,0,0", "0", "1" ); 
    distance(0); 
}





/**
CSGQueryTest::Load
-------------------

LOAD "L" mode loads an isect subset written by CSGOptiX/tests/CSGOptiXRenderTest.py
providing a way to rerun a pixel with exactly the same ray_origin, ray_direction and tmin using eg::

    YX=0,0 DUMP=3 CSGQueryTest
    YX=1,1 DUMP=3 CSGQueryTest

The pixel to rerun is chosed by (IY,IX) coordinate where::

    (0,0) is top left pixel 
    (1,0) is pixel below the top left pixel 

**/

void CSGQueryTest::Load()
{
    const char* defaultpath = "$TMP/CSGOptiX/CSGOptiXRenderTest/dy_1_dx_1.npy" ; 
    const char* loadpath_ = SSys::getenvvar("LOAD", defaultpath ) ;  
    int create_dirs = 0 ; 
    const char* loadpath = SPath::Resolve(loadpath_, create_dirs) ;   

    NP* a = NP::Load(loadpath); 

    int2 yx ; 
    qvals( yx, "YX", "0,0" ); 
    int iy = yx.x ; 
    int ix = yx.y ; 

    LOG(info) << " a " << a->sstr() << " LOAD loadpath " << loadpath << " YX ( " << iy << "," << ix << " )"   ; 

    assert( a->shape.size() == 4 ); 
    int ni = a->shape[0] ;  
    int nj = a->shape[1] ;
    int nk = a->shape[2] ;  
    int nl = a->shape[3] ;

    assert( nk == 4 && nl == 4 ); 

    int ny = ni ; 
    int nx = nj ; 
    assert( ix < nx ); 
    assert( iy < ny ); 

    quad4 load_isect ; 
    load_isect.zero(); 

    unsigned itemoffset = iy*nx + ix ; 

    unsigned itembytes = sizeof(float)*4*4 ;  
    void* dst = &load_isect.q0.f.x ; 
    void* src = a->bytes() + itembytes*itemoffset ; 

    memcpy( dst, src, itembytes  );  

    if(VERBOSE > 0) LOG(info) 
        << "load_isect "  << std::endl
        << " q0.f " << load_isect.q0.f << std::endl 
        << " q1.f " << load_isect.q1.f << std::endl 
        << " q2.f " << load_isect.q2.f << std::endl 
        << " q3.f " << load_isect.q3.f << std::endl 
        ;

    ray_origin.x = load_isect.q2.f.x ; 
    ray_origin.y = load_isect.q2.f.y ; 
    ray_origin.z = load_isect.q2.f.z ;
    tmin = load_isect.q2.f.w ;

    ray_direction.x = load_isect.q3.f.x ; 
    ray_direction.y = load_isect.q3.f.y ; 
    ray_direction.z = load_isect.q3.f.z ;

    num = 1 ; 
    create_isects(); 
    intersect(0); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    char mode = argc > 1 ? argv[1][0] : 'O' ; 
    if( SSys::getenvbool("YX") ) mode = 'L' ;  // L mode loads isect for rerunning 

    CSGQueryTest t ; 
    t(mode); 

    return 0 ; 
}

