/**
CSGQueryTest
==============

DUMP=2 NUM=210 CSGQueryTest A
   dump miss

**/

#include "SSys.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"

#include "CSGFoundry.h"
#include "CSGQuery.h"
#include "CSGDraw.h"


struct CSGQueryTest 
{ 
    const CSGFoundry* fd ; 
    const CSGQuery* q ; 
    CSGDraw* d ; 
    int gsid ; 

    static const char* DUMP ; 
    int dump ;  
    bool dump_hit ; 
    bool dump_miss ; 
 
    const char* name ; 
    float3 ray_origin ; 
    float3 ray_direction ; 
    float tmin ; 
    int num ; 


    std::vector<quad4>* isects ;
    CSGQueryTest(); 

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
    void One(); 
    void XScan(); 
    void LookForSeam(); 
    void PhiScan(); 
    void AxisScan(); 
    void PacmanPhiLine0();
    void PacmanPhiLine1();
    void PacmanPhiLine2();
}; 

const char* CSGQueryTest::DUMP=" ( 0:no 1:hit 2:miss 3:hit+miss ) " ; 

CSGQueryTest::CSGQueryTest()
    :
    fd(CSGFoundry::LoadGeom()),
    q(new CSGQuery(fd)),
    d(new CSGDraw(q)),
    gsid(0),
    dump(SSys::getenvint("DUMP",0)),
    dump_hit(  (dump & 1) != 0 ),
    dump_miss( (dump & 2) != 0 ),
    name("noname"),
    isects(nullptr)
{
    LOG(info) << " GEOM " << fd->geom ; 
    d->draw("CSGQueryTest");
}

void CSGQueryTest::operator()( char mode)
{
    LOG(info) << " mode " << mode ; 
    switch(mode)
    {
        case 'O': One()            ; break ; 
        case 'S': LookForSeam()    ; break ; 
        case 'X': XScan()          ; break ; 
        case 'P': PhiScan()        ; break ; 
        case 'A': AxisScan()       ; break ; 
        case '0': PacmanPhiLine0() ; break ; 
        case '1': PacmanPhiLine1() ; break ; 
        case '2': PacmanPhiLine2() ; break ; 
        default: assert(0 && "mode unhandled" ) ; break ; 
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

    isects = new std::vector<quad4>(num) ; 

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
    LOG(info) << std::endl << s ; 
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
void CSGQueryTest::One()
{
    config("One", "-150,0,0", "1,0,0", "0", "1" ); 
    intersect(0); 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGQueryTest t ; 
    t( argc > 1 ? argv[1][0] : 'O'  ); 

    return 0 ; 
}

