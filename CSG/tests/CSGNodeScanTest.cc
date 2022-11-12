/**
CSGNodeScanTest.cc
=====================

For lower level tests see::

   intersect_leaf_box3_test.cc
   intersect_leaf_cylinder_test.cc
   intersect_leaf_phicut_test.cc  

**/

#include <vector>
#include <cmath>

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "NP.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

//#define DEBUG 1 
#include "csg_intersect_leaf.h"
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

struct Scan 
{
    const char* geom ; 
    CSGNode nd ; 
    float t_min ; 
    int num ; 
    bool shifted ; 
    const char* modes_ ; 
    const char* axes_ ; 
    std::vector<int>* modes ; 
    std::vector<int>* axes ; 

    NP* simtrace ; 
    quad4* qq ;  
    const char* fold ; 

    Scan(); 
    void init(); 
    std::string desc() const ; 
    void save() const ; 
}; 

Scan::Scan()
    :
    geom(SSys::getenvvar("CSGNodeScanTest_GEOM", "iphi")),
    nd(CSGNode::MakeDemo(geom)),  
    t_min(SSys::getenvfloat("TMIN",0.f)),
    num(SSys::getenvint("NUM", 200)),
    shifted(true),
    modes_(SSys::getenvvar("MODES", "0,1,2,3")),
    axes_(SSys::getenvvar("AXES", "0,2,1")),  // X,Z,Y  3rd gets set to zero 
    modes(SStr::ISplit(modes_, ',')),
    axes(SStr::ISplit(axes_, ',')),
    simtrace(NP::Make<float>(modes->size(),num,4,4)),
    qq((quad4*)simtrace->values<float>()),
    fold(SPath::Resolve("$TMP/CSGNodeScanTest", geom, DIRPATH )) 
{
    init(); 
}

std::string Scan::desc() const
{
    std::stringstream ss ; 
    ss << "Scan::desc" << std::endl 
       << " geom " << geom << std::endl
       << " num " << num << std::endl
       << " modes_ " << modes_ << std::endl
       << " axes_ " << axes_ << std::endl
       << " simtrace.sstr " << simtrace->sstr() << std::endl
       << " fold " << fold << std::endl 
       ;

    std::string s = ss.str(); 
    return s ; 
}

void Scan::save() const 
{
    simtrace->save(fold, "simtrace.npy");
}

void Scan::init()
{
    assert( axes->size() == 3 ); 
    int h = (*axes)[0] ; 
    int v = (*axes)[1] ; 
    int d = (*axes)[2] ; 

    unsigned offset = 0 ; 
    for(unsigned m=0 ; m < modes->size() ; m++)
    {
        int mode = (*modes)[m]; 
        float vx, vy, ox, oy ; 
        for(int i=0 ; i < num ; i++)
        {
            int j = i - num/2 ; 
            if( mode == 0 )       // shoot upwards from X axis, or shifted line
            {
                vx = 0.f ; 
                vy = 1.f ; 
                ox = j*0.1f ; 
                oy = shifted ? -10.f : 0. ; 
            }
            else if( mode == 1 )  //  shoot downwards from X axis, or shifted line
            {
                vx = 0.f ; 
                vy = -1.f ; 
                ox = j*0.1f ; 
                oy = shifted ?  10.f : 0. ; 
            }
            else if( mode == 2 )   // shoot to right from Y axis, or shifted line 
            {
                vx = 1.f ; 
                vy = 0.f ; 
                ox = shifted ? -10.f : 0.  ; 
                oy = j*0.1f ; 
            }
            else if( mode == 3 )  // shoot to left from Y axis, or shifted line
            {
                vx = -1.f ; 
                vy = 0.f ; 
                ox = shifted ? 10.f : 0.  ; 
                oy = j*0.1f ; 
            }


            // standard simtrace layout see sevent.h sevent::add_simtrace
            quad4& _qq = qq[m*num+i] ; 
            float* oo = (float*)&_qq.q2.f.x ;  // ray_origin
            float* dd = (float*)&_qq.q3.f.x ;  // ray_direction

            float3* ray_origin    = (float3*)&_qq.q2.f.x ; 
            float3* ray_direction = (float3*)&_qq.q3.f.x ; 
            float3* position      = (float3*)&_qq.q1.f.x ; 

            oo[h] = ox ; 
            oo[v] = oy ; 
            oo[d] = 0.f ; 

            dd[h] = vx ; 
            dd[v] = vy ; 
            dd[d] = 0.f ; 
           
            float4* isect = &_qq.q0.f ; 
            const float4* plan = nullptr ; 
            const qat4* itra = nullptr ; 
            const CSGNode* node = &nd ; 

            bool valid_intersect = intersect_node(*isect, node, node, plan, itra, t_min, *ray_origin, *ray_direction ); 
            // TODO: this should be using higher level intersect_prim ???

            if(valid_intersect)
            {
                float t = (*isect).w ;  
                *position = *ray_origin + t*(*ray_direction) ; 
            }
        }
        offset += num ; 
    }
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);  

    LOG(info) << " running scan " ; 
    Scan s ; 
    LOG(info) << s.desc() ; 
    s.save(); 
    return 0 ; 
}   
