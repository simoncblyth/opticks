
/**
CSGNodeScanTest.cc
=====================

For lower level test see CSGIntersectTest.cc

**/

#include <vector>
#include <cmath>

#include "OPTICKS_LOG.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "NP.hh"

#include "scuda.h"
#include "squad.h"
#include "sqat4.h"

#define DEBUG 1 
#include "csg_intersect_node.h"
#include "csg_intersect_tree.h"

struct Scan 
{
    NP* ipos ; 
    NP* iray ; 
    NP* isec ; 

    Scan(int num);  
    static Scan* XY(const CSGNode* node,  int n=100, const char* modes="0,1,2,3", float t_min=0., bool shifted=true );  
    void save(const char* dir) const ; 
}; 

Scan::Scan(int num)
    :
    ipos(NP::Make<float>(num, 3)),
    iray(NP::Make<float>(num, 2, 3)),
    isec(NP::Make<float>(num, 4))
{
}

void Scan::save(const char* dir) const 
{
    ipos->save(dir, "ipos.npy");
    iray->save(dir, "iray.npy");
    isec->save(dir, "isec.npy");
}


Scan* Scan::XY(const CSGNode* node, int n, const char* modes_, float t_min, bool shifted )
{
    std::vector<int> modes ; 
    SStr::ISplit(modes_, modes, ',' ); 
    int num = n*modes.size()  ; 

    Scan* scan = new Scan(num);  

    float* _ipos = scan->ipos->values<float>() ; 
    float* _iray = scan->iray->values<float>() ; 
    float* _isec = scan->isec->values<float>() ; 
  
    unsigned offset = 0 ; 
    for(unsigned m=0 ; m < modes.size() ; m++)
    {
        int mode = modes[m]; 
        float dx, dy, ox, oy ; 
        for(int i=0 ; i < n ; i++)
        {
            int j = i - n/2 ; 
            if( mode == 0 )       // shoot upwards from X axis, or shifted line
            {
                dx = 0.f ; 
                dy = 1.f ; 
                ox = j*0.1f ; 
                oy = shifted ? -10.f : 0. ; 
            }
            else if( mode == 1 )  //  shoot downwards from X axis, or shifted line
            {
                dx = 0.f ; 
                dy = -1.f ; 
                ox = j*0.1f ; 
                oy = shifted ?  10.f : 0. ; 
            }
            else if( mode == 2 )   // shoot to right from Y axis, or shifted line 
            {
                dx = 1.f ; 
                dy = 0.f ; 
                ox = shifted ? -10.f : 0.  ; 
                oy = j*0.1f ; 
            }
            else if( mode == 3 )  // shoot to left from Y axis, or shifted line
            {
                dx = -1.f ; 
                dy = 0.f ; 
                ox = shifted ? 10.f : 0.  ; 
                oy = j*0.1f ; 
            }

            float3 ray_origin    = make_float3( ox, oy, 0.f ); 
            float3 ray_direction = make_float3( dx, dy, 0.f ); 

            _iray[(i+offset)*2*3+0] = ray_origin.x ; 
            _iray[(i+offset)*2*3+1] = ray_origin.y ; 
            _iray[(i+offset)*2*3+2] = ray_origin.z ;

            _iray[(i+offset)*2*3+3] = ray_direction.x ; 
            _iray[(i+offset)*2*3+4] = ray_direction.y ; 
            _iray[(i+offset)*2*3+5] = ray_direction.z ;
  
            float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ); 
            const float4* plan = nullptr ; 
            const qat4* itra = nullptr ; 
            bool valid_intersect = intersect_node(isect, node, plan, itra, t_min, ray_origin, ray_direction ); 

            if(valid_intersect)
            {
                float t = isect.w ;  
                _ipos[(i+offset)*3 + 0] = ray_origin.x + t*ray_direction.x ; 
                _ipos[(i+offset)*3 + 1] = ray_origin.y + t*ray_direction.y ; 
                _ipos[(i+offset)*3 + 2] = ray_origin.z + t*ray_direction.z ; 

                _isec[(i+offset)*4 + 0] = isect.x ; 
                _isec[(i+offset)*4 + 1] = isect.y ; 
                _isec[(i+offset)*4 + 2] = isect.z ; 
                _isec[(i+offset)*4 + 3] = isect.w ; 
            }
        }
        offset += n ; 
    }
    return scan ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);  
     
    const char* name = "iphi" ; 
    CSGNode nd = CSGNode::MakeDemo(name) ;  
    Scan* scan = Scan::XY(&nd); 
  
    const char* base = "$TMP/CSGNodeScanTest" ; 
    int create_dirs = 2 ; // 2:dirpath   
    const char* fold = SPath::Resolve(base, name, create_dirs ); 

    LOG(info) << " save to " << fold ; 
    scan->save(fold); 

    return 0 ; 
}   
