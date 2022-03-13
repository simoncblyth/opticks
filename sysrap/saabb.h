#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <sstream>
   #include <iostream>
   #include <iomanip>
   #include <vector>
   #include <string>
#endif 

#define AABB_METHOD inline 


struct AABB
{
    float3 mn ; 
    float3 mx ; 

    static AABB Make(const float* v ); 
    const float* data() const ; 
    float3 center() const ; 
    float  extent() const ; 
    float4 center_extent() const ;     
    void center_extent(float4& ce) const ; 

    bool empty() const ; 
    void include_point(const float* point); 
    void include_point(const float3& p);
    void include_aabb( const float* aabb);

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static std::string Desc(const float* data); 
    static std::string Compare(unsigned& mismatch, const float* a, const float* b, int detail, float epsilon=1e-4); 
    static void cube_corners(std::vector<float3>& corners, const float4& ce );
#endif

}; 


AABB_METHOD AABB AABB::Make( const float* v )
{
    AABB bb = {} ; 
    bb.mn.x = *(v+0);  
    bb.mn.y = *(v+1);  
    bb.mn.z = *(v+2);
    bb.mx.x = *(v+3);  
    bb.mx.y = *(v+4);  
    bb.mx.z = *(v+5);
    return bb ; 
}

AABB_METHOD const float* AABB::data() const 
{
    return (const float*)&mn ;     // hmm assumes compiler adds no padding between mn and mx 
}
AABB_METHOD float3 AABB::center() const 
{
    return ( mx + mn )/2.f ;  
}
AABB_METHOD float AABB::extent() const 
{   
    float3 d = mx - mn ; 
    return fmaxf(fmaxf(d.x, d.y), d.z) /2.f ; 
}   
AABB_METHOD float4 AABB::center_extent() const 
{
    return make_float4( center(), extent() ); 
}

AABB_METHOD void AABB::center_extent(float4& ce) const 
{
    float3 c = center(); 
    ce.x = c.x ; 
    ce.y = c.y ; 
    ce.z = c.z ; 
    ce.w = extent() ; 
}

AABB_METHOD bool AABB::empty() const 
{   
    return mn.x == 0.f && mn.y == 0.f && mn.z == 0.f && mx.x == 0.f && mx.y == 0.f && mx.z == 0.f  ;   
}   

/*
AABB::include_point
--------------------


      +-  - - - -*   <--- included point pushing out the max, leaves min unchanged
      .          | 
      +-------+  .
      |       |  |
      |       |  .
      |       |  |
      +-------+- +

      +-------+  
      |    *  |     <-- interior point doesnt change min/max  
      |       |  
      |       |  
      +-------+ 

      +-------+-->--+  
      |       |     |  
      |       |     *  <--- side point pushes out max, leaves min unchanged
      |       |     |
      +-------+-----+ 

*/

AABB_METHOD void AABB::include_point(const float* point)
{
    const float3 p = make_float3( *(point+0), *(point+1), *(point+2) ); 
    include_point(p); 
}

AABB_METHOD void AABB::include_point(const float3& p)
{
    if(empty())
    {
        mn = p ; 
        mx = p ; 
    } 
    else
    {
        mn = fminf( mn, p );
        mx = fmaxf( mx, p );
    }
}




AABB_METHOD void AABB::include_aabb(const float* aabb)
{
    const float3 other_mn = make_float3( *(aabb+0), *(aabb+1), *(aabb+2) ); 
    const float3 other_mx = make_float3( *(aabb+3), *(aabb+4), *(aabb+5) ); 

    if(empty())
    {
        mn = other_mn ; 
        mx = other_mx ; 
    } 
    else
    {
        mn = fminf( mn, other_mn );
        mx = fmaxf( mx, other_mx );
    }
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline std::ostream& operator<<(std::ostream& os, const AABB& bb)
{
    os 
       << " [ "
       << bb.mn
       << " : "
       << bb.mx 
       << " | "
       << ( bb.mx - bb.mn )
       << " ] "
       ;
    return os; 
}

/**

     ZYX 
   0:000    
   1:001    +X
   2:010    +Y
   3:011
   4:100    +Z
   5:101
   6:110
   7:111


               110----------111         
                |            |
                |            |
  +Z   100----------101      | 
        |       |    |       | 
        |       |    |       |
        |      010---|------011       +Y
        |            | 
        |            | 
  -Z   000----------001        -Y        
                
       -X           +X

**/

AABB_METHOD void AABB::cube_corners(std::vector<float3>& corners, const float4& ce )
{
    for(int c=0 ; c < 8 ; c++)  // loop around the corners 
    {
        float3 a = make_float3( 
                                ce.x + ( c & 1 ? ce.w : -ce.w ), 
                                ce.y + ( c & 2 ? ce.w : -ce.w ),   
                                ce.z + ( c & 4 ? ce.w : -ce.w )
                              ) ; 
        corners.push_back(a) ;  
    }
}


AABB_METHOD std::string AABB::desc() const 
{
    std::stringstream ss ; 
    ss 
        << " mn " << mn 
        << " mx " << mx  
        ; 
    std::string s = ss.str(); 
    return s ; 
}

AABB_METHOD std::string AABB::Desc(const float* data)
{
    std::stringstream ss ; 
    for(int j=0 ; j < 6 ; j++) ss << std::fixed << std::setw(10) << std::setprecision(2) << *(data + j) << " " ; 
    std::string s = ss.str(); 
    return s ; 
}

AABB_METHOD std::string AABB::Compare(unsigned& mismatch, const float* a, const float* b, int detail, float epsilon)
{
    mismatch = 0 ; 
    std::stringstream ss ; 
    for(int j=0 ; j < 6 ; j++)
    {
        float ab = a[j] - b[j] ;  

        if( detail == 3 )
        {
            ss 
               << std::fixed << std::setw(10) << std::setprecision(2) << a[j] << " "
               << std::fixed << std::setw(10) << std::setprecision(2) << b[j] << " "
               << std::fixed << std::setw(10) << std::setprecision(2) << ab   << " "
               ;
        }
        else if( detail == 1 )
        {
            ss 
               << std::fixed << std::setw(10) << std::setprecision(2) << ab   << " "
               ;
        }


        if(std::abs(ab) > epsilon)  mismatch += 1 ;  
    }
    ss << " mismatch " << mismatch ; 
    std::string s = ss.str(); 
    return s ; 
}
 

#endif 


