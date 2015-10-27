#pragma once

#include "GMatrix.hh"
#include <climits>



struct gfloat2 
{
    gfloat2() : u(0.f), v(0.f) {} ;
    gfloat2(float _u, float _v) : u(_u), v(_v) {} ;
    gfloat2(const gfloat2& other ) : u(other.u), v(other.v)  {} ;

    void Summary(const char* msg)
    {
        printf("%s gfloat2 %10.3f %10.3f \n", msg, u, v);
    }

    float u,v ;
};


struct gfloat3 
{
    gfloat3() : x(0.f), y(0.f), z(0.f) {} ;
    gfloat3(float _x) : x(_x), y(_x), z(_x) {} ;
    gfloat3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {} ;
    gfloat3(const gfloat3& other ) : x(other.x), y(other.y), z(other.z)  {} ;

    bool operator==(const gfloat3& other) const 
    {
        return x == other.x && y == other.y && z == other.z   ;
    }

    gfloat3& operator *= (const GMatrixF& m)
    {
       float _x, _y, _z ;   
       _x = m.a1 * x + m.a2 * y + m.a3 * z + m.a4;
       _y = m.b1 * x + m.b2 * y + m.b3 * z + m.b4;
       _z = m.c1 * x + m.c2 * y + m.c3 * z + m.c4;

       x = _x ; 
       y = _y ; 
       z = _z ; 

       return *this ;
    }

    void Summary(const char* msg)
    {
        printf("%s gfloat3 %10.3f %10.3f %10.3f\n", msg, x, y, z);
    }


    float x,y,z ;
};



struct gfloat4 
{
    gfloat4() : x(0.f), y(0.f), z(0.f), w(0.f) {} ;
    gfloat4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {} ;
    gfloat4(gfloat3& v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) {} ;
    gfloat4(const gfloat4& other ) : x(other.x), y(other.y), z(other.z), w(other.w) {} ;

    bool operator==(const gfloat4& other) const 
    {
        return x == other.x && y == other.y && z == other.z && w == other.w  ;
    }

    gfloat4& operator *= (const GMatrixF& m)
    {
       float _x, _y, _z, _w ;   
       _x = m.a1 * x + m.a2 * y + m.a3 * z + m.a4 * w;
       _y = m.b1 * x + m.b2 * y + m.b3 * z + m.b4 * w ;
       _z = m.c1 * x + m.c2 * y + m.c3 * z + m.c4 * w ;
       _w = m.d1 * x + m.d2 * y + m.d3 * z + m.d4 * w ;

       x = _x ; 
       y = _y ; 
       z = _z ; 
       w = _w ; 

       return *this ;
    }

    void Summary(const char* msg)
    {
        printf("%s gfloat4 %10.3f %10.3f %10.3f %10.3f\n", msg, x, y, z, w);
    }


    float x,y,z,w ;
};





struct gbbox 
{
   gbbox() : min(gfloat3(0.f)), max(gfloat3(0.f)) {} ;
   gbbox(const gfloat3& _min, const gfloat3& _max) :  min(_min), max(_max) {} ; 
   gbbox(const gbbox& other ) : min(other.min), max(other.max) {} ;

   gfloat3 dimensions()
   {
       return gfloat3(max.x - min.x, max.y - min.y, max.z - min.z );
   } 
   gfloat3 center()
   {
       return gfloat3( (max.x + min.x)/2.0f , (max.y + min.y)/2.0f , (max.z + min.z)/2.0f ) ;
   }

   float extent(const gfloat3& dim)
   {
       float _extent(0.f) ;
       _extent = std::max( dim.x , _extent );
       _extent = std::max( dim.y , _extent );
       _extent = std::max( dim.z , _extent );
       _extent = _extent / 2.0f ;         
       return _extent ; 
   }

   gfloat4 center_extent()
   {
       gfloat3 cen = center();
       gfloat3 dim = dimensions();
       float ext = extent(dim); 

       return gfloat4( cen.x, cen.y, cen.z, ext );
   } 

   void Summary(const char* msg)
   {
       printf("%s min %10.3f %10.3f %10.3f  max %10.3f %10.3f %10.3f \n", msg, min.x, min.y, min.z, max.x, max.y, max.z );
   }

   gfloat3 min ; 
   gfloat3 max ; 
};








struct guint3 
{
    guint3() : x(0), y(0), z(0) {} ;
    guint3(unsigned int _x, unsigned int _y, unsigned int _z) : x(_x), y(_y), z(_z) {} ;

    unsigned int x,y,z ;
};



struct guint4
{
    guint4() : x(0), y(0), z(0), w(0) {} ;
    guint4(unsigned int _x, unsigned int _y, unsigned int _z, unsigned int _w) : x(_x), y(_y), z(_z), w(_w) {} ;

    unsigned int const operator[](unsigned int index) const
    {
        switch(index)
        {  
           case 0:return x; break;
           case 1:return y; break;
           case 2:return z; break;
           case 3:return w; break;
        }
        return INT_MAX ; 
    } 

    bool operator==(const guint4& other) const 
    {
       return 
           x == other.x && 
           y == other.y && 
           z == other.z && 
           w == other.w 
           ;
    }
    void Summary(const char* msg) const 
    {
       printf("%s : %10u %10u %10u %10u \n", msg, x, y, z, w);
    }

    unsigned int x,y,z,w ;
};





