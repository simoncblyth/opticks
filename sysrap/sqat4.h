#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QAT4_METHOD __device__ __host__ __forceinline__
   #define QAT4_FUNCTION  __device__ __host__ __forceinline__  
#else
   #define QAT4_METHOD 
   #define QAT4_FUNCTION inline 
#endif 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
   #include <vector>
   #include <sstream>
   #include <cstring>
   #include <algorithm>
#endif 

#include "squad.h"

struct qat4 
{
    quad q0, q1, q2, q3 ; 

    QAT4_METHOD void zero()
    {
        q0.f.x = 0.f ; q0.f.y = 0.f ; q0.f.z = 0.f ; q0.f.w = 0.f ; 
        q1.f.x = 0.f ; q1.f.y = 0.f ; q1.f.z = 0.f ; q1.f.w = 0.f ; 
        q2.f.x = 0.f ; q2.f.y = 0.f ; q2.f.z = 0.f ; q2.f.w = 0.f ; 
        q3.f.x = 0.f ; q3.f.y = 0.f ; q3.f.z = 0.f ; q3.f.w = 0.f ; 
    }

    QAT4_METHOD void init()
    {
        q0.f.x = 1.f ; q0.f.y = 0.f ; q0.f.z = 0.f ; q0.f.w = 0.f ; 
        q1.f.x = 0.f ; q1.f.y = 1.f ; q1.f.z = 0.f ; q1.f.w = 0.f ; 
        q2.f.x = 0.f ; q2.f.y = 0.f ; q2.f.z = 1.f ; q2.f.w = 0.f ; 
        q3.f.x = 0.f ; q3.f.y = 0.f ; q3.f.z = 0.f ; q3.f.w = 1.f ; 
    }


    // notice that with *right_multiply* the .w column (with identity info) is not used
    QAT4_METHOD float3 right_multiply( const float3& v, const float w ) const 
    { 
        float3 ret;
        ret.x = q0.f.x * v.x + q1.f.x * v.y + q2.f.x * v.z + q3.f.x * w ;
        ret.y = q0.f.y * v.x + q1.f.y * v.y + q2.f.y * v.z + q3.f.y * w ;
        ret.z = q0.f.z * v.x + q1.f.z * v.y + q2.f.z * v.z + q3.f.z * w ;
        return ret;
    }
    QAT4_METHOD void right_multiply_inplace( float4& v, const float w ) const   // v.w is ignored
    { 
        float x = q0.f.x * v.x + q1.f.x * v.y + q2.f.x * v.z + q3.f.x * w ;
        float y = q0.f.y * v.x + q1.f.y * v.y + q2.f.y * v.z + q3.f.y * w ;
        float z = q0.f.z * v.x + q1.f.z * v.y + q2.f.z * v.z + q3.f.z * w ;
        v.x = x ; 
        v.y = y ; 
        v.z = z ; 
    }
    QAT4_METHOD void right_multiply_inplace( float3& v, const float w ) const 
    { 
        float x = q0.f.x * v.x + q1.f.x * v.y + q2.f.x * v.z + q3.f.x * w ;
        float y = q0.f.y * v.x + q1.f.y * v.y + q2.f.y * v.z + q3.f.y * w ;
        float z = q0.f.z * v.x + q1.f.z * v.y + q2.f.z * v.z + q3.f.z * w ;
        v.x = x ; 
        v.y = y ; 
        v.z = z ; 
    }


    /**
    qat4::left_multiply
    ---------------------

    Note that with *left_multiply* the .w column is used. 
    SO MUST CLEAR IDENTITY INFO.

    POTENTIAL FOR SLEEPER BUG HERE AS NORMALLY THE FLOAT 
    VIEW OF INTEGER IDENTITY COLUMNS GIVES VERY SMALL VALUES 
    SO IT WOULD NOT BE VISIBLE

    This gets used less that *right_multiply* but it is 
    used for normals in CSG/intersect_leaf.h 

    **/
    QAT4_METHOD float3 left_multiply( const float3& v, const float w ) const 
    { 
        float3 ret;
        ret.x = q0.f.x * v.x + q0.f.y * v.y + q0.f.z * v.z + q0.f.w * w ;
        ret.y = q1.f.x * v.x + q1.f.y * v.y + q1.f.z * v.z + q1.f.w * w ;
        ret.z = q2.f.x * v.x + q2.f.y * v.y + q2.f.z * v.z + q2.f.w * w ;
        return ret;
    }
    QAT4_METHOD void left_multiply_inplace( float4& v, const float w ) const 
    { 
        float x = q0.f.x * v.x + q0.f.y * v.y + q0.f.z * v.z + q0.f.w * w ;
        float y = q1.f.x * v.x + q1.f.y * v.y + q1.f.z * v.z + q1.f.w * w ;
        float z = q2.f.x * v.x + q2.f.y * v.y + q2.f.z * v.z + q2.f.w * w ;
        v.x = x ; 
        v.y = y ; 
        v.z = z ; 
    }

    /**
    qat4::copy_columns_3x3
    -----------------------

    Canonical usage from CSGOptiX/IAS_Builder::Build 
    Note that the 4th column is not read, so there
    is no need to clear any 4th ".w" column identity info.

    :: 
     
               x  y  z  w 
          q0   0  4  8  - 
          q1   1  5  9  -
          q2   2  6 10  -
          q3   3  7 11  -

    **/ 
    QAT4_METHOD void copy_columns_3x4( float* dst ) const 
    {
         dst[0]  = q0.f.x ; 
         dst[1]  = q1.f.x ; 
         dst[2]  = q2.f.x ; 
         dst[3]  = q3.f.x ; 

         dst[4]  = q0.f.y ; 
         dst[5]  = q1.f.y ; 
         dst[6]  = q2.f.y ; 
         dst[7]  = q3.f.y ; 

         dst[8]  = q0.f.z ; 
         dst[9]  = q1.f.z ; 
         dst[10] = q2.f.z ; 
         dst[11] = q3.f.z ; 
    }

    QAT4_METHOD qat4(const quad6& gs)   // ctor from genstep, needed for TORCH genstep generation 
    {
        q0.f.x = gs.q2.f.x ;  q0.f.y = gs.q2.f.y ;   q0.f.z = gs.q2.f.z ;  q0.f.w = gs.q2.f.w ;   
        q1.f.x = gs.q3.f.x ;  q1.f.y = gs.q3.f.y ;   q1.f.z = gs.q3.f.z ;  q1.f.w = gs.q3.f.w ;   
        q2.f.x = gs.q4.f.x ;  q2.f.y = gs.q4.f.y ;   q2.f.z = gs.q4.f.z ;  q2.f.w = gs.q4.f.w ;   
        q3.f.x = gs.q5.f.x ;  q3.f.y = gs.q5.f.y ;   q3.f.z = gs.q5.f.z ;  q3.f.w = gs.q5.f.w ;   
    } 

    QAT4_METHOD void write( quad6& gs ) const 
    {
         gs.q2.f.x = q0.f.x ; gs.q2.f.y = q0.f.y ; gs.q2.f.z = q0.f.z ; gs.q2.f.w = q0.f.w ; 
         gs.q3.f.x = q1.f.x ; gs.q3.f.y = q1.f.y ; gs.q3.f.z = q1.f.z ; gs.q3.f.w = q1.f.w ; 
         gs.q4.f.x = q2.f.x ; gs.q4.f.y = q2.f.y ; gs.q4.f.z = q2.f.z ; gs.q4.f.w = q2.f.w ; 
         gs.q5.f.x = q3.f.x ; gs.q5.f.y = q3.f.y ; gs.q5.f.z = q3.f.z ; gs.q5.f.w = q3.f.w ; 
    } 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    QAT4_METHOD qat4() 
    {
        q0.f.x = 1.f ;  q0.f.y = 0.f ;   q0.f.z = 0.f ;  q0.f.w = 0.f ;   
        q1.f.x = 0.f ;  q1.f.y = 1.f ;   q1.f.z = 0.f ;  q1.f.w = 0.f ;   
        q2.f.x = 0.f ;  q2.f.y = 0.f ;   q2.f.z = 1.f ;  q2.f.w = 0.f ;   
        q3.f.x = 0.f ;  q3.f.y = 0.f ;   q3.f.z = 0.f ;  q3.f.w = 1.f ;   
    } 

    QAT4_METHOD bool is_identity() const 
    {
        return 
            q0.f.x == 1.f && q0.f.y == 0.f && q0.f.z == 0.f && q0.f.w == 0.f  &&
            q1.f.x == 0.f && q1.f.y == 1.f && q1.f.z == 0.f && q1.f.w == 0.f  && 
            q2.f.x == 0.f && q2.f.y == 0.f && q2.f.z == 1.f && q2.f.w == 0.f  && 
            q3.f.x == 0.f && q3.f.y == 0.f && q3.f.z == 0.f && q3.f.w == 1.f ; 
    }

    QAT4_METHOD bool is_identity(float eps) const 
    {
        return 
            std::abs(q0.f.x-1.f)<eps && std::abs(q0.f.y)    <eps && std::abs(q0.f.z) < eps     && std::abs(q0.f.w) < eps  &&
            std::abs(q1.f.x)<eps     && std::abs(q1.f.y-1.f)<eps && std::abs(q1.f.z) < eps     && std::abs(q1.f.w) < eps  &&
            std::abs(q2.f.x)<eps     && std::abs(q2.f.y)<eps     && std::abs(q2.f.z-1.f) < eps && std::abs(q2.f.w) < eps  &&
            std::abs(q3.f.x)<eps     && std::abs(q3.f.y)<eps     && std::abs(q3.f.z) < eps     && std::abs(q3.f.w-1.f) < eps ;
    }


    QAT4_METHOD bool is_zero() const 
    {
        return 
            q0.f.x == 0.f && q0.f.y == 0.f && q0.f.z == 0.f && q0.f.w == 0.f  &&
            q1.f.x == 0.f && q1.f.y == 0.f && q1.f.z == 0.f && q1.f.w == 0.f  && 
            q2.f.x == 0.f && q2.f.y == 0.f && q2.f.z == 0.f && q2.f.w == 0.f  && 
            q3.f.x == 0.f && q3.f.y == 0.f && q3.f.z == 0.f && q3.f.w == 0.f ; 
    }

    QAT4_METHOD qat4(float tx, float ty, float tz) 
    {
        q0.f.x = 1.f ;  q0.f.y = 0.f ;   q0.f.z = 0.f ;  q0.f.w = 0.f ;   
        q1.f.x = 0.f ;  q1.f.y = 1.f ;   q1.f.z = 0.f ;  q1.f.w = 0.f ;   
        q2.f.x = 0.f ;  q2.f.y = 0.f ;   q2.f.z = 1.f ;  q2.f.w = 0.f ;   
        q3.f.x = tx  ;  q3.f.y = ty  ;   q3.f.z = tz  ;  q3.f.w = 1.f ;   
    } 

    QAT4_METHOD qat4(const float* v) 
    {
        if(v)
        { 
            read(v);        
        }
        else
        {
            init(); 
        }
    } 
    QAT4_METHOD qat4(const double* v) // narrowing 
    {
        if(v)
        { 
            read_narrow(v);        
        }
        else
        {
            init(); 
        }
    } 


    QAT4_METHOD void read(const float* v) 
    {
        q0.f.x = *(v+0)  ;  q0.f.y = *(v+1)  ;   q0.f.z = *(v+2)  ;  q0.f.w = *(v+3) ;   
        q1.f.x = *(v+4)  ;  q1.f.y = *(v+5)  ;   q1.f.z = *(v+6)  ;  q1.f.w = *(v+7) ;   
        q2.f.x = *(v+8)  ;  q2.f.y = *(v+9)  ;   q2.f.z = *(v+10) ;  q2.f.w = *(v+11) ;   
        q3.f.x = *(v+12) ;  q3.f.y = *(v+13) ;   q3.f.z = *(v+14) ;  q3.f.w = *(v+15) ;   
    }


    QAT4_METHOD void read_narrow(const double* v) 
    {
        q0.f.x = float(*(v+0))  ;  q0.f.y = float(*(v+1))  ;   q0.f.z = float(*(v+2))  ;  q0.f.w = float(*(v+3)) ;   
        q1.f.x = float(*(v+4))  ;  q1.f.y = float(*(v+5))  ;   q1.f.z = float(*(v+6))  ;  q1.f.w = float(*(v+7)) ;   
        q2.f.x = float(*(v+8))  ;  q2.f.y = float(*(v+9))  ;   q2.f.z = float(*(v+10)) ;  q2.f.w = float(*(v+11)) ;   
        q3.f.x = float(*(v+12)) ;  q3.f.y = float(*(v+13)) ;   q3.f.z = float(*(v+14)) ;  q3.f.w = float(*(v+15)) ;   
    }


    QAT4_METHOD float* data() 
    {
        return &q0.f.x ;
    }

    QAT4_METHOD const float* cdata() const 
    {
        return &q0.f.x ;
    }

    QAT4_METHOD qat4* copy() const 
    {
        return new qat4(cdata()); 
    }

    static QAT4_METHOD qat4* identity()
    {
        qat4* q = new qat4 ; 
        q->init() ; 
        return q ; 
    }

    static QAT4_METHOD void copy(qat4& b, const qat4& a )
    {
        b.q0.f.x = a.q0.f.x ; b.q0.f.y = a.q0.f.y ; b.q0.f.z = a.q0.f.z ; b.q0.f.w = a.q0.f.w ; 
        b.q1.f.x = a.q1.f.x ; b.q1.f.y = a.q1.f.y ; b.q1.f.z = a.q1.f.z ; b.q1.f.w = a.q1.f.w ; 
        b.q2.f.x = a.q2.f.x ; b.q2.f.y = a.q2.f.y ; b.q2.f.z = a.q2.f.z ; b.q2.f.w = a.q2.f.w ; 
        b.q3.f.x = a.q3.f.x ; b.q3.f.y = a.q3.f.y ; b.q3.f.z = a.q3.f.z ; b.q3.f.w = a.q3.f.w ; 
    }

    /**


            a.q0.f.x   a.q0.f.y   a.q0.f.z   a.q0.f.w         b.q0.f.x   b.q0.f.y   b.q0.f.z   b.q0.f.w  

            a.q1.f.x   a.q1.f.y   a.q1.f.z   a.q1.f.w         b.q1.f.x   b.q1.f.y   b.q1.f.z   b.q1.f.w  

            a.q2.f.x   a.q2.f.y   a.q2.f.z   a.q2.f.w         b.q2.f.x   b.q2.f.y   b.q2.f.z   b.q2.f.w  

            a.q3.f.x   a.q3.f.y   a.q3.f.z   a.q3.f.w         b.q3.f.x   b.q3.f.y   b.q3.f.z   b.q3.f.w  

    **/

    QAT4_METHOD qat4(const qat4& a_, const qat4& b_ , bool flip)  // flip=true matches glm A*B  
    { 
        const qat4& a = flip ? b_ : a_ ;  
        const qat4& b = flip ? a_ : b_ ;  
 
        q0.f.x = a.q0.f.x * b.q0.f.x  +  a.q0.f.y * b.q1.f.x  +  a.q0.f.z * b.q2.f.x  + a.q0.f.w * b.q3.f.x ; 
        q0.f.y = a.q0.f.x * b.q0.f.y  +  a.q0.f.y * b.q1.f.y  +  a.q0.f.z * b.q2.f.y  + a.q0.f.w * b.q3.f.y ; 
        q0.f.z = a.q0.f.x * b.q0.f.z  +  a.q0.f.y * b.q1.f.z  +  a.q0.f.z * b.q2.f.z  + a.q0.f.w * b.q3.f.z ; 
        q0.f.w = a.q0.f.x * b.q0.f.w  +  a.q0.f.y * b.q1.f.w  +  a.q0.f.z * b.q2.f.w  + a.q0.f.w * b.q3.f.w ; 

        q1.f.x = a.q1.f.x * b.q0.f.x  +  a.q1.f.y * b.q1.f.x  +  a.q1.f.z * b.q2.f.x  + a.q1.f.w * b.q3.f.x ; 
        q1.f.y = a.q1.f.x * b.q0.f.y  +  a.q1.f.y * b.q1.f.y  +  a.q1.f.z * b.q2.f.y  + a.q1.f.w * b.q3.f.y ; 
        q1.f.z = a.q1.f.x * b.q0.f.z  +  a.q1.f.y * b.q1.f.z  +  a.q1.f.z * b.q2.f.z  + a.q1.f.w * b.q3.f.z ; 
        q1.f.w = a.q1.f.x * b.q0.f.w  +  a.q1.f.y * b.q1.f.w  +  a.q1.f.z * b.q2.f.w  + a.q1.f.w * b.q3.f.w ; 

        q2.f.x = a.q2.f.x * b.q0.f.x  +  a.q2.f.y * b.q1.f.x  +  a.q2.f.z * b.q2.f.x  + a.q2.f.w * b.q3.f.x ; 
        q2.f.y = a.q2.f.x * b.q0.f.y  +  a.q2.f.y * b.q1.f.y  +  a.q2.f.z * b.q2.f.y  + a.q2.f.w * b.q3.f.y ; 
        q2.f.z = a.q2.f.x * b.q0.f.z  +  a.q2.f.y * b.q1.f.z  +  a.q2.f.z * b.q2.f.z  + a.q2.f.w * b.q3.f.z ; 
        q2.f.w = a.q2.f.x * b.q0.f.w  +  a.q2.f.y * b.q1.f.w  +  a.q2.f.z * b.q2.f.w  + a.q2.f.w * b.q3.f.w ; 

        q3.f.x = a.q3.f.x * b.q0.f.x  +  a.q3.f.y * b.q1.f.x  +  a.q3.f.z * b.q2.f.x  + a.q3.f.w * b.q3.f.x ; 
        q3.f.y = a.q3.f.x * b.q0.f.y  +  a.q3.f.y * b.q1.f.y  +  a.q3.f.z * b.q2.f.y  + a.q3.f.w * b.q3.f.y ; 
        q3.f.z = a.q3.f.x * b.q0.f.z  +  a.q3.f.y * b.q1.f.z  +  a.q3.f.z * b.q2.f.z  + a.q3.f.w * b.q3.f.z ; 
        q3.f.w = a.q3.f.x * b.q0.f.w  +  a.q3.f.y * b.q1.f.w  +  a.q3.f.z * b.q2.f.w  + a.q3.f.w * b.q3.f.w ; 
    }


    // .w column looks to be in use : potential bug ? : TODO: compare with longhand approach 
    QAT4_METHOD void transform_aabb_inplace( float* aabb ) const 
    {
        float4 xa = q0.f * *(aabb+0) ; 
        float4 xb = q0.f * *(aabb+3) ;
        float4 xmi = fminf(xa, xb);
        float4 xma = fmaxf(xa, xb);

        float4 ya = q1.f * *(aabb+1) ; 
        float4 yb = q1.f * *(aabb+4) ;
        float4 ymi = fminf(ya, yb);
        float4 yma = fmaxf(ya, yb);

        float4 za = q2.f * *(aabb+2) ; 
        float4 zb = q2.f * *(aabb+5) ;
        float4 zmi = fminf(za, zb);
        float4 zma = fmaxf(za, zb);

        float4 tmi = xmi + ymi + zmi + q3.f ; 
        float4 tma = xma + yma + zma + q3.f ; 

        *(aabb + 0) = tmi.x ; 
        *(aabb + 1) = tmi.y ; 
        *(aabb + 2) = tmi.z ; 
        *(aabb + 3) = tma.x ; 
        *(aabb + 4) = tma.y ; 
        *(aabb + 5) = tma.z ; 
    }

    QAT4_METHOD void getIdentity(int& ins_idx, int& gas_idx, int& sensor_identifier, int& sensor_index ) const 
    {
        ins_idx           = q0.i.w ;   // formerly used unsigbed and "- 1" 
        gas_idx           = q1.i.w ; 
        sensor_identifier = q2.i.w ; 
        sensor_index      = q3.i.w  ; 
    }

    /**
    sqat4::get_IAS_OptixInstance_instanceId
    ----------------------------------------

    Canonical use by IAS_Builder::CollectInstances

    * July 2023 : QPMT needs lpmtid GPU side so changing from ins_idx to sensor_identifier
    * HMM: there are -1 in there, which as unsigned becomes  ~0u 

    YEP: that ~0u may be cause of notes/issues/unspecified_launch_failure_with_simpleLArTPC.rst

    **/

    QAT4_METHOD int get_IAS_OptixInstance_instanceId() const 
    {
        //const unsigned& ins_idx = q0.u.w ;  
        //return ins_idx ; 
        const int& sensor_identifier = q2.i.w ; 
        assert( sensor_identifier >= 0 );  // 0 means not a sensor GPU side, so subtract 1 to get actual sensorId
        return sensor_identifier ; 
    }

    /**
    sqat4::setIdentity
    -------------------

    Canonical usage from CSGFoundry::addInstance  where sensor_identifier gets +1 
    with 0 meaning not a sensor. 
    **/ 

    QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier_1, int sensor_index )
    {
        assert( sensor_identifier_1 >= 0 ); 

        q0.i.w = ins_idx ;             // formerly unsigned and "+ 1"
        q1.i.w = gas_idx ; 
        q2.i.w = sensor_identifier_1 ;   // now +1 with 0 meaning not-a-sensor 
        q3.i.w = sensor_index ; 
    }

    /**
    sqat4::incrementSensorIdentifier
    ---------------------------------

    Canonical usage from CSGFoundry::addInstanceVector to adjust from 

    * CPU -1:not-a-sensor 
    * GPU  0:not-a-sensor

    **/
    QAT4_METHOD void incrementSensorIdentifier()
    {
        assert( q2.i.w >= -1 ); 
        q2.i.w += 1 ; 
        assert( q2.i.w >=  0 ); 
    }


    QAT4_METHOD void clearIdentity() // prepare for matrix multiply by clearing any auxiliary info in the "spare" 4th column 
    {
        q0.f.w = 0.f ; 
        q1.f.w = 0.f ; 
        q2.f.w = 0.f ; 
        q3.f.w = 1.f ; 
    }

    static QAT4_METHOD bool IsDiff( const qat4& a, const qat4& b )
    {
        return false ; 
    }

    // collects unique ins/gas/ias indices found in qv vector of instances
    static QAT4_METHOD void find_unique(const std::vector<qat4>& qv, 
              std::vector<int>& ins, 
              std::vector<int>& gas, 
              std::vector<int>& s_ident, 
              std::vector<int>& s_index )
    {
         for(unsigned i=0 ; i < qv.size() ; i++)
         {
             const qat4& q = qv[i] ; 
             int ins_idx,  gas_idx, sensor_identifier, sensor_index ; 
             q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  

             if(std::find(ins.begin(), ins.end(), ins_idx) == ins.end() ) ins.push_back(ins_idx); 
             if(std::find(gas.begin(), gas.end(), gas_idx) == gas.end() ) gas.push_back(gas_idx); 
             if(std::find(s_ident.begin(), s_ident.end(), sensor_identifier) == s_ident.end() ) s_ident.push_back(sensor_identifier); 
             if(std::find(s_index.begin(), s_index.end(), sensor_index)      == s_index.end() ) s_index.push_back(sensor_index); 

         }
    } 

    static QAT4_METHOD void find_unique_gas(const std::vector<qat4>& qv, std::vector<int>& gas )
    {
         for(unsigned i=0 ; i < qv.size() ; i++)
         {
             const qat4& q = qv[i] ; 
             int ins_idx,  gas_idx, sensor_identifier, sensor_index ; 
             q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
             if(std::find(gas.begin(), gas.end(), gas_idx) == gas.end() ) gas.push_back(gas_idx); 
         }
    } 


    // count the number of instances with the provided ias_idx, that are among the emm if that is non-zero 
    static QAT4_METHOD unsigned count_ias( const std::vector<qat4>& qv , int /*ias_idx*/, unsigned long long emm)
    {
        unsigned count = 0 ; 
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            int ins_idx,  gas_idx, sensor_identifier, sensor_index  ; 
            q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
            unsigned long long gas_idx_ull = gas_idx ; 
            bool gas_enabled = emm == 0ull ? true : emm & ( 0x1ull << gas_idx_ull ) ; 
            if( gas_enabled ) count += 1 ;
        }
        return count ; 
    }
    // select instances with the provided ias_idx, that are among the emm if that is non-zero,  ordered as they are found
    static QAT4_METHOD void select_instances_ias(const std::vector<qat4>& qv, std::vector<qat4>& select_qv, int /*ias_idx*/, unsigned long long emm  )
    {
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            int ins_idx,  gas_idx, sensor_identifier, sensor_index  ; 
            q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
            unsigned long long gas_idx_ull = gas_idx ; 
            bool gas_enabled = emm == 0ull ? true : emm & ( 0x1ull << gas_idx_ull ) ; 
            if( gas_enabled ) select_qv.push_back(q) ;
        }
    }


    // count the number of instances with the provided gas_idx 
    static QAT4_METHOD unsigned count_gas( const std::vector<qat4>& qv , int gas_idx_ )
    {
        unsigned count = 0 ; 
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            int ins_idx,  gas_idx, sensor_identifier, sensor_index  ; 
            q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
            if( gas_idx_ == gas_idx ) count += 1 ;
        }
        return count ; 
    }
    // select instances with the provided gas_idx, ordered as they are found
    static QAT4_METHOD void select_instances_gas(const std::vector<qat4>& qv, std::vector<qat4>& select_qv, int gas_idx_ )
    {
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            int ins_idx,  gas_idx, sensor_identifier, sensor_index  ; 
            q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
            if( gas_idx_ == gas_idx ) select_qv.push_back(q) ;
        }
    }

    // select instance pointers with the provided gas_idx, ordered as they are found
    static QAT4_METHOD void select_instance_pointers_gas(const std::vector<qat4>& qv, std::vector<const qat4*>& select_qi, int gas_idx_ )
    {
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4* qi = qv.data() + i ; 
            int ins_idx,  gas_idx, sensor_identifier, sensor_index  ; 
            qi->getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
            if( gas_idx_ == gas_idx ) select_qi.push_back(qi) ;
        }
    }

    // return index of the ordinal-th instance with the provided gas_idx or -1 if not found
    static QAT4_METHOD int find_instance_gas(const std::vector<qat4>& qv, int gas_idx_, unsigned ordinal  )
    {
        unsigned count = 0 ;
        int index = -1 ;  
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            int ins_idx,  gas_idx, sensor_identifier, sensor_index  ; 
            q.getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  
            if( gas_idx_ == gas_idx )
            {
                if( count == ordinal ) index = i  ; 
                count += 1 ; 
            }
        }
        return index ;  
    }

    QAT4_METHOD void right_multiply_inplace( std::vector<float3>& points, const float w ) const 
    {
        for(int i=0 ; i < int(points.size()) ; i++)
        {
            float3& p = points[i] ;
            right_multiply_inplace( p, w ); 
        } 
    }


    QAT4_METHOD float element(unsigned j, unsigned k) const
    {
        const float4& f = j == 0 ? q0.f : ( j == 1 ? q1.f : ( j == 2 ? q2.f : q3.f ) ) ;  
        return   k == 0  ? f.x : ( k == 1 ? f.y : ( k == 2 ? f.z : f.w )) ; 
    }

    QAT4_METHOD void add_translate(float tx, float ty, float tz) 
    {
        q3.f.x += tx ; 
        q3.f.y += ty ; 
        q3.f.z += tz ; 
    }


    static QAT4_METHOD int compare(const qat4& a, const qat4& b, float epsilon=1e-6 ) 
    {
        int rc = 0 ; 
        if( std::abs(a.q0.f.x - b.q0.f.x) > epsilon ) rc += 1 ;  
        if( std::abs(a.q0.f.y - b.q0.f.y) > epsilon ) rc += 1 ;  
        if( std::abs(a.q0.f.z - b.q0.f.z) > epsilon ) rc += 1 ;  
        if( std::abs(a.q0.f.w - b.q0.f.w) > epsilon ) rc += 1 ;  

        if( std::abs(a.q1.f.x - b.q1.f.x) > epsilon ) rc += 1 ;  
        if( std::abs(a.q1.f.y - b.q1.f.y) > epsilon ) rc += 1 ;  
        if( std::abs(a.q1.f.z - b.q1.f.z) > epsilon ) rc += 1 ;  
        if( std::abs(a.q1.f.w - b.q1.f.w) > epsilon ) rc += 1 ;  

        if( std::abs(a.q2.f.x - b.q2.f.x) > epsilon ) rc += 1 ;  
        if( std::abs(a.q2.f.y - b.q2.f.y) > epsilon ) rc += 1 ;  
        if( std::abs(a.q2.f.z - b.q2.f.z) > epsilon ) rc += 1 ;  
        if( std::abs(a.q2.f.w - b.q2.f.w) > epsilon ) rc += 1 ;  

        if( std::abs(a.q3.f.x - b.q3.f.x) > epsilon ) rc += 1 ;  
        if( std::abs(a.q3.f.y - b.q3.f.y) > epsilon ) rc += 1 ;  
        if( std::abs(a.q3.f.z - b.q3.f.z) > epsilon ) rc += 1 ;  
        if( std::abs(a.q3.f.w - b.q3.f.w) > epsilon ) rc += 1 ;  

        return rc ; 
    }


    static QAT4_METHOD qat4* from_string(const char* s0, const char* replace="[]()," )
    {
        char* s1 = strdup(s0); 
        for(unsigned i=0 ; i < strlen(s1) ; i++) if(strchr(replace, s1[i]) != nullptr) s1[i] = ' ' ;   

        std::stringstream ss(s1);  
        std::string s ; 
        std::vector<float> vv ; 
        float v ; 

        while(std::getline(ss, s, ' '))
        {   
            if(strlen(s.c_str()) == 0 ) continue;  
            std::stringstream tt(s);  
            tt >> v ; 
            vv.push_back(v); 
        }   

        unsigned num_vv = vv.size() ; 
        qat4* q = nullptr ; 

        if( num_vv == 16 ) 
        {
            q = new qat4(vv.data()) ; 
        }
        else if( num_vv == 3 || num_vv == 4 ) 
        {
            float tx = vv[0] ; 
            float ty = vv[1] ; 
            float tz = vv[2] ; 

            q = new qat4(tx, ty, tz) ; 
        } 
        return q ; 
    }  


    QAT4_METHOD std::string desc(char mat='t', unsigned wid=8, unsigned prec=3, bool id=true) const 
    {
        std::stringstream ss ; 
        ss << desc_(mat, wid, prec) 
           << " " 
           << ( id ? descId() : "" ) 
           ; 
        std::string s = ss.str() ; 
        return s ; 
    }
    QAT4_METHOD std::string desc_(char mat='t', unsigned wid=8, unsigned prec=3) const 
    {
        std::stringstream ss ; 
        ss << mat << ":" ; 
        for(int j=0 ; j < 4 ; j++ ) 
        {
            ss << "[" ;
            for(int k=0 ; k < 4 ; k++ ) ss << std::setw(wid) << std::fixed << std::setprecision(prec) << element(j,k) << " " ; 
            ss << "]" ; 
        }
        std::string s = ss.str() ; 
        return s ; 
    }

    QAT4_METHOD std::string descId() const 
    {
        int ins_idx,  gas_idx, sensor_identifier, sensor_index ; 
        getIdentity(ins_idx,  gas_idx, sensor_identifier, sensor_index );  

        std::stringstream ss ; 
        ss
            << "( i/g/si/sx " 
            << std::setw(7) << ins_idx  
            << std::setw(3) << gas_idx  
            << std::setw(7) << sensor_identifier
            << std::setw(5) << ( sensor_index == 1065353216 ? -1 : sensor_index )  // 1065353216 is 1.f viewed as int 
            << " )"
            ;

        std::string s = ss.str() ; 
        return s ; 
    }

    static QAT4_METHOD void dump(const std::vector<qat4>& qv);

#endif

}; 
   


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline std::ostream& operator<<(std::ostream& os, const qat4& v) 
{
    os 
       << v.q0.f  
       << v.q1.f  
       << v.q2.f  
       << v.q3.f
       ;
    return os; 
}
#endif 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

QAT4_FUNCTION void qat4::dump(const std::vector<qat4>& qv)
{
    for(unsigned i=0 ; i < qv.size() ; i++)
    {
        const qat4& q = qv[i] ; 
        std::cout 
            << " i " << std::setw(4) << i  
            << " " << q.descId() 
            << " " << q.desc() 
            << std::endl 
            ;

    }
}

#endif 

