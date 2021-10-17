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


    // notice that with *right_multiply* the .w column (with identity info) is not used
    QAT4_METHOD float3 right_multiply( const float3& v, const float w ) const 
    { 
        float3 ret;
        ret.x = q0.f.x * v.x + q1.f.x * v.y + q2.f.x * v.z + q3.f.x * w ;
        ret.y = q0.f.y * v.x + q1.f.y * v.y + q2.f.y * v.z + q3.f.y * w ;
        ret.z = q0.f.z * v.x + q1.f.z * v.y + q2.f.z * v.z + q3.f.z * w ;
        return ret;
    }
    QAT4_METHOD void right_multiply_inplace( float4& v, const float w ) const 
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


    // with *left_multiply* the .w column is used
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


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    QAT4_METHOD qat4() 
    {
        q0.f.x = 1.f ;  q0.f.y = 0.f ;   q0.f.z = 0.f ;  q0.f.w = 0.f ;   
        q1.f.x = 0.f ;  q1.f.y = 1.f ;   q1.f.z = 0.f ;  q1.f.w = 0.f ;   
        q2.f.x = 0.f ;  q2.f.y = 0.f ;   q2.f.z = 1.f ;  q2.f.w = 0.f ;   
        q3.f.x = 0.f ;  q3.f.y = 0.f ;   q3.f.z = 0.f ;  q3.f.w = 1.f ;   
    } 
    QAT4_METHOD qat4(const float* v) 
    {
        q0.f.x = *(v+0)  ;  q0.f.y = *(v+1)  ;   q0.f.z = *(v+2)  ;  q0.f.w = *(v+3) ;   
        q1.f.x = *(v+4)  ;  q1.f.y = *(v+5)  ;   q1.f.z = *(v+6)  ;  q1.f.w = *(v+7) ;   
        q2.f.x = *(v+8)  ;  q2.f.y = *(v+9)  ;   q2.f.z = *(v+10) ;  q2.f.w = *(v+11) ;   
        q3.f.x = *(v+12) ;  q3.f.y = *(v+13) ;   q3.f.z = *(v+14) ;  q3.f.w = *(v+15) ;   
    } 
    QAT4_METHOD qat4(const double* v) // narrowing 
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

    QAT4_METHOD void getIdentity(unsigned& ins_idx, unsigned& gas_idx, unsigned& ias_idx ) const 
    {
        ins_idx = q0.u.w - 1u ; 
        gas_idx = q1.u.w - 1u ; 
        ias_idx = q2.u.w - 1u ; 
    }
    QAT4_METHOD void setIdentity(unsigned ins_idx, unsigned gas_idx, unsigned ias_idx )
    {
        q0.u.w = ins_idx + 1u ; 
        q1.u.w = gas_idx + 1u ; 
        q2.u.w = ias_idx + 1u ; 
    }
    QAT4_METHOD void clearIdentity() // prepare for matrix multiply by clearing any auxiliary info in the "spare" 4th column 
    {
        q0.f.w = 0.f ; 
        q1.f.w = 0.f ; 
        q2.f.w = 0.f ; 
        q3.f.w = 1.f ; 
    }

    // collects unique ins/gas/ias indices found in qv vector of instances
    static QAT4_METHOD void find_unique(const std::vector<qat4>& qv, std::vector<unsigned>& ins, std::vector<unsigned>& gas, std::vector<unsigned>& ias) 
    {
         for(unsigned i=0 ; i < qv.size() ; i++)
         {
             const qat4& q = qv[i] ; 
             unsigned ins_idx,  gas_idx, ias_idx ; 
             q.getIdentity(ins_idx,  gas_idx, ias_idx);  
             if(std::find(ins.begin(), ins.end(), ins_idx) == ins.end() ) ins.push_back(ins_idx); 
             if(std::find(gas.begin(), gas.end(), gas_idx) == gas.end() ) gas.push_back(gas_idx); 
             if(std::find(ias.begin(), ias.end(), ias_idx) == ias.end() ) ias.push_back(ias_idx); 
         }
    } 

    // count the number of instances with the provided ias_idx, that are among the emm if that is non-zero 
    static QAT4_METHOD unsigned count_ias( const std::vector<qat4>& qv , unsigned ias_idx_, unsigned long long emm)
    {
        unsigned count = 0 ; 
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            unsigned ins_idx,  gas_idx, ias_idx ; 
            q.getIdentity(ins_idx,  gas_idx, ias_idx);  
            unsigned long long gas_idx_ull = gas_idx ; 
            bool gas_enabled = emm == 0ull ? true : emm & ( 0x1ull << gas_idx_ull ) ; 
            if( ias_idx_ == ias_idx && gas_enabled ) count += 1 ;
        }
        return count ; 
    }
    // select instances with the provided ias_idx, that are among the emm if that is non-zero,  ordered as they are found
    static QAT4_METHOD void select_instances_ias(const std::vector<qat4>& qv, std::vector<qat4>& select_qv, unsigned ias_idx_, unsigned long long emm  )
    {
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            unsigned ins_idx,  gas_idx, ias_idx ; 
            q.getIdentity(ins_idx,  gas_idx, ias_idx );  
            unsigned long long gas_idx_ull = gas_idx ; 
            bool gas_enabled = emm == 0ull ? true : emm & ( 0x1ull << gas_idx_ull ) ; 
            if( ias_idx_ == ias_idx && gas_enabled ) select_qv.push_back(q) ;
        }
    }


    // count the number of instances with the provided gas_idx 
    static QAT4_METHOD unsigned count_gas( const std::vector<qat4>& qv , unsigned gas_idx_ )
    {
        unsigned count = 0 ; 
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            unsigned ins_idx,  gas_idx, ias_idx ; 
            q.getIdentity(ins_idx,  gas_idx, ias_idx);  
            if( gas_idx_ == gas_idx ) count += 1 ;
        }
        return count ; 
    }
    // select instances with the provided gas_idx, ordered as they are found
    static QAT4_METHOD void select_instances_gas(const std::vector<qat4>& qv, std::vector<qat4>& select_qv, unsigned gas_idx_ )
    {
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            unsigned ins_idx,  gas_idx, ias_idx ; 
            q.getIdentity(ins_idx,  gas_idx, ias_idx );  
            if( gas_idx_ == gas_idx ) select_qv.push_back(q) ;
        }
    }

    // return index of the ordinal-th instance with the provided gas_idx or -1 if not found
    static QAT4_METHOD int find_instance_gas(const std::vector<qat4>& qv, unsigned gas_idx_, unsigned ordinal  )
    {
        unsigned count = 0 ;
        int index = -1 ;  
        for(unsigned i=0 ; i < qv.size() ; i++)
        {
            const qat4& q = qv[i] ; 
            unsigned ins_idx,  gas_idx, ias_idx ; 
            q.getIdentity(ins_idx,  gas_idx, ias_idx );  
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

    static QAT4_METHOD qat4* from_string(const char* s0, const char* repl="[]()," )
    {
        char* s1 = strdup(s0); 
        for(unsigned i=0 ; i < strlen(s1) ; i++) if(strchr(repl, s1[i]) != nullptr) s1[i] = ' ' ;   

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
        return vv.size() == 16 ? new qat4(vv.data()) : nullptr ; 
    }  


    QAT4_METHOD std::string desc(char mat, unsigned wid=8, unsigned prec=3) const 
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
        unsigned ins_idx,  gas_idx, ias_idx ; 
        q.getIdentity(ins_idx,  gas_idx, ias_idx );  

        std::cout 
            << " i " << std::setw(4) << i  
            << " ins " << std::setw(7) << ins_idx  
            << " gas " << std::setw(3) << gas_idx  
            << " ias " << std::setw(2) << ias_idx  
            << " " << q << std::endl 
            ;

    }
}

#endif 

