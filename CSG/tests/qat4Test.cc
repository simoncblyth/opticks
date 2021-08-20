// ./qat4Test.sh 

#include "sutil_vec_math.h"

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#include "qat4.h"
#include "AABB.h"

#include <glm/mat4x4.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include <glm/gtc/random.hpp>

const float EPSILON = 1e-4 ; 


bool check( const float& a, const float& b,  const char* msg  )
{
    float cf = std::abs(a - b) ; 
    bool chk = cf < EPSILON ; 
    if(!chk) 
        std::cout 
            << msg
            << " cf "
            << std::fixed << std::setprecision(8) << cf
            << " ( "
            << std::fixed << std::setprecision(8) << a 
            << "," 
            << std::fixed << std::setprecision(8) << b 
            << ") "
            << std::endl 
            ; 
     return chk ; 
}

void check( const glm::vec4& a , const glm::vec4& b, const char* msg )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z") && check(a.w,b.w,"w") ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}
void check( const float3& a , const float3& b, const char* msg )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z")  ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}
void check( const float3& a , const glm::vec4& b, const char* msg )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z")  ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}
void check( const glm::vec4& a , const float3& b, const char* msg  )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z") ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}




struct points
{
    std::vector<std::string> vn ; 
    std::vector<glm::vec3>   vv ; 
    std::vector<float3>      ff ; 

    void add(float x, float y, float z, const char* label )
    {
        ff.push_back(make_float3(x,y,z));  
        vv.push_back(glm::vec3(x,y,z)) ;
        vn.push_back(label); 
    } 

    void dump(const char* msg)
    {
        for(unsigned i=0 ; i < vn.size() ; i++)
        {
            std::cout 
                << vn[i] 
                << " vv: " << glm::to_string(vv[i]) 
                << " ff: " << ff[i] 
                << std::endl 
                ; 
        }
    }


    void test_multiply( const float* d )
    {
        dump16(d, "d:"); 

        glm::mat4 g = glm::make_mat4(d); 
        glm::mat4 t = glm::transpose(g); 
        std::cout << "g " << glm::to_string(g) << std::endl ;  
        std::cout << "t " << glm::to_string(t) << std::endl ;  

        qat4 q(glm::value_ptr(g)); 
        qat4 r(glm::value_ptr(t));       // r is tranposed(q)
        std::cout << "q " << q << std::endl ;  
        std::cout << "r " << r << std::endl ;  

        float w = 1.f ; 

        for(unsigned i=0 ; i < vn.size() ; i++)
        {
            const glm::vec3& v = vv[i] ; 
            const float3&    f = ff[i] ; 

            glm::vec4 v4( v.x, v.y, v.z, w ); 


            //            mat*vec
            float3    qf = q.right_multiply(f, w) ;   
            glm::vec4 gv = g * v4 ; 

            check(qf, gv, "qf == gv : glm/qat consistency for mat*vec "); 

            //            vec*transposed(mat)
            float3    fr = r.left_multiply(f, w) ; 
            glm::vec4 vt = v4 * t ; 

            check(fr, vt, "fr == vt : glm/qat consisency of vec*transposed(mat)"); 
            check(qf, fr, "qf == fr : qat consistency  mat*vec == vec*transposed(mat) ");    
            check(gv, vt, "gv == vt : glm consistency  mat*vec == vec*transposed(mat)");    

            //            vec*mat
            float3    fq = q.left_multiply(f, w) ; 
            glm::vec4 vg = v4 * g ; 

            check(fq, vg, "fq == vg : glm/qat consisency of vec*mat"); 

            //transposed(mat)*vec  
            float3    rf = r.right_multiply(f, w); 
            glm::vec4 tv = t * v4 ; 

            check(rf, tv,  "rf == tv : glm/qat consistency  vec*mat == transposed(mat)*vec ");    
            check(fq, rf,  "fq == rf : qat consistency      vec*mat == transposed(mat)*vec ");
            check(vg, tv,  "vg == tv : glm consistency      vec*mat == transposted(mat)*vec " );  


            std::cout 
                << vn[i] 
                << " v: " << glm::to_string(v) 
                << " gv: " << glm::to_string(gv)
                << " qf: " << qf
                << std::endl 
                ; 

            std::cout 
                << vn[i] 
                << " v: " << glm::to_string(v) 
                << " vg: " << glm::to_string(vg)
                << " fq: " << fq
                << std::endl 
                ; 

        }
    }



    void dump16( const float* f, const char* label)
    {
        std::cout << label << " " ; 
        for(unsigned i=0 ; i < 16 ; i++ ) std::cout << *(f+i) << " " ; 
        std::cout << std::endl; 
    }
};

glm::mat4 make_transform(float tx, float ty, float tz, float sx, float sy, float sz, float ax, float ay, float az, float degrees )
{
    float radians = (degrees/180.f)*glm::pi<float>() ; 

    glm::mat4 m(1.f); 
    m = glm::translate(  m, glm::vec3(tx, ty, tz) );             std::cout << " t " << glm::to_string(m) << std::endl ;  
    m = glm::rotate(     m, radians, glm::vec3(ax, ay, az)  );   std::cout << " rt " << glm::to_string(m) << std::endl ;  
    m = glm::scale(      m, glm::vec3(sx, sy, sz) );             std::cout << " srt " << glm::to_string(m) << std::endl ;

    return m ;
}

glm::mat4 make_transform_0()
{
    float tx = 100.f ;  
    float ty = 200.f ;  
    float tz = 300.f ;  

    float sx = 10.f ;  
    float sy = 10.f ;  
    float sz = 10.f ;  

    float ax = 0.f ;  
    float ay = 0.f ;  
    float az = 1.f ;  

    float degrees = 45.0 ; 

    glm::mat4 srt = make_transform(tx, ty, tz, sx, sy, sz, ax, ay, az, degrees); 
    return srt ; 
}

glm::mat4 make_transform_1()
{
    glm::vec3 t = glm::sphericalRand(20000.f); 
    glm::vec3 a = glm::sphericalRand(1.f); 
    //glm::vec3 s(  glm::linearRand(1.f, 100.f), glm::linearRand(1.f, 100.f), glm::linearRand(1.f, 100.f) );  
    glm::vec3 s(1.f,1.f,1.f );  
    float degrees = glm::linearRand(0.f, 360.f ); 
    glm::mat4 srt = make_transform(t.x, t.y, t.z, s.x, s.y, s.z, a.x, a.y, a.z, degrees); 
    return srt ; 
}


void test_multiply()
{
    points p ; 
    p.add(0.f, 0.f, 0.f, "po"); 
    p.add(1.f, 0.f, 0.f, "px"); 
    p.add(0.f, 1.f, 0.f, "py"); 
    p.add(0.f, 0.f, 1.f, "pz"); 

    for(unsigned i=0 ; i < 100 ; i++)
    {
        glm::vec3 r = glm::sphericalRand(20000.f); 
        p.add( r.x, r.y, r.z, "r" ); 
    } 
    p.dump("points.p"); 
    glm::mat4 t0 = make_transform_0(); 
    p.test_multiply( glm::value_ptr(t0) ); 

/*
    for(unsigned i=0 ; i < 100 ; i++)
    {
        glm::mat4 t1 = make_transform_1(); 
        p.test_multiply( glm::value_ptr(t1) ); 
    }
*/
}

void test_multiply_inplace()
{
    glm::mat4 m(1.f); 
    glm::vec3 s(1.f, 1.f, 1.f) ; 

    m = glm::scale(m, s ); 
    std::cout << glm::to_string( m) << std::endl ; 

    qat4 q(glm::value_ptr(m)); 
    float4 isect = make_float4(10.f, 10.f, 10.f, 42.f ); 
    q.right_multiply_inplace( isect, 0.f ) ;
    printf("isect: (%10.4f, %10.4f, %10.4f, %10.4f) \n", isect.x, isect.y, isect.z, isect.w ); 
}



/*
inline std::ostream& operator<<(std::ostream& os, const AABB& v)
{
    os << " mn " << v.mn  << " mx " << v.mx ;
    return os; 
}
*/

void test_transform_aabb_inplace()
{
    float tx = 100.f ; 
    float ty = 200.f ; 
    float tz = 300.f ; 

    glm::mat4 m(1.f); 
    m = glm::translate(m, glm::vec3(tx, ty, tz)); 

    qat4 q(glm::value_ptr(m)); 
    std::cout << q << std::endl ; 

    AABB aabb = { -100.f, -100.f, -100.f, 100.f, 100.f, 100.f } ; 
    std::cout << "aabb " << aabb << std::endl ; 
    q.transform_aabb_inplace((float*)&aabb);
    std::cout << "aabb " << aabb << std::endl ; 
}


std::string desc(const char* label, const std::vector<unsigned>& uu )
{
    std::stringstream ss ; 
    ss 
       << std::setw(10) << label
       << std::setw(4) << uu.size()
       << " ( "
       ;

    for(int i=0 ; i < int(uu.size()) ; i++) ss << uu[i] << " "  ; 
    ss << " ) " ; 
    std::string s = ss.str(); 
    return s ; 
}


void test_find_unique()
{
    std::vector<qat4> qq ; 
 
    qat4 a ; a.setIdentity( 1, 10, 100 ); 
    qat4 b ; b.setIdentity( 2, 20, 200 ); 
    qat4 c ; c.setIdentity( 3, 30, 300 ); 
    qat4 d ; d.setIdentity( 4, 40, 400 ); 

    qq.push_back(a) ;

    qq.push_back(b) ;
    qq.push_back(b) ;

    qq.push_back(c) ;
    qq.push_back(c) ;
    qq.push_back(c) ;

    qq.push_back(d) ;
    qq.push_back(d) ;
    qq.push_back(d) ;
    qq.push_back(d) ;


    std::vector<unsigned> ins, gas, ias ; 
    qat4::find_unique(qq, ins, gas, ias ); 
   
    std::cout 
        << desc("ins", ins ) << std::endl  
        << desc("gas", gas ) << std::endl  
        << desc("ias", ias ) << std::endl  
        ;


    unsigned long long emm = 0ull ; 
    for(unsigned i=0 ; i < ias.size() ; i++)
    {
        unsigned ias_idx = ias[i]; 
        unsigned num_ins = qat4::count_ias(qq, ias_idx, emm ); 
        std::cout 
            << " ias_idx " << std::setw(3) << ias_idx
            << " num_ins " << std::setw(3) << num_ins
            << std::endl
            ;
    }
}


void test_right_multiply_translate()
{
    float3 t = make_float3( 100.f, 200.f, 300.f ); 

    qat4 q ; 
    q.q3.f.x = t.x ; 
    q.q3.f.y = t.y ; 
    q.q3.f.z = t.z ; 

    q.setIdentity( 42, 43, 44 ); 

    std::cout << q << std::endl ; 

    float3 a = make_float3(0.f, 0.f, 0.f); 
    float3 b = q.right_multiply(a, 1.f ); 

    std::cout << " t " << t << std::endl ; 
    std::cout << " a " << a << std::endl ; 
    std::cout << " b " << b << std::endl ; 

    assert( b.x == a.x + t.x ); 
    assert( b.y == a.y + t.y ); 
    assert( b.z == a.z + t.z ); 
}

void test_right_multiply_scale()
{
    float3 s = make_float3( 1.f, 2.f, 3.f ); 

    qat4 q ; 
    q.q0.f.x = s.x ; 
    q.q1.f.y = s.y ; 
    q.q2.f.z = s.z ; 

    q.setIdentity( 42, 43, 44 ); 

    std::cout << q << std::endl ; 

    float3 a = make_float3(1.f, 1.f, 1.f); 
    float3 b = q.right_multiply(a, 1.f ); 

    std::cout << " s " << s << std::endl ; 
    std::cout << " a " << a   << std::endl ; 
    std::cout << " b " << b   << std::endl ; 

    assert( b.x == a.x*s.x ); 
    assert( b.y == a.y*s.y ); 
    assert( b.z == a.z*s.z ); 
}

void test_right_multiply_rotate()
{
    float th = M_PI/4.f ; // 45 degrees 
    float ct = cos(th); 
    float st = sin(th); 

    qat4 q ;   //  rotate about Z axis 
    q.q0.f.x = ct  ;   q.q0.f.y = -st  ;  q.q0.f.z = 0.f ; 
    q.q1.f.x = st  ;   q.q1.f.y =  ct  ;  q.q1.f.z = 0.f ; 
    q.q2.f.x = 0.f ;   q.q2.f.y =  0.f ;  q.q2.f.z = 1.f ; 

    q.setIdentity( 42, 43, 44 ); 

    float3 a = make_float3(0.f, 1.f, 0.f); 
    float3 b = q.right_multiply(a, 1.f ); 

    std::cout << " q " << q << std::endl ; 
    std::cout << " a " << a   << std::endl ; 
    std::cout << " b " << b   << std::endl ; 
}

void test_cube_corners()
{
    float4 ce = make_float4( 0.f , 0.f, 0.f, 1.f ); 
    std::vector<float3> corners ; 
    AABB::cube_corners(corners, ce); 

    for(int i=0 ; i < int(corners.size()) ; i++)
    {
        const float3& a = corners[i] ; 
        std::cout << i << ":" << a << std::endl ; 
    }
}



void test_transform_aabb_inplace_2()
{
    float tx = 100.f ; 
    float ty =   0.f ; 
    float tz =   0.f ; 

    qat4 q ; 
    q.q3.f.x = tx ; 
    q.q3.f.y = ty ; 
    q.q3.f.z = tz ; 

    std::cout << q << std::endl ; 

    AABB aabb = { -100.f, -100.f, -100.f, 100.f, 100.f, 100.f } ; 
    std::cout << "aabb " << aabb << std::endl ; 
    q.transform_aabb_inplace((float*)&aabb);
    std::cout << "aabb " << aabb << std::endl ; 
}







int main(int argc, char** argv)
{
    //test_transform_aabb_inplace();
    //test_find_unique();

    //test_right_multiply_translate(); 
    //test_right_multiply_scale(); 
    //test_right_multiply_rotate(); 
    //test_cube_corners(); 
    test_transform_aabb_inplace_2();
    return 0 ; 
}


