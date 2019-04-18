// TEST=NGLMTest om-t

#include "NGLM.hpp"

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY.hpp"
#include <glm/gtx/matrix_interpolation.hpp>
#include <glm/gtc/random.hpp>

#include "BStr.hh"

#include "OPTICKS_LOG.hh"



void test_linearRand()
{
    for(int i=0 ; i < 100 ; i++)
    {
        float v = glm::linearRand(10.f, 20.f);
        std::cout << v << std::endl ;
    }
}



void test_axisAngle()
{

    NPY<float>* txf = NPY<float>::load("/tmp/txf.npy") ;
    if(!txf) return ; 

    std::cout << "txf: " << txf->getShapeString() << std::endl ; 
    assert( txf->hasShape(8,24,4,4)) ;

    unsigned ni = txf->getShape(0) ; 
    unsigned nj = txf->getShape(1) ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
    for(unsigned j=0 ; j < nj ; j++)
    {
        glm::mat4 trs = txf->getMat4(i,j) ;

/*
*/

/*
   // huh : shuffling ... matrix rows   gives a much more reasonable z-axis rotation result


       [[    0.    ,     0.    ,     1.    ,     0.    ],
        [   -0.7934,     0.6088,     0.    ,     0.    ],
        [   -0.6088,    -0.7934,     0.    ,     0.    ],
        [-1415.0659, -1844.1498,   750.    ,     1.    ]],



txf: 8,24,4,4
( 0, 0) {    0.0000    0.0000    1.0000} 1.7017 (  {   -0.13    0.99    0.00    0.00} {   -0.99   -0.13    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 1) {    0.0000    0.0000    1.0000} 1.9635 (  {   -0.38    0.92    0.00    0.00} {   -0.92   -0.38    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 2) {    0.0000    0.0000    1.0000} 2.2253 (  {   -0.61    0.79    0.00    0.00} {   -0.79   -0.61    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 3) {    0.0000    0.0000    1.0000} 2.4871 (  {   -0.79    0.61    0.00    0.00} {   -0.61   -0.79    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 4) {    0.0000    0.0000    1.0000} 2.7489 (  {   -0.92    0.38    0.00    0.00} {   -0.38   -0.92    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 5) {    0.0000    0.0000    1.0000} 3.0107 (  {   -0.99    0.13    0.00    0.00} {   -0.13   -0.99    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 6) {   -0.0000   -0.0000   -1.0000} 3.0107 (  {   -0.99   -0.13    0.00    0.00} {    0.13   -0.99    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 7) {   -0.0000   -0.0000   -1.0000} 2.7489 (  {   -0.92   -0.38    0.00    0.00} {    0.38   -0.92    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 8) {   -0.0000   -0.0000   -1.0000} 2.4871 (  {   -0.79   -0.61    0.00    0.00} {    0.61   -0.79    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0, 9) {   -0.0000   -0.0000   -1.0000} 2.2253 (  {   -0.61   -0.79    0.00    0.00} {    0.79   -0.61    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,10) {   -0.0000   -0.0000   -1.0000} 1.9635 (  {   -0.38   -0.92    0.00    0.00} {    0.92   -0.38    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,11) {   -0.0000   -0.0000   -1.0000} 1.7017 (  {   -0.13   -0.99    0.00    0.00} {    0.99   -0.13    0.00    0.00} {    0.00    0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,12) {    0.0000    0.0000   -1.0000} 1.4399 (  {    0.13   -0.99   -0.00    0.00} {    0.99    0.13    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,13) {    0.0000    0.0000   -1.0000} 1.1781 (  {    0.38   -0.92   -0.00    0.00} {    0.92    0.38    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,14) {    0.0000    0.0000   -1.0000} 0.9163 (  {    0.61   -0.79   -0.00    0.00} {    0.79    0.61    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,15) {    0.0000    0.0000   -1.0000} 0.6545 (  {    0.79   -0.61   -0.00    0.00} {    0.61    0.79    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,16) {    0.0000    0.0000   -1.0000} 0.3927 (  {    0.92   -0.38   -0.00    0.00} {    0.38    0.92    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,17) {    0.0000    0.0000   -1.0000} 0.1309 (  {    0.99   -0.13   -0.00    0.00} {    0.13    0.99    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,18) {    0.0000    0.0000    1.0000} 0.1309 (  {    0.99    0.13   -0.00    0.00} {   -0.13    0.99    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,19) {    0.0000    0.0000    1.0000} 0.3927 (  {    0.92    0.38   -0.00    0.00} {   -0.38    0.92    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,20) {    0.0000    0.0000    1.0000} 0.6545 (  {    0.79    0.61   -0.00    0.00} {   -0.61    0.79    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,21) {    0.0000    0.0000    1.0000} 0.9163 (  {    0.61    0.79   -0.00    0.00} {   -0.79    0.61    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,22) {    0.0000    0.0000    1.0000} 1.1781 (  {    0.38    0.92   -0.00    0.00} {   -0.92    0.38    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )
( 0,23) {    0.0000    0.0000    1.0000} 1.4399 (  {    0.13    0.99   -0.00    0.00} {   -0.99    0.13    0.00    0.00} {    0.00   -0.00    1.00    0.00} {    0.00    0.00    0.00    1.00} )




*/

        glm::mat4 trs2(1.f) ; 
        trs2[0] = trs[1] ;  //  Y->X
        trs2[1] = trs[2] ;  //  Z->Y
        trs2[2] = trs[0] ;  //  X->Z
        trs2[3] = trs[3] ;

        //  ( X,Y,Z ) -> ( Y,Z,X )

        glm::mat4 rot = glm::extractMatrixRotation( trs2 );

        glm::vec3 axis(1.0f) ; 
        float angle(0.f) ; 

        glm::axisAngle(rot ,axis, angle );

        std::cout << "(" << std::setw(2) << i << "," << std::setw(2) << j << ") " << axis << " " << angle << " " << rot << std::endl ; 

    }
    }



}


void test_decompose_tr()
{
    LOG(info) << "test_decompose_tr" ; 

    glm::vec3 axis(1,1,1);
    glm::vec3 tlat(0,0,100) ; 
    float angle = 45.f ; 

    glm::mat4 mtr ;
    mtr = glm::rotate(mtr, angle, axis );
    mtr = glm::translate(mtr, tlat );    
 
    glm::mat4 mrt ;
    mrt = glm::translate(mrt, tlat );
    mrt = glm::rotate(mrt, angle, axis );

    // hmm : the above way of constructing matrix 
    //       yields the same matrix no matter the order
    //  NOPE NOT TRUE ... OPERATOR ERROR ... ORDER MATTERS IN GLM MATRIX CONSTRUCTION  

    //assert( mtr == mrt );
 
    glm::mat3 mtr_r(mtr) ; 
    glm::vec3 mtr_t(mtr[3]);

    std::cout << gpresent(" mtr ", mtr)  << std::endl ; 
    std::cout << gpresent(" mtr_r ", mtr_r)  << std::endl ; 
    std::cout << gpresent(" mtr_t ", mtr_t)  << std::endl ; 

    glm::mat3 mrt_r(mrt) ; 
    glm::vec3 mrt_t(mrt[3]);

    std::cout << gpresent(" mrt ", mrt)  << std::endl ; 
    std::cout << gpresent(" mrt_r ", mrt_r)  << std::endl ; 
    std::cout << gpresent(" mrt_t ", mrt_t)  << std::endl ; 
}


void test_decompose_tr_invert()
{
    LOG(info) << "test_decompose_tr_invert" ;
 
    glm::vec3 axis(1,1,1);
    glm::vec3 tlat(0,0,100) ; 
    float angle = 45.f ; 

    glm::mat4 tr(1.f) ;
    tr = glm::translate(tr, tlat );
    tr = glm::rotate(tr, angle, axis );

    /*
    tr follows familar fourth column translation 4x4 matrix layout :
    which is a rotation followed by the translation on multiplying to the right, hence TR

          tr

          ir it t r 

    */

    std::cout << gpresent(" tr ", tr) << std::endl ; 
 
    // dis-member tr into r and t by inspection and separately 
    // transpose the rotation and negate the translation
    glm::mat4 ir = glm::transpose(glm::mat4(glm::mat3(tr)));
    glm::mat4 it = glm::translate(glm::mat4(1.f), -glm::vec3(tr[3])) ; 
    glm::mat4 irit = ir*it ;    // <--- inverse of tr 

    std::cout << gpresent(" ir ", ir) << std::endl ; 
    std::cout << gpresent(" it ", it) << std::endl ; 
    std::cout << gpresent(" irit ", irit ) << std::endl ; 
    std::cout << gpresent(" irit*tr ", irit*tr ) << std::endl ; 
    std::cout << gpresent(" tr*irit ", tr*irit ) << std::endl ; 
}


void test_rotate()
{
    LOG(info) << "test_rotate" ; 

    char* rot = getenv("ROTATE") ;
    std::string srot = rot ? rot : "0,0,1,45" ;  

    glm::vec4 vrot = gvec4(srot) ;
    std::cout << vrot << std::endl ; 

    glm::vec3 axis(vrot);
    float angle_d = vrot.w ; 


    glm::mat4 tr(1.f) ; 
    tr = glm::rotate(tr, glm::pi<float>()*angle_d/180.f , axis );
    glm::mat4 irit = nglmext::invert_tr( tr );

    nmat4pair mp(tr, irit);


    std::cout << mp << std::endl ; 

}



void test_dump_trs()
{
    LOG(info) << "test_dump_trs" ; 

    const char* mspecs = "t,r,s,tr,rt,trs,rts,srt" ;

    std::vector<std::string> specs ; 
    BStr::split( specs, mspecs, ',');  

    for(unsigned i=0 ; i < specs.size() ; i++)
    { 
        std::string spec = specs[i];  
        std::cout << gpresent( spec.c_str(), nglmext::make_transform(spec) )  << std::endl ; 
    }
}



void test_compDiff()
{
    LOG(info) << "test_compDiff" ; 

    glm::mat4 a = nglmext::make_transform("trs", glm::vec3(0.f), glm::vec4(0,0,1,45), glm::vec3(1) );
    glm::mat4 b = nglmext::make_transform("trs", glm::vec3(0.f), glm::vec4(0,0,1,46), glm::vec3(1) );

    std::cout << gpresent( "a", a ) << std::endl ; 
    std::cout << gpresent( "b", b ) << std::endl ; 
    std::cout << "compDiff " << nglmext::compDiff(a,b) << std::endl ; 
}


void test_compDiff_vec3()
{
    LOG(info)  ; 

    float eps = 0.001f ; 
    glm::vec3 a(1.f, 2.f, 3.f ); 
    glm::vec3 b(1.f+2.f*eps, 2.f+3.f*eps, 3.f+4.f*eps ); 

    float cd = nglmext::compDiff(a,b) ; 
    float x_cd = 4.f*eps ; 

    std::cout << gpresent( "a", a ) << std::endl ; 
    std::cout << gpresent( "b", b ) << std::endl ; 
    std::cout << "compDiff " << cd << std::endl ; 
    std::cout << "x_compDiff " << cd << std::endl ; 
    float delta = abs( x_cd - cd  ) ; 
    std::cout << "delta  " << std::setprecision(20) << std::fixed << delta << std::endl ; 

    assert( delta < 1e-5 ); 
}





/*

PBRT 2nd Edition, p97

Next we’d like to extract the pure rotation component of M. We’ll use a
technique called polar decomposition to do this. It can be shown that the polar
decomposition of a matrix M into rotation R and scale S can be computed by
successively averaging M to its inverse transpose

... if M is a pure rotation, then averaging it with its inverse transpose 
will leave it unchanged, since its inverse is equal to its transpose.

*/



void test_polar_decomposition_pluck_scale()
{
    LOG(info) ;
 
    glm::vec3 tla(10,0,10);
    glm::vec4 rot(1,1,1,45);   // 45 degrees about some axes
    glm::vec3 sca(1,2,3);

    glm::mat4 s   = nglmext::make_transform("s", tla, rot, sca);
    glm::mat4 r   = nglmext::make_transform("r", tla, rot, sca);
    glm::mat4 t   = nglmext::make_transform("t", tla, rot, sca);
    glm::mat4 trs = nglmext::make_transform("trs", tla, rot, sca);

    ndeco d ;
    nglmext::polar_decomposition( trs, d );

    std::cout << gpresent( "trs", trs ) << std::endl ;
    std::cout << gpresent( "d.trs", d.trs ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "s", s ) << std::endl ;
    std::cout << gpresent( "d.s", d.s ) << std::endl ;
    std::cout << gpresent( "d.is", d.is ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "r", r ) << std::endl ;
    std::cout << gpresent( "d.r", d.r ) << std::endl ;
    std::cout << gpresent( "d.ir", d.ir ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "t", t ) << std::endl ;
    std::cout << gpresent( "d.t", d.t ) << std::endl ;
    std::cout << gpresent( "d.it", d.it ) << std::endl ;
    std::cout << std::endl; 


    glm::vec3 dsca = nglmext::pluck_scale( d ); 
    std::cout << gpresent( "dsca", dsca ) << std::endl ;

    bool has_scale = nglmext::has_scale( dsca ); 
    std::cout << " has_scale " << has_scale << std::endl ; 


}



void test_polar_decomposition_trs()
{
    LOG(info) << "test_polar_decomposition_trs" ; 

    glm::vec3 tla(0,0,100);
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(2.f);

    glm::mat4 s   = nglmext::make_transform("s", tla, rot, sca);
    glm::mat4 r   = nglmext::make_transform("r", tla, rot, sca);
    glm::mat4 t   = nglmext::make_transform("t", tla, rot, sca);
    glm::mat4 trs = nglmext::make_transform("trs", tla, rot, sca);

    ndeco d ;
    nglmext::polar_decomposition( trs, d );

    std::cout << gpresent( "trs", trs ) << std::endl ;
    std::cout << gpresent( "d.trs", d.trs ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "s", s ) << std::endl ;
    std::cout << gpresent( "d.s", d.s ) << std::endl ;
    std::cout << gpresent( "d.is", d.is ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "r", r ) << std::endl ;
    std::cout << gpresent( "d.r", d.r ) << std::endl ;
    std::cout << gpresent( "d.ir", d.ir ) << std::endl ;
    std::cout << std::endl; 

    std::cout << gpresent( "t", t ) << std::endl ;
    std::cout << gpresent( "d.t", d.t ) << std::endl ;
    std::cout << gpresent( "d.it", d.it ) << std::endl ;
    std::cout << std::endl; 

    glm::mat4 i_trs = glm::inverse(d.trs);

    std::cout << gpresent( "i_trs", i_trs ) << std::endl ;
    std::cout << gpresent( "d.isirit", d.isirit ) << std::endl ;

    glm::mat4 i_trs_x_trs = i_trs * trs ;  
    std::cout << gpresent( "i_trs_x_trs", i_trs_x_trs ) << std::endl ;

    glm::mat4 isirit_x_trs = d.isirit * trs ;  
    std::cout << gpresent( "isirit_x_trs", isirit_x_trs ) << std::endl ;
}

/*
/usr/local/opticks/externals/glm/glm-0.9.6.3/glm/detail/type_mat4x4.inl
449     template <typename T, precision P>
450     template <typename U>
451     GLM_FUNC_QUALIFIER tmat4x4<T, P> & tmat4x4<T, P>::operator*=(tmat4x4<U, P> const & m)
452     {
453         return (*this = *this * m);
454     }
*/

void test_order()
{
    LOG(info) << "test_order" ;
 
    const int n = 3 ; 
    glm::mat4 r[n] ; 

    r[0] = nglmext::make_transform("r", glm::vec3(0.f), glm::vec4(1,0,0,45), glm::vec3(1.f) );
    r[1] = nglmext::make_transform("r", glm::vec3(0.f), glm::vec4(0,1,0,45), glm::vec3(1.f) );
    r[2] = nglmext::make_transform("r", glm::vec3(0.f), glm::vec4(0,0,1,45), glm::vec3(1.f) );

    glm::mat4 ri(1.f) ; 
    glm::mat4 rj(1.f) ; 

    for(unsigned i=0,j=n-1 ; i < n ; i++,j-- )
    {
        ri *= r[i] ; 
        rj *= r[j] ; 
    }
    
    glm::mat4 r012 = r[0] * r[1] * r[2] ; 
    glm::mat4 r210 = r[2] * r[1] * r[0] ; 

    assert(nglmext::compDiff(r012, ri) < 1e-4 );
    assert(nglmext::compDiff(r210, rj) < 1e-4 );

    std::cout << gpresent( "r[0]", r[0] ) << std::endl ; 
    std::cout << gpresent( "r[1]", r[1] ) << std::endl ; 
    std::cout << gpresent( "r[2]", r[2] ) << std::endl ; 
    std::cout << gpresent( "ri", ri ) << std::endl ; 
    std::cout << gpresent( "r012", r012 ) << std::endl ; 
    std::cout << gpresent( "rj", rj ) << std::endl ; 
    std::cout << gpresent( "r210", r210 ) << std::endl ; 
}

void test_make_transform()
{
    // compare with opticks/dev/csg/glm.py  
    LOG(info) << "test_make_transform" ;

    glm::vec3 tla(0,0,100);
    glm::vec4 rot(0,0,1,45);
    glm::vec3 sca(1,2,3);

    glm::mat4 t = nglmext::make_transform("t", tla, rot, sca );
    glm::mat4 r = nglmext::make_transform("r", tla, rot, sca );
    glm::mat4 s = nglmext::make_transform("s", tla, rot, sca );

    glm::mat4 trs = nglmext::make_transform("trs", tla, rot, sca );

    std::cout << gpresent( "t", t ) << std::endl ; 
    std::cout << gpresent( "r", r ) << std::endl ; 
    std::cout << gpresent( "s", s ) << std::endl ; 
    std::cout << gpresent( "trs", trs ) << std::endl ; 
}




glm::vec3 transform_normal(const glm::vec3& nrm_, const glm::mat4& trs, const glm::mat4& isirit, bool verbose=false)
{
    glm::vec4 nrm(glm::normalize(nrm_), 0.f );

    glm::mat4 i_trs_T = glm::transpose(glm::inverse(trs)) ; //  transpose of inverse
    glm::mat4 isirit_T = glm::transpose( isirit ) ;

    if(verbose)
    {
        std::cout << gpresent( "trs", trs) << std::endl ; 
        std::cout << gpresent( "i_trs_T", i_trs_T) << std::endl ; 
        std::cout << gpresent( "isirit_T", isirit_T) << std::endl ; 
    }

    glm::vec4 t_nrm_0 = isirit_T * nrm ; 
    glm::vec4 t_nrm_1 = nrm * isirit  ;   // dont bother with transpose just pre-multiply rather than post-multiply 
    assert( t_nrm_0 == t_nrm_1 );


    glm::vec4 t_nrm_2 = i_trs_T * nrm ; 

    std::cout << gpresent( "nrm", nrm) ; 
    std::cout << gpresent( "t_nrm_0", t_nrm_0) ; 
    std::cout << gpresent( "t_nrm_0n", glm::normalize(t_nrm_0)) ; 
    std::cout << gpresent( "t_nrm_1", t_nrm_1) ; 
    std::cout << gpresent( "t_nrm_2", t_nrm_2) ;
    std::cout << std::endl ; 


    return glm::vec3(t_nrm_1) ; 
}


void test_transform_normal()
{
    LOG(info) << "test_transform_normal" ;
 
    glm::vec3 tla(0,0,0);
    glm::vec4 rot(0,0,1,0);
    glm::vec3 sca(1,1,2);

    glm::mat4 trs = nglmext::make_transform("trs", tla, rot, sca);

    ndeco d ;
    nglmext::polar_decomposition( trs, d );


    transform_normal( glm::vec3(1,0,0) , trs, d.isirit, true );
    transform_normal( glm::vec3(0,1,0) , trs, d.isirit );
    transform_normal( glm::vec3(0,0,1) , trs, d.isirit );

    transform_normal( glm::vec3(0,1,1) , trs, d.isirit );
    transform_normal( glm::vec3(1,0,1) , trs, d.isirit );
    transform_normal( glm::vec3(1,1,0) , trs, d.isirit );


    // when the normal has some component in the scaled direction the 
    // and there is some translation getting stiff in the w ?
}


void test_mix()
{
    LOG(info) << "test_mix" ; 
    float x0 = 10.f ; 
    float x1 = 20.f ; 

    for(float a=0 ; a < 1.01 ; a+=0.1 )
    {
         float xa = glm::mix(x0,x1,a) ;
         std::cout 
            << " a " << std::setw(7) << a  
            << " xa " << std::setw(7) << xa 
            << std::endl ;

    } 

}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

/*
    //test_decompose_tr_invert();
    //test_rotate();

    //test_decompose_tr();
    test_compDiff();
    test_dump_trs();
    test_polar_decomposition_trs();

    test_order();

    test_make_transform();

    test_transform_normal();

    test_axisAngle();

    test_mix();
    test_linearRand();
*/

    test_polar_decomposition_pluck_scale();
    //test_compDiff_vec3();


    return 0 ; 
}


