

#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include <glm/gtc/epsilon.hpp>


#include "PLOG.hh"

void test_stream()
{
    glm::ivec3 iv[4] = {
       {1,2,3},
       {10,20,30},
       {100,200,300},
       {1000,2000,3},
    };

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << glm::to_string(iv[i]) << std::endl ; 

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << iv[i] << std::endl ; 



    glm::vec3 fv[4] = {
       {1.23,2.45,3},
       {10.12345,20,30.2235263},
       {100,200,300},
       {1000,2000,3},
    };

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << glm::to_string(fv[i]) << std::endl ; 

    for(int i=0 ; i < 4 ; i++)
        std::cout << std::setw(20) << fv[i] << std::endl ; 

}


void test_invert_tr()
{
    LOG(info) << "test_invert_tr" ;
 
    glm::vec3 axis(1,1,1);
    glm::vec3 tlat(0,0,100) ; 
    float angle = 45.f ; 

    glm::mat4 tr(1.f) ;
    tr = glm::translate(tr, tlat );
    tr = glm::rotate(tr, angle, axis );

    glm::mat4 irit = nglmext::invert_tr( tr );

    //std::cout << gpresent(" tr ", tr) << std::endl ; 
    //std::cout << gpresent(" irit ", irit ) << std::endl ; 

    nmat4pair mp(tr, irit);
    std::cout << " mp " << mp << " dig " << mp.digest() << std::endl ; 

}


void test_make_mat()
{
    glm::vec3 tlat(0,0,100) ; 
    glm::vec4 trot(0,0,1,45) ; 
    glm::vec3 tsca(1,1,2) ; 

    glm::vec4 orig(0,0,0,1);
    glm::vec4 px(1,0,0,1);
    glm::vec4 py(0,1,0,1);
    glm::vec4 pz(0,0,1,1);
    glm::vec4 pxyz(1,1,1,1);


    glm::mat4 T = nglmext::make_translate(tlat);
    glm::mat4 R = nglmext::make_rotate(trot);
    glm::mat4 S = nglmext::make_scale(tsca);


    glm::mat4 SRT = nglmext::make_transform("srt", tlat, trot, tsca );
    glm::mat4 TRS = nglmext::make_transform("trs", tlat, trot, tsca );

    glm::mat4 S_R_T = S*R*T ; 
    glm::mat4 T_R_S = T*R*S ; 


    std::cout 
         << std::endl 

         << gpresent( "tlat",  tlat ) 
         << gpresent( "trot",  trot ) 
         << gpresent( "tsca",  tsca )

         << std::endl 
         << std::endl 

         << gpresent( "T",  T )
         << std::endl 
         << gpresent( "R",  R )
         << std::endl 
         << gpresent( "S",  S )

         << std::endl 
         << std::endl 

         << gpresent( "SRT",  SRT )
         << std::endl 
         << gpresent( "S*R*T",  S_R_T )
         << std::endl 
          
         << gpresent( "SRT*orig",  SRT*orig )
         << gpresent( "SRT*px",    SRT*px )
         << gpresent( "SRT*py",    SRT*py )
         << gpresent( "SRT*pz",    SRT*pz )
         << gpresent( "SRT*pxyz",  SRT*pxyz )

         << std::endl 
         << " S really does happen last when using \"srt\" argument to make_transform "
         << std::endl 
         << std::endl 
         << std::endl 

         << gpresent( "TRS",  TRS )
         << std::endl 
         << gpresent( "T*R*S",  T_R_S )
         << std::endl 

         << gpresent( "TRS*orig",  TRS*orig )
         << gpresent( "TRS*px",    TRS*px )
         << gpresent( "TRS*py",    TRS*py )
         << gpresent( "TRS*pz",    TRS*pz )
         << gpresent( "TRS*pxyz",  TRS*pxyz )

         << std::endl
         << " T really does happen last when using \"trs\" argument to make_transform "

         << std::endl  
         ;

}





void test_make_transform()
{
    LOG(info) << "test_make_transform" ;

    glm::vec3 tlat(0,0,100) ; 
    glm::vec4 trot(0,0,1,45) ; 
    glm::vec3 tsca(1,1,2) ; 

    glm::mat4 srt = nglmext::make_transform("srt", tlat, trot, tsca );
    glm::mat4 trs = nglmext::make_transform("trs", tlat, trot, tsca );

    glm::vec4 orig(0,0,0,1);
    glm::vec4 px(1,0,0,1);
    glm::vec4 py(0,1,0,1);
    glm::vec4 pz(0,0,1,1);
    glm::vec4 pxyz(1,1,1,1);

    std::cout 
         << std::endl 
         << "NB gpresent(mat4) does i=0..3 j=0..3  ( flip ? m[j][i] : m[i][j] )  with flip=false  ie row-by-row ROWMAJOR ? "   
         << std::endl 
         << std::endl 

         << gpresent( "tlat",  tlat ) 
         << gpresent( "trot",  trot ) 
         << gpresent( "tsca",  tsca )

         << std::endl 
        
         << gpresent( "srt",  srt ) << std::endl
         << gpresent( "trs",  trs ) << std::endl 

         << gpresent( "orig",  orig ) 
         << gpresent( "px",  px ) 
         << gpresent( "py",  py ) 
         << gpresent( "pz",  pz ) 
         << gpresent( "pxyz",  pxyz ) 

         << std::endl
         << " z-values ~200 shows the scaling of z by 2 happened after the translate " 
         << std::endl 

         << gpresent( "srt*orig",  srt*orig )  
         << gpresent( "srt*px",    srt*px  ) 
         << gpresent( "srt*py",    srt*py  ) 
         << gpresent( "srt*pz",    srt*pz  )
         << gpresent( "srt*pxyz",  srt*pxyz  )

         << std::endl 
         << " z-values ~100 shows the scaling of z by 2 happened before the translate " 
         << std::endl 


         << gpresent( "trs*orig",  trs*orig ) 
         << gpresent( "trs*px",    trs*px  ) 
         << gpresent( "trs*py",    trs*py  )
         << gpresent( "trs*pz",    trs*pz  )
         << gpresent( "trs*pxyz",  trs*pxyz  )

        << std::endl 
         << " wrong way around gives crazy w for some" 
         << std::endl 

         << gpresent( "orig*srt",  orig*srt ) 
         << gpresent( "px*srt",    px*srt  ) 
         << gpresent( "py*srt",    py*srt  ) 
         << gpresent( "pz*srt",    pz*srt  )
         << gpresent( "pxyz*srt",  pxyz*srt  )

         << std::endl 

         << gpresent( "orig*trs ",  orig*trs ) 
         << gpresent( "px*trs",     px*trs  ) 
         << gpresent( "py*trs",     py*trs  )
         << gpresent( "pz*trs",     pz*trs  )
         << gpresent( "pxyz*trs",   pxyz*trs  )

         << std::endl 
         ;  
}



void test_nmat4triple_make_translated()
{
    LOG(info) << "test_nmat4triple_make_translated" ;

    glm::vec3 tlat(0,0,100) ; 
    glm::vec4 trot(0,0,1,0) ; 
    glm::vec3 tsca(1,1,2) ; 

    glm::mat4 trs = nglmext::make_transform("trs", tlat, trot, tsca );

    nmat4triple tt0(trs);

    glm::vec3 tlat2(-100,0,0) ; 

    bool reverse(true);
    const nmat4triple* tt1 = tt0.make_translated( tlat2, reverse, "test_nmat4triple_make_translated" );
    const nmat4triple* tt2 = tt0.make_translated( tlat2, !reverse, "test_nmat4triple_make_translated" );

    std::cout 
         << std::endl 
         << gpresent("trs", trs) 
         << std::endl 
         << gpresent("tt0.t", tt0.t ) 
         << std::endl 
         << gpresent("tt1.t", tt1->t ) 
         << std::endl 
         << gpresent("tt2.t", tt2->t ) 
         << std::endl 
        ;  
}


void test_nmat4triple_id_digest()
{
    LOG(info) << "test_nmat4triple_id_digest" ; 

    const nmat4triple* id = nmat4triple::make_identity() ;
    std::cout << " id " << *id << " dig " << id->digest() << std::endl ; 
}

void test_nmat4triple_is_identity()
{
    LOG(info) << "test_nmat4triple_is_identity" ; 
    const nmat4triple* id = nmat4triple::make_identity() ;
    std::cout << " id " << *id << " dig " << id->digest() << std::endl ; 

    assert( id->is_identity() );


}




void test_apply_transform()
{
    LOG(info) << "test_apply_transform" ;
 
    const nmat4triple* id = nmat4triple::make_identity() ;
    const nmat4triple* sc = nmat4triple::make_scale(10,10,10) ;

    glm::vec3 p(1,2,3);
    glm::vec3 q = id->apply_transform_t( p );
    glm::vec3 qs = sc->apply_transform_t( p );

    assert( q.x == p.x );
    assert( q.y == p.y );
    assert( q.z == p.z );

    assert( qs.x == 10*p.x );
    assert( qs.y == 10*p.y );
    assert( qs.z == 10*p.z );
}


void test_apply_transform_vec()
{
    LOG(info) << "test_apply_transform_vec" ; 

    const nmat4triple* sc = nmat4triple::make_scale(10,10,10) ;

    std::vector<glm::vec3> src ; 
    src.push_back(glm::vec3(1,0,0));
    src.push_back(glm::vec3(0,1,0));
    src.push_back(glm::vec3(0,0,1));
    src.push_back(glm::vec3(1,1,1));

    std::vector<glm::vec3> dst ; 
    sc->apply_transform_t( dst, src );
    assert( dst.size() == src.size() ); 


    for(unsigned i=0 ; i < dst.size() ; i++)
    {
        glm::vec3 s = src[i] ;
        glm::vec3 d = dst[i] ;

        std::cout
            << " i " << std::setw(3) << i 
            << " s " << gpresent(s) 
            << " d " << gpresent(d)
            << std::endl 
            ; 
    }
}




void get_directions(std::vector<glm::vec3>& dirs)
{
    dirs.push_back(glm::vec3(1,0,0));
    dirs.push_back(glm::vec3(0,1,0));
    dirs.push_back(glm::vec3(0,0,1));
    dirs.push_back(glm::vec3(0,1,1));
    dirs.push_back(glm::vec3(1,0,1));
    dirs.push_back(glm::vec3(1,1,0));
    dirs.push_back(glm::vec3(1,1,1));
}


void test_pick_up()
{
    LOG(info) << "test_pick_up" ; 

    std::vector<glm::vec3> dirs ;
    get_directions(dirs);

    glm::vec3 up ; 
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        nglmext::_pick_up(up, dirs[i] );
        std::cout << std::setw(2) << i 
                  << " dir " << gpresent(dirs[i])
                  << " up " << gpresent(up)
                  << std::endl 
                  ;
    }
}


void test_define_uv_basis()
{
    LOG(info) << "test_define_uv_basis" ; 

    std::vector<glm::vec3> dirs ;
    get_directions(dirs);

    glm::vec3 udir ; 
    glm::vec3 vdir ; 
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        glm::vec3 perp = dirs[i] ;
        nglmext::_define_uv_basis(perp, udir, vdir) ;

        std::cout << std::setw(2) << i 
                  << " perp " << gpresent(perp)
                  << " udir " << gpresent(udir)
                  << " vdir " << gpresent(vdir)
                  << std::endl 
                  ;
    }
}

void test_make_yzflip()
{
    glm::mat4 yz = nglmext::make_yzflip(); 
    std::cout << gpresent("yz", yz ) << std::endl ; 
}





int main(int argc, char** argv)
{
    PLOG_(argc, argv) ; 

    //test_stream();
    //test_invert_tr();
    //test_make_mat();
    //test_make_transform();
    //test_nmat4triple_make_translated();
    //test_nmat4triple_id_digest();

    //test_apply_transform();
    //test_apply_transform_vec();

    //test_pick_up() ;
    //test_define_uv_basis();

    //test_nmat4triple_is_identity();

    test_make_yzflip(); 


    return 0 ; 
}
