#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "Composition.hh"
#include "SGLM.h"


void test_getEyeUVW(const Opticks* ok)
{
    Composition* comp = new Composition(ok) ;

    // CSGOptiX::setComposition 

    glm::vec4 ce(0.f, 0.f, 0.f, 100.f ); 
    bool autocam = true ;
    const qat4* m2w = nullptr ; 
    const qat4* w2m = nullptr ; 
    comp->setCenterExtent( ce, autocam, m2w, w2m ); 

    float extent = ce.w ;
    float tmin_model = 0.1f ; 
    float tmin = extent*tmin_model ;
    comp->setNear(tmin); 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    glm::vec4 ZProj ;

    comp->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first

    float tmin_ = comp->getNear(); 
    float tmax = comp->getFar(); 
    unsigned cameratype = comp->getCameraType(); 

    LOG(info)
        << " extent " << extent
        << " tmin " << tmin 
        << " tmin_ " << tmin_ 
        << " tmax " << tmax 
        << " eye (" << eye.x << " " << eye.y << " " << eye.z << " ) "
        << " U (" << U.x << " " << U.y << " " << U.z << " ) "
        << " V (" << V.x << " " << V.y << " " << V.z << " ) "
        << " W (" << W.x << " " << W.y << " " << W.z << " ) "
        << " cameratype " << cameratype
        ;   



   SGLM::SetCE( ce.x, ce.y, ce.z, ce.w ); 
   SGLM sglm ; 
   //sglm.near = tmin ; 
   LOG(info) << sglm.desc() ; 


   LOG(info) << sglm.descEyeBasis() ; 
   LOG(info) << comp->descEyeBasis(); 


}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    test_getEyeUVW(&ok); 



    return 0 ;
}

