#include "Composition.hh"
#include "Camera.hh"
#include "View.hh"

#include <iomanip>
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "GLMFormat.hpp"

void test_rotate()
{
    glm::vec3 X(1,0,0);
    glm::vec3 Y(0,1,0);
    glm::vec3 Z(0,0,1);

    float angle = 0.f ; 
    for(unsigned int i=0 ; i < 6 ; i++)
    {
        switch(i)
        {
            case 0:angle = 0.f ; break;   
            case 1:angle = 30.f ; break;   
            case 2:angle = 45.f ; break;   
            case 3:angle = 60.f ; break;   
            case 4:angle = 90.f ; break;   
            case 5:angle = 180.f ; break;   
        }

        float a = angle*M_PI/180. ; 
        printf(" angle %10.4f a %10.4f \n", angle, a );

        glm::mat4 rotX = glm::rotate(glm::mat4(1.0), a, X );
        glm::mat4 rotY = glm::rotate(glm::mat4(1.0), a, Y );
        glm::mat4 rotZ = glm::rotate(glm::mat4(1.0), a, Z );

        glm::mat4 irotX = glm::transpose(rotX);
        glm::mat4 irotY = glm::transpose(rotY);
        glm::mat4 irotZ = glm::transpose(rotZ);

        print(rotX, "rotX"); 
        print(irotX, "irotX"); 

        print(rotY, "rotY"); 
        print(irotY, "irotY"); 

        print(rotZ, "rotZ"); 
        print(irotZ, "irotZ"); 



   }
}



void test_center_extent()
{
   NPY<float>* dom = NPY<float>::load("domain", "1", "dayabay");
   dom->dump();
   glm::vec4 ce = dom->getQuad(0,0);
   print(ce, "ce");

   Composition c ; 
   c.setCenterExtent(ce);
   c.update();
   c.dumpAxisData();
}

void test_setCenterExtent()
{
    glm::vec4 ce(100.,100.,100.,500.);
    glm::vec3 sc(ce.w);
    glm::vec3 tr(ce.x, ce.y, ce.z);

    glm::vec3 isc(1.f/ce.w);

    glm::mat4 m_model_to_world = glm::scale( glm::translate(glm::mat4(1.0), tr), sc); 

    glm::mat4 m_world_to_model = glm::translate( glm::scale(glm::mat4(1.0), isc), -tr); 
 

    glm::mat4 check = m_world_to_model * m_model_to_world ;

    print(m_model_to_world, "m_model_to_world");
    print(m_world_to_model, "m_world_to_model");
    print(check, "check");
}


void test_depth()
{
   Composition* comp = new Composition ;
   View* view = comp->getView();
   Camera* cam = comp->getCamera();

   float s = 100.0 ; 

   glm::vec4 ce(0.,0.,0.,s);

    // extent normalized inputs to view
   view->setEye(-1,0,0) ;   
   view->setLook(0,0,0) ;
   view->setUp(0,1,0) ;

   bool autocam ;  
   comp->setCenterExtent(ce, autocam=true );

   cam->Summary("test_depth cam");
   view->Summary("test_depth view");

   glm::vec4 vp = comp->getViewpoint();
   glm::vec4 lp = comp->getLookpoint();
   glm::vec4 gaze = comp->getGaze();
   glm::vec4 front = glm::normalize(gaze);

   print(vp, "viewpoint");
   print(lp, "lookpoint");
   print(gaze, "gaze");
   print(front, "front");


   comp->update();

   glm::vec4 zproj ; 
   cam->fillZProjection(zproj);

   print(zproj, "zproj");

   //unsigned int ix = cam->getWidth()/2 ;  
   //unsigned int iy = cam->getHeight()/2 ;  
   float near = cam->getNear();
   float far  = cam->getFar();

   std::cout
        << " near " << near 
        << " far " << far
        << std::endl ;  
  

   std::cout << " step along the gaze direction, from near to far  " << std::endl ; 
   // (world frame : X axis from -100 to 0)
   // (eye frame   : Z axis from    0 to -100 ) 

   int N = 20 ;
   for(int i=0 ; i <= N ; i++)
   {
       float t = near + (far - near)*float(i)/float(N)  ;

       const glm::vec4 p_world = vp + front*t ;
       const glm::vec4 p_eye = comp->transformWorldToEye(p_world);
       const glm::vec3 p_ndc = comp->getNDC(p_world); 
       const glm::vec3 p_ndc2 = comp->getNDC2(p_world); 

       float eyeDist = p_eye.z ;   // from eye frame definition, this is general
       assert(eyeDist <= 0.f ); 
       float ndc_z = -zproj.z - zproj.w/eyeDist ;     // range -1:1 for visibles
       float ndc_z2 = comp->getNDCDepth(p_world); 


       float clip_z = 0.5f*ndc_z + 0.5f ;             // range  0:1 
       float depth = clip_z ; 
       float clip_z2 = comp->getClipDepth(p_world);

       glm::vec3 unp = comp->unProject(0,0,-depth); // ix,iy ??

       std::cout << " t " << std::setw(5) << t
                 << " p_world " << std::setw(30) << gformat(p_world)
                 << " p_eye   " << std::setw(30) << gformat(p_eye)
                 << " p_ndc " << std::setw(30) << gformat(p_ndc)
                 << " p_ndc2 " << std::setw(30) << gformat(p_ndc2)
                 << " eyeDist " << std::setw(8) << eyeDist
                 << " ndc_z " << std::setw(8) << ndc_z
                 << " ndc_z2 " << std::setw(8) << ndc_z2
                 << " clip_z " << std::setw(8) << clip_z
                 << " clip_z2 " << std::setw(8) << clip_z2
                 << " unp   " << std::setw(20) << gformat(unp)
                 << std::endl ; 

   } 



}


int main()
{
   //test_center_extent();
   //test_setCenterExtent();
   //test_rotate();
   test_depth();

    
   return 0 ;
}
