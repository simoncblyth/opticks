#include "Composition.hh"

#include "Camera.hh"
#include "Trackball.hh"
#include "View.hh"


#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

const char* Composition::PRINT = "print" ; 

Composition::Composition()
  :
  m_camera(NULL),
  m_view(NULL),
  m_trackball(NULL),
  m_model_to_world()
{
  m_camera = new Camera() ;
  m_view   = new View() ;
  m_trackball = new Trackball() ;
}

Composition::~Composition()
{
}


void Composition::configureI(const char* name, std::vector<int> values )
{
    printf("Composition::configureI\n");
    if(values.empty()) return ; 
    if(strcmp(name, PRINT)==0)
    {
        //int print = values.back();
        Summary("Composition::configureI");
    }
}



unsigned int Composition::getWidth()
{
   return m_camera->getWidth();
}
unsigned int Composition::getHeight()
{
   return m_camera->getHeight();
}
void Composition::setSize(unsigned int width, unsigned int height)
{
    m_camera->setSize(width, height);
}

Camera* Composition::getCamera()
{
    return m_camera ;
}
View* Composition::getView()
{
    return m_view ;
}
Trackball* Composition::getTrackball()
{
    return m_trackball ;
}
glm::mat4& Composition::getModelToWorld()
{
    return m_model_to_world ; 
}

void Composition::setModelToWorld(float* m2w)
{
    m_model_to_world = glm::make_mat4(m2w);
}

void Composition::defineViewMatrices(glm::mat4& ModelView, glm::mat4& ModelViewInverse, glm::mat4& ModelViewProjection)
{
    glm::mat4 lookat;
    glm::mat4 projection;
    glm::mat4 eye2look ;
    glm::mat4 trackballing;
    glm::mat4 itrackballing;
    glm::mat4 look2eye ;

    // view inputs are in model coordinates (model coordinates are all within -1:1)
    // model_to_world matrix constructed from geometry center and extent
    // is used to construct the lookat matrix 

    glm::mat4 world2camera ;
    glm::mat4 camera2world ;
    glm::vec4 gaze ;

    m_view->getTransforms(m_model_to_world, world2camera, camera2world, gaze );  
    float gazelen = glm::length(gaze);

    // # look is at (0,0,-gazelen) in eye frame, so here we shunt to the look and back again

    eye2look              = glm::translate( glm::mat4(1.), glm::vec3(0,0,gazelen));

    m_trackball->getCombinedMatrices(trackballing, itrackballing);

    look2eye              = glm::translate( glm::mat4(1.), glm::vec3(0,0,-gazelen));


    //lookat = m_view->getLookAt(m_model_to_world);
    // TODO: assert allclose(lookat, world2camera)

    ModelView = look2eye * trackballing * eye2look * world2camera ;           // world2eye

    m_world2eye = ModelView ; 

    ModelViewInverse = camera2world * look2eye * itrackballing * eye2look ;   // eye2world

    m_eye2world = ModelViewInverse ; 

    projection = m_camera->getProjection();

    ModelViewProjection = projection * ModelView ;


/*
  //  env/geant4/geometry/collada/g4daeview/daetransform.py

 51     def _get_world2eye(self):
 52         """
 53         Objects are transformed from **world** space to **eye** space using GL_MODELVIEW matrix, 
 54         as daeviewgl regards model spaces as just input parameter conveniences
 55         that OpenGL never gets to know about those.  
 56 
 57         So need to invert MODELVIEW and apply it to the origin (eye position in eye space)
 58         to get world position of eye.  Can then convert that into model position.  
 59 
 60         Motivation:
 61 
 62            * determine effective view point (eye,look,up) after trackballing around
 ..
 88         return reduce(np.dot, [self.downscale,
 89                                self.trackball.translate,
 90                                self.view.translate_look2eye,   # (0,0,-distance)
 91                                self.trackball.rotation,
 92                                self.view.translate_eye2look,   # (0,0,+distance)
 93                                self.view.world2camera ])
 94     world2eye = property(_get_world2eye)   # this matches GL_MODELVIEW
 ..
 96     def _get_eye2world(self):
 97         return reduce(np.dot, [self.view.camera2world,
 98                                self.view.translate_look2eye,
 99                                self.trackball.rotation.T,
100                                self.view.translate_eye2look,
101                                self.trackball.untranslate,
102                                self.upscale])
103     eye2world = property(_get_eye2world)
*/


 }


void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
   //
   //  Eye space basis needs to be transformed into worldspace
   //
   //       x  [1,0,0,0]
   //       y  [0,1,0,0]
   //       z  [0,0,1,0]
   //       e  [0,0,0,1]   // origin
   //
   //  need to transform eye space basis into world space to provide to OptiX
   //  so need inverse of ModelView  ie the eye2world matrix
   //

   eye = glm::vec3(m_eye2world[3]);  
   U = glm::vec3(m_eye2world[0]);  
   V = glm::vec3(m_eye2world[1]);  
   W = glm::vec3(m_eye2world[2]);  
}

    
void Composition::getEyeUVW_no_trackball(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
{
   glm::vec3 e ;  
   glm::vec3 unorm ;  
   glm::vec3 vnorm ;  
   glm::vec3 gaze ;  

   m_view->getFocalBasis( m_model_to_world, e,unorm,vnorm, gaze );

   float tanYfov = m_camera->getTanYfov();
   float aspect = m_camera->getAspect();

   float v_half_height = glm::length(gaze) * tanYfov ; 
   float u_half_width  = v_half_height * aspect ; 

   eye = e ;
   U = unorm * u_half_width ; 
   V = vnorm * v_half_height ; 
   W = gaze ; 
}


void Composition::getLookAt(glm::mat4& lookat)
{
    lookat = m_view->getLookAt(m_model_to_world);
}




void Composition::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_trackball->Summary("m_trackball");
    m_camera->Summary("m_camera");
    m_view->Summary("m_view");
}



