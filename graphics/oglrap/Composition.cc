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


void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W)
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

void Composition::defineViewMatrices(glm::mat4& ModelView, glm::mat4& ModelViewProjection)
{
    glm::mat4 lookat;
    glm::mat4 projection;
    glm::mat4 shunt_to_look ;
    glm::mat4 rotate_around_look ;
    glm::mat4 shunt_back_to_eye ;

    glm::vec4 gaze ;

    // view inputs are in model coordinates (model coordinates are all within -1:1)
    // model_to_world matrix constructed from geometry center and extent
    // is used to construct the lookat matrix 

    lookat = m_view->getLookAt(m_model_to_world);
    gaze = m_view->getGaze(m_model_to_world);
    float gazelen = glm::length(gaze);

    projection = m_camera->getProjection();

    // # look is at (0,0,-gazelen) in eye frame, so here we shunt to the look and back again

    shunt_to_look = glm::translate( glm::mat4(1.), glm::vec3(0,0,gazelen));

    //rotate_around_look = m_trackball->getOrientationMatrix();
    rotate_around_look = m_trackball->getCombinedMatrix();

    shunt_back_to_eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-gazelen));

    ModelView = shunt_back_to_eye * rotate_around_look * shunt_to_look * lookat ;
    ModelViewProjection = projection * ModelView ;

    // chain of transforms   
    //
    //    projection * lookat * (world vertices)
    // 
    // lookat transform (  world frame -> eye frame )
    //
    //    * no scaling, ie still world distances
    //    * eye at (0,0,0)            
    //    * look at (0,0,-gazelength) 
    //   
}

void Composition::Summary(const char* msg)
{
    printf("%s\n", msg);
    m_trackball->Summary("m_trackball");
    m_camera->Summary("m_camera");
    m_view->Summary("m_view");
}



