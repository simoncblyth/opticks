#include <GL/glew.h>

#include "Scene.hh"
#include "Shader.hh"
#include "Camera.hh"
#include "Trackball.hh"
#include "View.hh"
#include "Common.hh"


// assimpwrap
#include "AssimpWrap/AssimpGGeo.hh"

// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
#include "GDrawable.hh"
#include "GMergedMesh.hh"
#include "GGeo.hh"

#include "stdio.h"
#include "stdlib.h"

#include <glm/glm.hpp>  
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>


Scene::Scene()
 
  :
  m_shader(NULL),
  m_shaderdir(NULL),
  m_camera(NULL),
  m_view(NULL),
  m_trackball(NULL),
  m_geometry(NULL),
  m_draw_count(0)

{
  m_camera = new Camera() ;
  m_view   = new View() ;
  m_trackball = new Trackball() ;
}

Scene::~Scene()
{
}

float* Scene::getModelToWorld()
{
    return m_model_to_world ; 
}

void Scene::setCamera(Camera* camera)
{
    m_camera = camera ;
}
Camera* Scene::getCamera()
{
    return m_camera ;
}
void Scene::setView(View* view)
{
    m_view = view ;
}
View* Scene::getView()
{
    return m_view ;
}

void Scene::setTrackball(Trackball* trackball)
{
    m_trackball = trackball ;
}
Trackball* Scene::getTrackball()
{
    return m_trackball ;
}




void Scene::setGeometry(GDrawable* geometry)
{
    m_geometry = geometry ;
}
GDrawable* Scene::getGeometry()
{
    return m_geometry ;
}
void Scene::setShaderDir(const char* dir)
{
    m_shaderdir = strdup(dir);
}
char* Scene::getShaderDir()
{
    return m_shaderdir ? m_shaderdir : getenv("SHADER_DIR") ;
}






GLuint Scene::upload(GLenum target, GLenum usage, GBuffer* buffer)
{
    buffer->Summary("Scene::upload");
    GLuint id ; 
    glGenBuffers(1, &id);
    glBindBuffer(target, id);
    glBufferData(target, buffer->getNumBytes(), buffer->getPointer(), usage);
    return id ; 
}


void Scene::load(const char* envprefix)
{
    GGeo* ggeo = AssimpGGeo::load(envprefix);

    GMergedMesh* geo = ggeo->getMergedMesh(); 
    //GMesh* geo = ggeo->getMesh(0); 
    //Demo* geo = new Demo()

    assert(geo);
    geo->setColor(0.5,0.5,1.0);
    geo->Summary("Scene::load Sumary");
    //geo->Dump("Scene::load Dump");

    setGeometry(geo);
}


void Scene::dump(const char* msg)
{
    printf("%s\n", msg );
    printf("vertices  %u \n", m_vertices);
    printf("colors    %u \n", m_colors);
    printf("indices   %u \n", m_indices);
    printf("nelem     %d \n", m_indices_count);
    printf("shaderdir %s \n", getShaderDir());

    m_shader->dump(msg);
}

void Scene::init_opengl()
{
    // as there are two GL_ARRAY_BUFFER for vertices and colors need
    // to bind them again (despite bound in upload) in order to 
    // make the desired one active when create the VertexAttribPointer :
    // the currently active buffer being recorded "into" the VertexAttribPointer 
    //
    // without 
    //     glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);
    // got a blank despite being bound in the upload 
    // when VAO creation was after upload. It appears necessary to 
    // moving VAO creation to before the upload in order for it 
    // to capture this state.
    //
    // As there is only one GL_ELEMENT_ARRAY_BUFFER there is 
    // no need to repeat the bind, but doing so for clarity
    //
    assert(m_geometry);

    m_model_to_world  = (float*)m_geometry->getModelToWorldBuffer()->getPointer();


    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    GBuffer* vbuf = m_geometry->getVerticesBuffer();
    GBuffer* cbuf = m_geometry->getColorsBuffer();
    GBuffer* ibuf = m_geometry->getIndicesBuffer();

    assert(vbuf->getNumBytes() == cbuf->getNumBytes());
    assert(vbuf->getNumBytes() == cbuf->getNumBytes());

    m_vertices = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  vbuf );
    m_colors   = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  cbuf );

    m_indices  = upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, ibuf );
    m_indices_count = ibuf->getNumItems(); // number of indices

    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    // the vbuf and cbuf NumElements refer to the number of elements 
    // within the vertex and color items ie 3 in both cases

    glBindBuffer (GL_ARRAY_BUFFER, m_vertices);
    glVertexAttribPointer(vPosition, vbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vPosition);   

    glBindBuffer (GL_ARRAY_BUFFER, m_colors);
    glVertexAttribPointer(vColor, cbuf->getNumElements(), GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vColor);   

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);

    m_shader = new Shader(getShaderDir());
    m_program = m_shader->getId();
    m_mvp_location = m_shader->getMVPLocation();

    glUseProgram (m_program);       

    dump("Scene::init");
}


void Scene::draw(int width, int height)
{ 
    setupView(width, height);
    glBindVertexArray (m_vao);

    // count is the number of indices (which point at vertices) to form into triangle faces
    // in the single triangle Demo this would be 3 
    GLsizei count = m_indices_count ; 
    glDrawElements( GL_TRIANGLES, count, GL_UNSIGNED_INT, NULL ) ;
    m_draw_count += 1 ; 
}


void Scene::setupView(int width, int height)
{
    glm::mat4 MVP;
    glm::mat4 lookat;
    glm::mat4 projection;
    glm::mat4 M2W ;
    glm::mat4 scale ;

    glm::mat4 shunt_to_look ;
    glm::mat4 rotate_around_look ;
    glm::mat4 shunt_back_to_eye ;

    glm::vec4 gaze ;

    // view inputs are in model coordinates (model coordinates are all within -1:1)
    // model_to_world matrix constructed from geometry center and extent
    // is used to construct the lookat matrix 
   
    //m_view->setEye(0,0,1);
    //m_view->setLook(0,0,0);
    //m_view->setUp(0,1,0);

    //scale = glm::scale(glm::vec3(1.0f/1000.f))  ;
    M2W = glm::make_mat4(m_model_to_world);
    lookat = m_view->getLookAt(M2W, m_draw_count == 0);
    gaze = m_view->getGaze(M2W, m_draw_count == 0);

    float gazelen = glm::length(gaze);

    m_camera->setSize(width, height);    

    projection = m_camera->getProjection();

    // # look is at (0,0,-gazelen) in eye frame, so here we shunt to the look and back again

    shunt_to_look = glm::translate( glm::mat4(1.), glm::vec3(0,0,gazelen));

    //rotate_around_look = m_trackball->getOrientationMatrix();
    rotate_around_look = m_trackball->getCombinedMatrix();

    shunt_back_to_eye = glm::translate( glm::mat4(1.), glm::vec3(0,0,-gazelen));


    MVP = projection * shunt_back_to_eye * rotate_around_look * shunt_to_look * lookat ;

    glUniformMatrix4fv(m_mvp_location, 1, GL_FALSE, glm::value_ptr(MVP));

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


    if(m_draw_count == 0)
    {
        m_trackball->Summary("Scene::setupView m_trackball");
        m_camera->Summary("Scene::setupView m_camera");
        m_view->Summary("Scene::setupView m_view");

        print(M2W, "M2W");
        print(lookat, "lookat");
        print(projection, "projection");
        print(MVP, "MVP");
        print(gaze, "gaze");
        printf("gaze length %10.3f \n", glm::length(gaze));
    }
    else if (m_draw_count % 100 == 0 )
    {
        m_trackball->Summary("Scene::setupView m_trackball");
    }

}



