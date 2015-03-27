#include <GL/glew.h>

#include "Scene.hh"
#include "Shader.hh"
#include "Camera.hh"
#include "View.hh"
#include "Common.hh"

// ggeo
#include "GArray.hh"
#include "GBuffer.hh"
#include "GDrawable.hh"

#include "stdio.h"
#include "stdlib.h"

#include <glm/glm.hpp>  
#include <glm/gtc/type_ptr.hpp>


Scene::Scene()
 
  :
  m_shader(NULL),
  m_shaderdir(NULL),
  m_camera(NULL),
  m_view(NULL),
  m_geometry(NULL),
  m_draw_count(0)

{
  m_camera = new Camera(1024, 768) ;
  m_view   = new View() ;
}

Scene::~Scene()
{
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
    GLuint id ; 
    glGenBuffers(1, &id);
    glBindBuffer(target, id);
    glBufferData(target, buffer->getNumBytes(), buffer->getPointer(), usage);
    return id ; 
}




void Scene::dump(const char* msg)
{
    printf("%s\n", msg );
    printf("nelem     %d \n", m_nelem);
    printf("vertices  %u \n", m_vertices);
    printf("colors    %u \n", m_colors);
    printf("indices   %u \n", m_indices);
    printf("shaderdir %s \n", getShaderDir());

    m_shader->dump(msg);
}

void Scene::init()
{
    assert(m_geometry);

    glGenVertexArrays (1, &m_vao); // OSX: undefined without glew 
    glBindVertexArray (m_vao);     

    m_nelem    = m_geometry->getNumFaces();
    // hmm the above duplicates info from the GBuffer ?  eliminate or assert on consistency

    m_vertices = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  m_geometry->getVerticesBuffer());
    m_colors   = upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW,  m_geometry->getColorsBuffer());
    m_indices  = upload(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, m_geometry->getFacesBuffer());

    GLboolean normalized = GL_FALSE ; 
    GLsizei stride = 0 ;
    const GLvoid* offset = NULL ;
 
    // as there are two GL_ARRAY_BUFFER for vertices and colors need
    // to bind them again (despite bound in upload) in order to 
    // make the desired one active when create the VertexAttribPointer :
    // the currently active buffer being recorded "into" the VertexAttribPointer 

    glBindBuffer (GL_ARRAY_BUFFER, m_vertices);
    glVertexAttribPointer(vPosition, m_nelem, GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vPosition);   

    glBindBuffer (GL_ARRAY_BUFFER, m_colors);
    glVertexAttribPointer(vColor, m_nelem, GL_FLOAT, normalized, stride, offset);
    glEnableVertexAttribArray (vColor);   

    glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, m_indices);

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

    m_shader = new Shader(getShaderDir());
    m_program = m_shader->getId();
    m_mvp_location = m_shader->getMVPLocation();

    glUseProgram (m_program);       

    dump("Scene::init");
}


void Scene::setupView(int width, int height)
{
    bool debug = m_draw_count == 0 ; 

    m_camera->setSize(width, height);    
    m_camera->setParallel(true);
    m_view->setEye(0,0,-2);

    glm::mat4 identity ;
    glm::mat4 projection;
    glm::mat4 modelview;

    projection = m_camera->getProjection();

    GBuffer* m2w_buf = m_geometry->getModelToWorldBuffer();
    assert(m2w_buf->getNumBytes() == 16*sizeof(float));
    glm::mat4 m2w = glm::make_mat4( (float*)m2w_buf->getPointer() );

    glm::mat4 lka = m_view->getLookAt(m2w, debug);

    modelview *= lka ; 

    glm::mat4 MVP = projection * modelview ;

    GLsizei count = 1; 
    GLboolean transpose = GL_FALSE ; 
    //glUniformMatrix4fv(m_mvp_location, count, transpose, glm::value_ptr(MVP));
    glUniformMatrix4fv(m_mvp_location, count, transpose, glm::value_ptr(identity));

    if(debug)
    {
        m_camera->Summary("Scene::setupView m_camera");
        m_view->Summary("Scene::setupView m_view");

        print(projection, "projection");
        print(m2w, "m2w");
        print(lka, "lka");
        print(modelview, "modelview");
        print(MVP, "MVP");
    }
}


void Scene::draw(int width, int height)
{ 
    setupView(width, height);


    glBindVertexArray (m_vao);
    glDrawElements( GL_TRIANGLES, m_nelem, GL_UNSIGNED_INT, NULL ) ;
    m_draw_count += 1 ; 
}


