#pragma once
/**
SGLFW_Record.h : create OpenGL buffers with SMesh and instance data and render
==============================================================================

Canonical use is from SGLFW_Scene::initMesh

* (in a former incarnation this was called SGLFW_Render)

**/


struct NP ; 
struct SGLFW_Program ; 
struct SRecorder ; 
struct sfr;

struct SGLFW_Record
{
    bool  dump ; 
    
    const SRecorder*     sr ;
    SGLFW_Buffer*  rpos ; 
    SGLFW_VAO*     vao ;  
    
    glm::vec4      param;
    GLint          param_location;

    SGLFW_Record(const SRecorder* recorder) ; 
    void init(); 


    void render(const SGLFW_Program* prog); 
    //void render_drawElements() const ; 
};

inline SGLFW_Record::SGLFW_Record(const SRecorder* _sr )
    :
    dump(false),
    sr(_sr),
    rpos(nullptr),
    vao(nullptr),
    param(0.f,0.f,0.f,0.f),
    param_location(0)
{
    init(); 
}

/**
SGLFW_Record::init
----------------

Creates vtx, nrm, idx OpenGL buffers using SGLFW_Buffers


**/


inline void SGLFW_Record::init()
{
    
    //sfr fr;
    //fr.set_ce(&((sr->ce).x));
    //gm->set_frame(fr);
    //gm->dump();
     
    float t0 = sr->get_t0(); 
    float t1 = sr->get_t1(); 
    float ts = sr->get_ts(); 
    param = glm::vec4(t0, t1, (t1-t0)/ts, t0); 

    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not GL_ARRAY_BUFFER)
    vao->bind(); 
    
    rpos = new SGLFW_Buffer( sr->record->arr_bytes(), sr->record->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    rpos->bind();
    rpos->upload(); 



}




/**
SGLFW_Record::render
---------------------

Use argument prog to render the mesh 

NB: careful that the intended buffer is bound (making it the active GL_ARRAY_BUFFER)
when the vertex attrib is enabled.  Getting this wrong can for example easily cause
normals to appear in position slots causing perplexing renders. 

**/

inline void SGLFW_Record::render(const SGLFW_Program* prog)
{
    
    param_location = prog->getUniformLocation("Param"); 
    prog->use(); 
    vao->bind(); 
 
    rpos->bind();
    prog->enableVertexAttribArray("rpos" ,SRecorder::RPOS_SPEC ); 
    //prog->locateMVP("ModelViewProjection", gm->MVP_ptr ); 
 
 
    param.w += param.z ;  // input propagation time 
    if( param.w > param.y ) param.w = param.x ;  // input time : Param.w from .x to .y with .z steps
    
    //gl.UniformMatrix4fv( gl.mvp_location, mvp );  
    if(param_location > -1 ) prog->Uniform4fv(param_location, glm::value_ptr(param), false );
    prog->updateMVP();
    
    GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;  
//    std::cout<<" GL_LINE_STRIP " << GL_LINE_STRIP
//             <<" GL_POINTS " << GL_POINTS
//             <<" mode = " <<mode
//             << std::endl;
    glDrawArrays(mode, sr->record_first,  sr->record_count);
    
    //render_drawElements(); 
}

//inline void SGLFW_Record::render_drawElements() 
//{
//
//}



