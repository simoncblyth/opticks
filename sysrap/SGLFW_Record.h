#pragma once
/**
SGLFW_Record.h : 
==============================================================================


**/


struct NP ; 
struct SGLFW_Program ; 
struct SRecordInfo ; 
struct sfr;

struct SGLFW_Record
{
    bool  dump ; 
    
    const SRecordInfo*     sr ;
    SGLFW_Buffer*  rpos ; 
    SGLFW_VAO*     vao ;  
    
    glm::vec4      param;
    GLint          param_location;

    SGLFW_Record(const SRecordInfo* recorder) ; 
    void init(); 


    void render(const SGLFW_Program* prog); 
};

inline SGLFW_Record::SGLFW_Record(const SRecordInfo* _sr )
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



**/


inline void SGLFW_Record::init()
{
    
     
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


**/

inline void SGLFW_Record::render(const SGLFW_Program* prog)
{
    
    param_location = prog->getUniformLocation("Param"); 
    prog->use(); 
    vao->bind(); 
 
    rpos->bind();
    prog->enableVertexAttribArray("rpos" ,SRecordInfo::RPOS_SPEC ); 
    //prog->locateMVP("ModelViewProjection", gm->MVP_ptr ); 
 
 
    param.w += param.z ;  // input propagation time 
    if( param.w > param.y ) param.w = param.x ;  // input time : Param.w from .x to .y with .z steps
    
    //gl.UniformMatrix4fv( gl.mvp_location, mvp );  
    if(param_location > -1 ) prog->Uniform4fv(param_location, glm::value_ptr(param), false );
    prog->updateMVP();
    
    GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;  
    glDrawArrays(mode, sr->record_first,  sr->record_count);
    
}




