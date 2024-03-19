#pragma once

struct NP ; 
struct SMesh ; 

/**
SGLFW_Render
-------------

Try same prog for multiple mesh 

**/
struct SGLFW_Render
{
    const SMesh*         mesh ;  
    const SGLFW_Program* prog ;  

    SGLFW_Buffer*  vtx ; 
    SGLFW_Buffer*  nrm ;
    SGLFW_Buffer*  ins ;
    int num_inst ; 

    SGLFW_VAO*     vao ;  
    SGLFW_Buffer*  idx ; 
    int   render_count ; 

    SGLFW_Render(const SMesh* mesh, const SGLFW_Program* prog ) ; 
    void set_inst(const NP* _inst );
    void set_inst(int _num_inst, const float* values );

    void render(); 
    void render_drawElements() const ; 
};

inline SGLFW_Render::SGLFW_Render(const SMesh* _mesh, const SGLFW_Program* _prog )
    :
    mesh(_mesh),
    prog(_prog),
    vtx(nullptr),
    nrm(nullptr),
    ins(nullptr),
    num_inst(0),    
    vao(nullptr),
    idx(nullptr),
    render_count(0)
{
    vtx = new SGLFW_Buffer( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    vtx->bind();
    vtx->upload(); 

    nrm = new SGLFW_Buffer( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    nrm->bind();
    nrm->upload(); 

    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    vao->bind(); 

    idx = new SGLFW_Buffer( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    idx->bind();
    idx->upload(); 
}

inline void SGLFW_Render::set_inst(const NP* _inst )
{
    if(_inst == nullptr) return ; 
    assert( _inst->uifc == 'f' ); 
    assert( _inst->ebyte == 4 ); 
    assert( _inst->has_shape(-1,4,4)); 
    set_inst( _inst->num_items(), _inst->cvalues<float>() ); 
}

inline void SGLFW_Render::set_inst(int _num_inst, const float* values )
{
    num_inst = _num_inst ; 
    int num_bytes = num_inst*4*4*sizeof(float); 
    ins = new SGLFW_Buffer( num_bytes, values, GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    ins->bind();
    ins->upload(); 
}


/**
SGLFW_Render::render
---------------------

NB: careful that the intended buffer is bound (making it the active GL_ARRAY_BUFFER)
when the vertex attrib is enabled.  Getting this wrong can for example easily cause
normals to appear in position slots causing perplexing renders. 

**/

inline void SGLFW_Render::render()
{
   prog->use(); 
   vao->bind(); 

   vtx->bind();
   prog->enableVertexAttribArray( prog->vtx_attname, SMesh::VTX_SPEC ); 

   nrm->bind();
   prog->enableVertexAttribArray( prog->nrm_attname, SMesh::NRM_SPEC ); 

   if(ins)
   {
       ins->bind(); 
       prog->enableVertexAttribArray_OfTransforms( prog->ins_attname ) ; 
   }

   idx->bind();
   prog->updateMVP();

   render_drawElements(); 
   render_count += 1 ; 
}

inline void SGLFW_Render::render_drawElements() const 
{
    GLenum mode = GL_TRIANGLES ; 
  	GLsizei count = mesh->indices_num() ;  // number of elements to render (eg 3 for 1 triangle)
  	GLenum type = GL_UNSIGNED_INT ; 
  	const void * indices = (GLvoid*)(sizeof(GLuint) * mesh->indices_offset() ) ;
  	GLsizei instancecount = ins ? num_inst : 0 ; 

    if(instancecount > 0)
    {

#ifdef __APPLE__
        glDrawElementsInstanced(mode, count, type, indices, instancecount );
        if(render_count < 10 ) std::cout 
            << "SGLFW_Render::render_drawElements.glDrawElementsInstanced" 
            << " render_count " << render_count
            << " instancecount " << instancecount
            << std::endl
            ;

#else
        GLint basevertex = 0 ; 
        GLuint baseinstance = 0 ; 
        glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance ); 
        // SEGV on laptop, OK on worktation 
        // https://github.com/moderngl/moderngl/issues/346
        if(render_count < 10 ) std::cout 
            << "SGLFW_Render::render_drawElements.glDrawElementsInstancedBaseVertexBaseInstance" 
            << " render_count " << render_count
            << " instancecount " << instancecount
            << std::endl
            ;
#endif 


    }
    else
    {
        glDrawElements(mode, count, type, indices );
    }
}



