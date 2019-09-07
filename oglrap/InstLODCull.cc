/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <GL/glew.h>

#include "G.hh"
#include "RBuf.hh"
#include "RBuf4.hh"
#include "Composition.hh"
#include "Prog.hh"
#include "InstLODCull.hh"

#include "PLOG.hh"


const unsigned InstLODCull::INSTANCE_MINIMUM = 10000 ; 
const unsigned InstLODCull::QSIZE = sizeof(float)*4 ;

const unsigned InstLODCull::LOC_InstanceTransform = 4 ;

InstLODCull::~InstLODCull()
{
}


InstLODCull::InstLODCull(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path, true),
    m_src(NULL),
    m_dst(NULL),
    m_num_instance(0),
    m_num_lod(0),
    m_launch_count(0)
{
}


std::string InstLODCull::desc() const 
{
    std::stringstream ss ; 
    ss << " InstLODCull"
       << " verbosity " << m_verbosity 
       << " num_instance " << m_num_instance
       << " num_lod " << m_num_lod
       ;

    return ss.str();
}


void InstLODCull::setupFork(RBuf* src, RBuf4* dst, RBuf4* dst_devnull )
{    
    // invoked from Renderer::upload, for each LOD level generate output count queries 
    // and bind tranform feedback stream output buffers, 
    // create single forking VAO

    m_src = src ; 
    m_dst = dst ; 
    m_dst_devnull = dst_devnull ;
 
    m_num_instance = src->getNumItems();
    m_num_lod = dst->num_buf() ; 

    assert( m_num_instance > INSTANCE_MINIMUM );
    assert( m_num_lod <= LOD_MAX && m_num_lod > 0 );

    LOG(fatal) << "InstLODCull::setupFork" << desc() ;  

    initShader();

    for(unsigned i=0 ; i < m_num_lod ; i++) glGenQueries(1, &m_lodQuery[i]);

    m_forkVAO = createForkVertexArray(m_src, m_dst);

    m_workaroundVAO = m_dst_devnull ? createForkVertexArray(m_src, m_dst_devnull ) : 0 ;

    m_dst->dump();

}


void InstLODCull::applyFork()
{
    // http://rastergrid.com/blog/2010/10/gpu-based-dynamic-geometry-lod/

    if(m_verbosity > 1)
    LOG(info) << "InstLODCull::applyFork"
              << " m_num_lod " << m_num_lod
              << " m_num_instance " << m_num_instance
              ; 

    assert(m_src);

    glUseProgram(m_program);
    glBindVertexArray(m_forkVAO);

    glEnable(GL_RASTERIZER_DISCARD);

    for(unsigned i=0 ; i < m_num_lod ; i++)
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, m_lodQuery[i]  );

    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, m_num_instance );
    glEndTransformFeedback();

    for(unsigned i=0 ; i < m_num_lod ; i++)
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );

    glDisable(GL_RASTERIZER_DISCARD);

    for (unsigned i=0; i< m_num_lod; i++)
        glGetQueryObjectiv(m_lodQuery[i], GL_QUERY_RESULT, &m_dst->at(i)->query_count);
  
}


void InstLODCull::applyForkStreamQueryWorkaround()
{
/* 
    As investigated in tests/txfStream.cc

    Forking into 4 separate streams works, but the 
    query counts only work for stream 0 ? The others 
    yielding zero.

    Looks like a driver bug...

    This workaround run the transform 
    feedback again to get the counts for streams > 0

    In an attempt to minimize the time to do that 
    "devnull" zero-sized buffers are attached 
    to avoid data movement.

*/

    if(m_verbosity > 1)
    LOG(info) << "InstLODCull::applyForkStreamQueryWorkaround"
              << " m_src->num_items " << m_src->num_items
              ;

    glUseProgram(m_program);
    glBindVertexArray(m_workaroundVAO);

    unsigned i0 = 1 ; // no need to repeat stream 0 query, that one works

    // pointing tranform feedback at devnull : to avoid data movement 
    // is done via the m_workaroundVAO

    //for (unsigned i=i0; i < m_num_lod ; i++)
    //    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, m_dst_devnull->at(i)->id );

    glEnable(GL_RASTERIZER_DISCARD);
    for(unsigned i=i0 ; i < m_num_lod ; i++)
    {
        glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, m_lodQuery[i]  );
        glBeginTransformFeedback(GL_POINTS);
        glDrawArrays(GL_POINTS, 0, m_src->num_items );
        glEndTransformFeedback();
        glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );
    }
    glDisable(GL_RASTERIZER_DISCARD);

    for (unsigned i=i0 ; i < m_num_lod ; i++)
        glGetQueryObjectiv(m_lodQuery[i], GL_QUERY_RESULT, &m_dst->at(i)->query_count);

}



GLuint InstLODCull::createForkVertexArray(RBuf* src, RBuf4* dst) 
{
    G::ErrCheck("InstLODCull::createForkVertexArray.0", true);
    GLuint loc = LOC_InstanceTransform ;

    LOG(info) << "InstLODCull::createForkVertexArray"
              << " loc " << loc
              << " src.id " << src->id
              ;

    if(!src->isUploaded())
    {
        src->upload(GL_ARRAY_BUFFER, GL_STATIC_DRAW) ;
    } 
    assert(src->isUploaded());

    GLuint vertexArray;
    glGenVertexArrays(1, &vertexArray);
    glBindVertexArray(vertexArray);

    G::ErrCheck("InstLODCull::createForkVertexArray.1", true);

    unsigned num_buf = dst->num_buf();
    for (unsigned i=0; i< num_buf ; i++) 
    {
        glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, dst->at(i)->id );
        glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, i, dst->at(i)->id );
    }

    G::ErrCheck("InstLODCull::createForkVertexArray.2", true);
    glBindBuffer(GL_ARRAY_BUFFER, src->id ); // original transforms fed in 
    G::ErrCheck("InstLODCull::createForkVertexArray.3", true);

    GLint size = 4 ; 
    GLsizei stride = 4*QSIZE ; 

    glEnableVertexAttribArray(loc + 0); 
    glVertexAttribPointer(    loc + 0, size, GL_FLOAT, GL_FALSE, stride, (void*)(0*QSIZE) );

    glEnableVertexAttribArray(loc + 1); 
    glVertexAttribPointer(    loc + 1, size, GL_FLOAT, GL_FALSE, stride, (void*)(1*QSIZE) );

    glEnableVertexAttribArray(loc + 2); 
    glVertexAttribPointer(    loc + 2, size, GL_FLOAT, GL_FALSE, stride, (void*)(2*QSIZE) );

    glEnableVertexAttribArray(loc + 3); 
    glVertexAttribPointer(    loc + 3, size, GL_FLOAT, GL_FALSE, stride, (void*)(3*QSIZE) );

    // NB no divisor, are accessing instance transforms in a non-instanced manner to do the culling 

    G::ErrCheck("InstLODCull::createForkVertexArray.4", true);
    return vertexArray;
}
  
void InstLODCull::initShader()
{
    //if(m_verbosity > 1)
    LOG(info) << "InstLODCull::initShader START " << desc() ; 

    G::ErrCheck("InstLODCull::initShader.-2", true );
    create_shader();
    G::ErrCheck("InstLODCull::initShader.-1", true );
    setNoFrag(true);

    const char *vars[] = {
                           "VizTransform0LOD0",
                           "VizTransform1LOD0",
                           "VizTransform2LOD0",
                           "VizTransform3LOD0",
                           "gl_NextBuffer",
                           "VizTransform0LOD1",
                           "VizTransform1LOD1",
                           "VizTransform2LOD1",
                           "VizTransform3LOD1",
                           "gl_NextBuffer",
                           "VizTransform0LOD2",
                           "VizTransform1LOD2",
                           "VizTransform2LOD2",
                           "VizTransform3LOD2"
                         };

    glTransformFeedbackVaryings(m_program, 14, vars, GL_INTERLEAVED_ATTRIBS);
    G::ErrCheck("InstLODCull::initShader.0", true );

    glBindAttribLocation(m_program, LOC_InstanceTransform, "InstanceTransform");
    G::ErrCheck("InstLODCull::initShader.1", true );

    link_shader();
    G::ErrCheck("InstLODCull::initShader.2", true );

    if(m_verbosity > 1)
    LOG(info) << "InstLODCull::initShader DONE " << desc() ; 
} 


void InstLODCull::launch()
{   
    applyFork() ;
    applyForkStreamQueryWorkaround() ;  
    m_dst->bind(); // bind back the m_dst buffers after the workaround targetting m_dst_devnull

    //if(m_launch_count < 3) pullback() ;
    m_launch_count++ ;
}


void InstLODCull::pullback()
{
    LOG(info) << "InstLODCull::pullback launch_count " << m_launch_count ; 
    m_src->pullback(0);
    m_src->dump("InstLODCull::pullback.src");
    m_dst->pullback( "InstLODCull::pullback.m_dst");   
}



