#include "GTestBox.hh"
#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GMesh.hh"
#include "GBndLib.hh"
#include "GVector.hh"

#include "NLog.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"

#include <boost/lexical_cast.hpp>

const char* GTestBox::DEFAULT_CONFIG = 
    "frame=3153;"
    "boundary=MineralOil/Rock//;"
    ;


const char* GTestBox::FRAME_ = "frame"; 
const char* GTestBox::BOUNDARY_ = "boundary"; 


void GTestBox::configure(const char* config_)
{
    m_config = config_ ? strdup(config_) : DEFAULT_CONFIG ; 

    std::string config(m_config);
    typedef std::pair<std::string,std::string> KV ; 
    std::vector<KV> ekv = ekv_split(config.c_str(),';',"=");

    printf("GTestBox::configure %s \n", config.c_str() );
    for(std::vector<KV>::const_iterator it=ekv.begin() ; it!=ekv.end() ; it++)
    {
        printf(" %20s : %s \n", it->first.c_str(), it->second.c_str() );
        set(getParam(it->first.c_str()), it->second.c_str());
    }
}


GTestBox::Param_t GTestBox::getParam(const char* k)
{
    Param_t param = UNRECOGNIZED ; 
    if(     strcmp(k,FRAME_)==0)          param = FRAME ; 
    else if(strcmp(k,BOUNDARY_)==0)       param = BOUNDARY ; 
    return param ;   
}


void GTestBox::set(Param_t p, const char* s)
{
    switch(p)
    {
        case FRAME          : setFrame(s)          ;break;
        case BOUNDARY       : setBoundary(s)       ;break;
        case UNRECOGNIZED   :
                    LOG(warning) << "GTestBox::set WARNING ignoring unrecognized parameter " ;
    }
}


void GTestBox::dump(const char* msg)
{
    LOG(info) << msg  
              << " config " << m_config 
              << " boundary " << m_boundary 
              ; 

    if(m_mesh)
    {
       m_mesh->Dump(); 
    }
}



void GTestBox::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}

void GTestBox::setBoundary(const char* s)
{
    GBndLib* blib = m_ggeo->getBndLib();
    m_boundary = blib->addBoundary(s); 
}

void GTestBox::make()
{
    GMergedMesh* mesh0 = m_ggeo->getMergedMesh(0);
    assert(mesh0);

    gbbox bb = mesh0->getBBox( m_frame.x );   
    unsigned int index = 1000 ; // TODO: find a propa index
    m_mesh = makeMesh(index, bb ); 
}



GMesh* GTestBox::makeMesh(unsigned int index, gbbox& bb)
{
    gfloat3* vertices = new gfloat3[NUM_VERTICES] ;
    guint3* faces = new guint3[NUM_FACES] ;
    gfloat3* normals = new gfloat3[NUM_VERTICES] ;

    GBBoxMesh::twentyfour(bb, vertices, faces, normals );

    GMesh* mesh = new GMesh(index, vertices, NUM_VERTICES,  
                                   faces, NUM_FACES,    
                                   normals,  
                                   NULL ); // texcoords

    mesh->setColors(  new gfloat3[NUM_VERTICES]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}





