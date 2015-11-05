#include "GTestBox.hh"
#include "GCache.hh"
#include "GMergedMesh.hh"
#include "GBBoxMesh.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GBndLib.hh"
#include "GVector.hh"
#include "GMatrix.hh"

#include "NLog.hpp"
#include "GLMFormat.hpp"
#include "stringutil.hpp"

#include <boost/lexical_cast.hpp>

const char* GTestBox::DEFAULT_CONFIG = 
    "frame=3153;"
    "size=2;"   // bbox enlargement in units of extent  
    "boundary=MineralOil/Rock//;"
    ;

const char* GTestBox::FRAME_ = "frame"; 
const char* GTestBox::BOUNDARY_ = "boundary"; 
const char* GTestBox::SIZE_ = "size"; 


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
    else if(strcmp(k,SIZE_)==0)           param = SIZE ; 
    return param ;   
}


void GTestBox::set(Param_t p, const char* s)
{
    switch(p)
    {
        case FRAME          : setFrame(s)          ;break;
        case BOUNDARY       : setBoundary(s)       ;break;
        case SIZE           : setSize(s)           ;break;
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

    if(m_solid)
    {
       m_solid->Summary(); 
    }
}

void GTestBox::setFrame(const char* s)
{
    std::string ss(s);
    m_frame = givec4(ss);
}

void GTestBox::setBoundary(const char* s)
{
    // passing in bndlib just for this seems funny
    m_boundary = m_bndlib->addBoundary(s); 
}

void GTestBox::setSize(const char* s)
{
    m_size = boost::lexical_cast<float>(s) ;
}


GMesh* GTestBox::makeMesh(gbbox& bb, unsigned int meshindex)
{
    gfloat3* vertices = new gfloat3[NUM_VERTICES] ;
    guint3* faces = new guint3[NUM_FACES] ;
    gfloat3* normals = new gfloat3[NUM_VERTICES] ;

    GBBoxMesh::twentyfour(bb, vertices, faces, normals );

    GMesh* mesh = new GMesh(meshindex, vertices, NUM_VERTICES,  
                                       faces, NUM_FACES,    
                                       normals,  
                                       NULL ); // texcoords

    mesh->setColors(  new gfloat3[NUM_VERTICES]);
    mesh->setColor(0.5,0.5,0.5);  

    return mesh ; 
}


GSolid* GTestBox::makeSolid(unsigned int nodeindex)
{
    GMatrixF* transform = new GMatrix<float>();

    GSolid* solid = new GSolid(nodeindex, transform, m_mesh, UINT_MAX, NULL );     

    solid->setBoundary(m_boundary);    // unlike ctor these setters creates corresponding indices array

    solid->setSensor( NULL );    

    return solid ; 
}


void GTestBox::make(gbbox& bb, unsigned int meshindex, unsigned int nodeindex)
{
    float factor = getSize();

    // hmm frame not currently used

    gbbox cbb(bb);
    cbb.enlarge( factor );

    LOG(info) << "GTestBox::make"
              << " factor " << factor 
              ;

    m_mesh = makeMesh(cbb, meshindex);
    m_solid = makeSolid( nodeindex );
}


