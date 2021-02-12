#include "PLOG.hh"
#include "NPY.hpp"
#include "SensorLib.hh"
#include "OCtx.hh"  // the watertight wrapper, not the old OContext 
#include "OSensorLib.hh"

const plog::Severity OSensorLib::LEVEL = PLOG::EnvLevel("OSensorLib", "DEBUG"); 

const char* OSensorLib::TEXID       = "OSensorLib_texid"  ; 
const char* OSensorLib::SENSOR_DATA = "OSensorLib_sensor_data"  ; 

OSensorLib::OSensorLib(const OCtx* octx, const SensorLib* sensorlib)
    :    
    m_octx(octx),
    m_sensorlib(sensorlib),
    m_sensor_data(m_sensorlib->getSensorDataArray()),
    m_angular_efficiency(m_sensorlib->getSensorAngularEfficiencyArray()),
    m_num_dim(   m_angular_efficiency ? m_angular_efficiency->getNumDimensions() : 0),
    m_num_cat(   m_angular_efficiency ? m_angular_efficiency->getShape(0) : 0),
    m_num_theta( m_angular_efficiency ? m_angular_efficiency->getShape(1) : 0),
    m_num_phi(   m_angular_efficiency ? m_angular_efficiency->getShape(2) : 0),
    m_num_elem(  m_angular_efficiency ? m_angular_efficiency->getShape(3) : 0),
    m_texid( NPY<int>::make(m_num_cat, 4) )    // small buffer of texid, NB empty when no angular efficiency     
{
    init(); 
}


void OSensorLib::init() 
{
    assert( m_sensorlib->isClosed() ); 

    if(!m_sensor_data) LOG(fatal) << " sensor_data NULL " ; 
    assert( m_sensor_data ); 

    if( m_angular_efficiency )
    {
        assert( m_num_dim == 4 ); 
        assert( m_num_cat < 10 ); 
        assert( m_num_elem == 1 ); 
        assert( m_texid ); 
    }
    m_texid->zero();
}




const NPY<float>*  OSensorLib::getSensorAngularEfficiencyArray() const 
{
    return m_angular_efficiency ; 
}

unsigned OSensorLib::getNumSensorCategories() const 
{
    return m_num_cat ; 
}
unsigned OSensorLib::getNumTheta() const 
{
    return m_num_theta ; 
}
unsigned OSensorLib::getNumPhi() const 
{
    return m_num_phi ; 
}
unsigned OSensorLib::getNumElem() const 
{
    return m_num_elem ; 
}


int OSensorLib::getTexId(unsigned icat) const
{
    assert( icat < m_num_cat );   
    glm::ivec4 q = m_texid->getQuad_(icat) ;
    return q.x ;  
}

const OCtx* OSensorLib::getOCtx() const 
{
    return m_octx ; 
}

void OSensorLib::convert()
{
    LOG(LEVEL) << "[" ; 
    makeSensorDataBuffer() ;
    makeSensorAngularEfficiencyTexture() ; 
    LOG(LEVEL) << "]" ; 
}

void OSensorLib::makeSensorDataBuffer()
{
    LOG(LEVEL) << "[" ; 
    const char* key = SENSOR_DATA ; 
    char type = 'I' ;          // I:INPUT
    char flag = ' ' ;          // default 
    unsigned item = -1 ;       // whole array in one GPU buffer
    bool transpose = true ;    // doesnt matter for 1d buffer 
    m_octx->create_buffer(m_sensor_data, key, type, flag, item, transpose ); 
    LOG(LEVEL) << "] m_sensor_data " << m_sensor_data->getShapeString() << " upload to " << key ;   
}

void OSensorLib::makeSensorAngularEfficiencyTexture()
{
    LOG(LEVEL) << "[ m_num_cat " << m_num_cat  ; 
    const char* config = "INDEX_NORMALIZED_COORDINATES" ; 
    for(unsigned i=0 ; i < m_num_cat ; i++)
    {
         const char* key = NULL ;  // no-key as cannot do normal reads from tex buffers 
         char type = 'I' ;         // I:INPUT
         char flag = ' ' ; 
         unsigned item = i ; 
         bool transpose = true ; 
         void* buffer_ptr = m_octx->create_buffer(m_angular_efficiency, key, type, flag, item, transpose ); 
         unsigned tex_id = m_octx->create_texture_sampler(buffer_ptr, config );
         LOG(LEVEL) << " item " << i << " tex_id " << tex_id ; 

         glm::ivec4 q(tex_id, 0,0,0);  // placeholder zeros: eg for dimensions or ranges 
         m_texid->setQuad_(q, i); 
    }

    // create GPU buffer and upload small texid array into it 
    m_octx->create_buffer(m_texid, TEXID, 'I', ' ', -1, true ); 
    LOG(LEVEL) << "] m_texid " << m_texid->getShapeString() << " upload to " << TEXID ;   
}


