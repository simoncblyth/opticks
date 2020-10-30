// om-;TEST=OSensorLibGeoTest om-t 

#include "OKConf.hh"
#include "SStr.hh"
#include "NPY.hpp"
#include "SphereOfTransforms.hh"
#include "SensorLib.hh"
#include "OCtx.hh"
#include "OSensorLib.hh"
#include "OPTICKS_LOG.hh"

const char* CMAKE_TARGET = "OSensorLibTest" ; 
const char* PTXPATH = OKConf::PTXPath(CMAKE_TARGET, SStr::Concat(CMAKE_TARGET, ".cu" ), "tests" );      


class OSensorLibGeoTest 
{
    private:
        OCtx*       m_octx ; 
        OSensorLib* m_osenlib ;
        float       m_radius ; 
        unsigned    m_num_theta ; 
        unsigned    m_num_phi ; 
        NPY<float>* m_transforms ; 
    public: 
        OSensorLibGeoTest(const SensorLib* senlib);
    private:
        void init(); 
        void initGeometry(); 
};

OSensorLibGeoTest::OSensorLibGeoTest(const SensorLib* senlib)
    :
    m_octx(OCtx::Get()),
    m_osenlib(new OSensorLib(m_octx, senlib)),
    m_radius(1000.f),
    m_num_theta(5),
    m_num_phi(8),    
    m_transforms(SphereOfTransforms::Make(m_radius, m_num_theta, m_num_phi))
{

    init(); 
}

void OSensorLibGeoTest::init()
{
    // 0. creates GPU textures for each sensor category + small texid buffer 
    m_osenlib->convert();  
    initGeometry(); 
}

void OSensorLibGeoTest::initGeometry()
{
    LOG(info) << "transforms " << m_transforms->getShapeString() ; 

}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    // -1. load mock SensorLib

    const char* dir = "$TMP/opticksgeo/tests/MockSensorLibTest" ;
    SensorLib* senlib = SensorLib::Load(dir); 
    if( senlib == NULL )
    {
        LOG(fatal) << " FAILED to load from " << dir ; 
        return 0 ;
    }
    senlib->dump("OSensorLibGeoTest"); 

    OSensorLibGeoTest oslgt(senlib); 


    return 0 ; 
}
// om-;TEST=OSensorLibGeoTest om-t 
