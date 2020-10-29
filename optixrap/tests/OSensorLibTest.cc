// om-;TEST=OSensorLibTest om-t 
#include "OKConf.hh"
#include "SStr.hh"
#include "NPY.hpp"
#include "SensorLib.hh"
#include "OCtx.hh"
#include "OSensorLib.hh"
#include "OPTICKS_LOG.hh"

const char* CMAKE_TARGET = "OSensorLibTest" ; 
const char* PTXPATH = OKConf::PTXPath(CMAKE_TARGET, SStr::Concat(CMAKE_TARGET, ".cu" ), "tests" );      

struct OSensorLibTest 
{
    OCtx*       m_octx ; 
    OSensorLib* m_osenlib ;
 
    OSensorLibTest(const SensorLib* senlib);

    NPY<float>* duplicateAngularEfficiency() const ; 
    void        compareToOriginal( const NPY<float>* out ) const ;
};

OSensorLibTest::OSensorLibTest(const SensorLib* senlib)
    :
    m_octx(OCtx::Get()),
    m_osenlib(new OSensorLib(m_octx, senlib))    
{
    // 0. creates textures for each sensor category and small texid buffer 
    m_osenlib->convert();  
}

/**
OSensorLibTest::duplicateAngularEfficiency
---------------------------------------------

Recreate the original angular efficiency array by making GPU texture queries at all 3d points.

**/


NPY<float>* OSensorLibTest::duplicateAngularEfficiency() const 
{
    // 1. create "out" array shaped just like the sensor lib angular efficiency array 

    unsigned num_cat   = m_osenlib->getNumSensorCategories();
    unsigned num_theta = m_osenlib->getNumTheta(); 
    unsigned num_phi   = m_osenlib->getNumPhi(); 
    unsigned num_elem  = m_osenlib->getNumElem(); 

    assert( num_cat < 10 );  
    assert( num_elem == 1 ); 

    NPY<float>* out = NPY<float>::make(num_cat, num_theta, num_phi, num_elem ); 

    // 2. create corresponding GPU output buffer, with transposed dimensions  

    const char* key = "OSensorLibTest_out" ; 
    bool transpose = true ; 
    m_octx->create_buffer(out, key, 'O', ' ', -1, transpose );
  
    // 3. setup pipeline and launch  

    unsigned entry_point_index = 0u ; 
    m_octx->set_raygen_program(    entry_point_index, PTXPATH, "raygen" );
    m_octx->set_exception_program( entry_point_index, PTXPATH, "exception" );

    unsigned l0 = transpose ? num_phi   : num_cat   ; 
    unsigned l1 = transpose ? num_theta : num_theta ; 
    unsigned l2 = transpose ? num_cat   : num_phi   ; 
    assert( transpose == true );  // see OCtx{2,3}dTest.cc to understand why transpose=true is necessary  
    LOG(info) << " launch (l0,l1,l2) (" << l0 << "," << l1 << "," << l2 << ")" ; 

    m_octx->launch( entry_point_index, l0, l1, l2 );

    // 4. allocate array and copy GPU buffer into it 

    out->zero();
    m_octx->download_buffer(out, key, -1);

    return out ;
} 

void OSensorLibTest::compareToOriginal( const NPY<float>* dupe ) const 
{
    const NPY<float>* original = m_osenlib->getSensorAngularEfficiencyArray(); 
    unsigned mismatch = NPY<float>::compare( original, dupe, true ); 
    LOG(info) << " mismatch " << mismatch ; 
    assert( mismatch == 0 ); 
}


void test_duplication(const OSensorLibTest* oslt)
{
    NPY<float>* dupe = oslt->duplicateAngularEfficiency(); 
    dupe->dump(); 
    const char* path = "$TMP/optixrap/tests/OSensorLibTest/dupe.npy" ; 
    LOG(info) << "dupe " << dupe->getShapeString() << " saving to " << path ; 
    dupe->save(path); 
    oslt->compareToOriginal(dupe); 
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
    senlib->dump("OSensorLibTest"); 

    OSensorLibTest oslt(senlib); 

    test_duplication(&oslt); 

    return 0 ; 
}
// om-;TEST=OSensorLibTest om-t 
