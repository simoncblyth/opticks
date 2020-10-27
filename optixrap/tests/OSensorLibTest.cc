#include "OKConf.hh"
#include "SStr.hh"
#include "NPY.hpp"
#include "SensorLib.hh"
#include "OCtx.hh"
#include "OSensorLib.hh"
#include "OPTICKS_LOG.hh"

const char* CMAKE_TARGET = "OSensorLibTest" ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* dir = "$TMP/opticksgeo/tests/MockSensorLibTest" ;

    SensorLib* senlib = SensorLib::Load(dir); 

    if( senlib == NULL )
    {
        LOG(fatal) << " FAILED to load from " << dir ; 
        return 0 ;
    }

    senlib->dump("OSensorLibTest"); 


    OCtx* octx = new OCtx ; 

    OSensorLib osenlib(octx, senlib);    
    osenlib.convert(); 

    const char* ptxpath = OKConf::PTXPath(CMAKE_TARGET, SStr::Concat(CMAKE_TARGET, ".cu" ), "tests" );      
    LOG(info) << " ptxpath " << ptxpath ; 

    unsigned entry_point_index = 0u ; 
    octx->set_raygen_program(    entry_point_index, ptxpath, "raygen" );
    octx->set_exception_program( entry_point_index, ptxpath, "exception" );

    unsigned num_cat  = osenlib.getNumSensorCategories();
    unsigned height   = osenlib.getNumTheta(); 
    unsigned width    = osenlib.getNumPhi(); 
    unsigned num_elem = osenlib.getNumElem(); 

    assert( num_cat == 1 );  
    assert( num_elem == 1 ); 

    unsigned icat = 0 ; 
    int tex_id = osenlib.getTexId(icat);  // need this cat->tex_id done GPU side, a small buffer? 
    octx->set_context_int("tex_id",  tex_id);  

/**
p19 of OptiX 5.0 PDF : buffer of tex_id 
**/


    NPY<float>* out = NPY<float>::make(num_cat, height, width, num_elem ); 

    void* outBuf = octx->create_buffer(out, "output_buffer", 'O', ' ', icat);
    assert( outBuf != NULL );  

    double t_prelaunch ; 
    double t_launch ;
    octx->launch_instrumented( entry_point_index, width, height, t_prelaunch, t_launch );

    std::cout
         << " prelaunch " << std::setprecision(4) << std::fixed << std::setw(15) << t_prelaunch
         << " launch    " << std::setprecision(4) << std::fixed << std::setw(15) << t_launch
         << std::endl
         ;

    out->zero();
    octx->download_buffer(out, "output_buffer", -1);
    out->dump(); 
    const char* path = "$TMP/optixrap/tests/OSensorLibTest.npy" ; 
    out->save(path); 

    return 0 ; 
}

