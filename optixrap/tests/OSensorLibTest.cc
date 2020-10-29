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

    OCtx* octx = OCtx::Get() ; 
    OSensorLib osenlib(octx, senlib);    
    osenlib.convert();  // creates textures for each sensor category 

    unsigned entry_point_index = 0u ; 
    octx->set_raygen_program(    entry_point_index, PTXPATH, "raygen" );
    octx->set_exception_program( entry_point_index, PTXPATH, "exception" );

    unsigned num_cat   = osenlib.getNumSensorCategories();
    unsigned num_theta = osenlib.getNumTheta(); 
    unsigned num_phi   = osenlib.getNumPhi(); 
    unsigned num_elem  = osenlib.getNumElem(); 

    assert( num_cat == 1 );  
    assert( num_elem == 1 ); 

    NPY<int>* texid = NPY<int>::make(num_cat, 4);   // small buffer of texid    
    texid->zero(); 

    for(unsigned icat=0 ; icat < num_cat ; icat++)
    {
        int tex_id = osenlib.getTexId(icat);   
        glm::ivec4 tquad( tex_id, 0,0,0);   // placeholder zeros: eg for dimensions or ranges 
        texid->setQuad_(tquad, icat);
    }

    octx->create_buffer(texid, "texid_buffer", 'I', ' ', -1); // upload the texid array into the GPU buffer

    NPY<float>* out = NPY<float>::make(num_cat, num_theta, num_phi, num_elem ); 

    bool transpose_buffer = true ; 
    octx->create_buffer(out, "output_buffer", 'O', ' ', -1, transpose_buffer );
  
    unsigned l0 = transpose_buffer ? num_phi    :  num_cat   ; 
    unsigned l1 = transpose_buffer ? num_theta  :  num_theta ; 
    unsigned l2 = transpose_buffer ? num_cat    :  num_phi   ; 
    assert( transpose_buffer == true );
    // see OCtx{2,3}dTest.cc to understand why transpose_buffer is necessary  

    LOG(info) << " launch (l0,l1,l2) (" << l0 << "," << l1 << "," << l2 << ")" ; 
    octx->launch( entry_point_index, l0, l1, l2 );

    out->zero();
    octx->download_buffer(out, "output_buffer", -1);
    out->dump(); 

    const char* path = "$TMP/optixrap/tests/OSensorLibTest/out.npy" ; 
    LOG(info) << " output array " << out->getShapeString() << " saving to " << path ; 
    out->save(path); 
    return 0 ; 
}

// om-;TEST=OSensorLibTest om-t 
