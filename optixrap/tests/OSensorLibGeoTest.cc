// om-;TEST=OSensorLibGeoTest om-t 

#include "OKConf.hh"
#include "SStr.hh"
#include "OPTICKS_LOG.hh"

#include "NPY.hpp"
#include "NGLMExt.hpp"
#include "ImageNPY.hpp"

#include "SphereOfTransforms.hh"
#include "SensorLib.hh"

#include "OCtx.hh"
#include "OSensorLib.hh"

const char* CMAKE_TARGET = "OSensorLibGeoTest" ; 
const char* MAIN_PTXPATH   = OKConf::PTXPath(CMAKE_TARGET, "OSensorLibGeoTest.cu", "tests" );      
const char* SPHERE_PTXPATH = OKConf::PTXPath(CMAKE_TARGET, "sphere.cu",            "tests" );

/**
OSensorLibGeoTest
===================

Creates a geometry of many spheres arranged at positions on a large sphere
and oriented to point the local z-directions of the small spheres at the center 
of the large one. Renders of the geometry are shaded using the sensor efficiency which 
is obtained using OSensorLib_sensor_efficiency using the local theta-phi positions 
of intersects and the sensor category.

Global frame intersect positions and intersected geometry identities are recorded 
into the posi.npy array. The posi.npy array is used by OSensorLibGeoTest.py 
together with the inverse of the saved transforms to obtain local frame intersect
positions which are checked to correspond to the known sphere radius.

This test was used to develop OSensorLib

**/

class OSensorLibGeoTest 
{
    private:
        OCtx*               m_octx ; 
        OSensorLib*         m_osenlib ;
        float               m_sphere_radius ; 
        float               m_assembly_radius ; 
        unsigned            m_num_theta ; 
        unsigned            m_num_phi ; 
        glm::vec4           m_ce_m ; 
        bool                m_identity_from_transform_03 ;
        NPY<float>*         m_transforms ; 
        unsigned            m_entry_point_index ; 
        const char*         m_closest_hit ; 
        const char*         m_raygen ; 
        unsigned            m_width ; 
        unsigned            m_height ; 
        NPY<unsigned char>* m_pixels ; 
        NPY<float>*         m_posi ;   
        const char*         m_dir ; 
        bool                m_transpose ; 
    public: 
        OSensorLibGeoTest(const SensorLib* senlib);
        void setViewpoint(); 
        void render(); 
        void save() const ; 
    private:
        void init(); 
        void initSensorEfficiency(); 
        void initGeometry(); 
        void initPipeline(); 
        void initView(); 
};

OSensorLibGeoTest::OSensorLibGeoTest(const SensorLib* senlib)
    :
    m_octx(OCtx::Get()),
    m_osenlib(new OSensorLib(m_octx, senlib)),
    m_sphere_radius(10.f),
    m_assembly_radius(500.f),
    m_num_theta(64+1),   // odd to plant one on equator
    m_num_phi(128),     
    m_ce_m( 0.f, 0.f, 0.f , m_assembly_radius ),
    m_identity_from_transform_03(true), // plant and use identity uint sneakily inserted into the top right element of the transform aka m4[0].w 
    m_transforms(SphereOfTransforms::Make(m_assembly_radius, m_num_theta, m_num_phi, m_identity_from_transform_03)),
    m_entry_point_index(0),
    m_closest_hit("closest_hit"),
    m_raygen("raygen"),
    m_width(1024),
    m_height(768),
    m_pixels(NPY<unsigned char>::make(m_height, m_width, 4)),
    m_posi(          NPY<float>::make(m_height, m_width, 4)),
    m_dir("$TMP/optixrap/tests/OSensorLibGeoTest"),
    m_transpose(true)
{
    init(); 
}


void OSensorLibGeoTest::init()
{
    initSensorEfficiency(); 
    initGeometry(); 
    initPipeline(); 
}

/**
OSensorLibGeoTest::initSensorEfficiency
----------------------------------------

Creates GPU textures for each sensor category and small texid buffer
with texture ID for each.

**/

void OSensorLibGeoTest::initSensorEfficiency()
{
    m_osenlib->convert();  
}


/**
OSensorLibGeoTest::initGeometry
--------------------------------

Creates an OptiX geometry using OCtx::create_instanced_assembly with the 
transforms array,

**/

void OSensorLibGeoTest::initGeometry()
{
    LOG(info) << "transforms " << m_transforms->getShapeString() ; 
    m_transforms->save(m_dir, "transforms.npy");

    LOG(info) << " MAIN_PTXPATH " << MAIN_PTXPATH ; 
    LOG(info) << " SPHERE_PTXPATH " << SPHERE_PTXPATH ; 

    unsigned num = 1 ; 
    void* sph_ptr = m_octx->create_geometry(num, SPHERE_PTXPATH, "bounds", "intersect" );  
    m_octx->set_geometry_float4( sph_ptr, "sphere",  0.f, 0.f, 0.f, m_sphere_radius );
    void* mat_ptr = m_octx->create_material( MAIN_PTXPATH,  m_closest_hit, m_entry_point_index );  

    void* sph_assembly_ptr = m_octx->create_instanced_assembly( m_transforms, sph_ptr, mat_ptr, m_identity_from_transform_03 );

    void* top_ptr = m_octx->create_group("top_object", NULL );  
    void* top_accel = m_octx->create_acceleration("Trbvh");
    m_octx->set_group_acceleration( top_ptr, top_accel );  

    m_octx->group_add_child_group( top_ptr, sph_assembly_ptr );  
}

/**
OSensorLibGeoTest::initPipeline
---------------------------------

Link up the pipeline and output buffers.

**/

void OSensorLibGeoTest::initPipeline() 
{
    m_octx->set_raygen_program( m_entry_point_index, MAIN_PTXPATH, m_raygen );
    m_octx->set_miss_program(   m_entry_point_index, MAIN_PTXPATH, "miss" );

    void* pixelsBuf = m_octx->create_buffer(m_pixels, "pixels_buffer", 'O', ' ', -1, m_transpose );
    void* posiBuf   = m_octx->create_buffer(m_posi,     "posi_buffer", 'O', ' ', -1,  m_transpose );

    m_octx->desc_buffer( pixelsBuf );
    m_octx->desc_buffer( posiBuf );

    m_pixels->zero();
    m_posi->zero();
}


/**
OSensorLibGeoTest::setViewpoint
--------------------------------

Convert whole assembly model frame eye-look-up into global frame 
and set into context.

**/

void OSensorLibGeoTest::setViewpoint() 
{
    float scene_epsilon = 1.f ;  // when cut into earth, see circle of back to front geography from inside the sphere
    float tanYfov = 1.0f ;       // 0.5:zoom-in, 1.0:overview, 2.0:extreme-overview-distorting-edges
    const bool dump = false ;

    // eye point, look point, up point : in unit model frame
    //glm::vec3  eye_m(  -0.1, -0.1f,  0.1f );   
    //glm::vec3 look_m(  0.7f,  0.7f, -0.7f ); 

    glm::vec3  eye_m(  0.6,   0.0f,  0.0f );   
    glm::vec3 look_m(  1.0f,  0.0f,  0.0f ); 

    glm::vec3   up_m(   0.f,   0.f,  1.f  );  

    glm::vec3 eye,U,V,W ;
    nglmext::GetEyeUVW( m_ce_m, eye_m, look_m, up_m, m_width, m_height, tanYfov, eye, U, V, W, dump );

    m_octx->set_context_viewpoint( eye, U, V, W, scene_epsilon );
}

/**
OSensorLibGeoTest::render
--------------------------

Do the GPU launch and download output buffers

**/

void  OSensorLibGeoTest::render()
{
    double t_prelaunch ;
    double t_launch ;

    assert( m_transpose == true ); 
    // transpose:true is necessary for binary compatibility between row-major NPY arrays and column-major OptiX buffers
    unsigned l0 = m_transpose ? m_width  : m_height ;
    unsigned l1 = m_transpose ? m_height : m_width  ;

    m_octx->launch_instrumented( m_entry_point_index, l0, l1, t_prelaunch, t_launch );

    LOG(info) 
        << " t_prelaunch " << t_prelaunch
        << " t_launch " << t_launch
        ;

    m_octx->download_buffer(m_pixels, "pixels_buffer", -1);
    m_octx->download_buffer(m_posi, "posi_buffer", -1);
}

void OSensorLibGeoTest::save() const 
{
    LOG(info) << m_dir ; 
    const bool yflip = true ;
    ImageNPY::SavePPM(m_dir, "pixels.ppm", m_pixels, yflip );

    m_pixels->save(m_dir, "pixels.npy");
    m_posi->save(m_dir, "posi.npy");
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
    //senlib->dump("OSensorLibGeoTest"); 

    OSensorLibGeoTest slt(senlib); 

    slt.setViewpoint(); 
    slt.render(); 
    slt.save(); 

    LOG(info) << "open $TMP/optixrap/tests/OSensorLibGeoTest/pixels.ppm  " ;
    LOG(info) << "ipython -i ~/opticks/optixrap/tests/OSensorLibGeoTest.py " ; 

    return 0 ; 
}
// om-;TEST=OSensorLibGeoTest om-t 
