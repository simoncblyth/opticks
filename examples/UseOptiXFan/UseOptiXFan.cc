#include <optix_world.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>

using namespace optix;

int SP_NUM_X = 20;
int SP_NUM_Y = 20;
int SP_NUM_Z = 20;
float SP_RAD = 15;
float SP_SEP = 50;

const int NUM_DOM = 8000;


struct GeoConfig 
{
    typedef std::array<float,16> Transform_t ; 
    std::vector<Transform_t>  transforms ; 
    static Transform_t MakeTranslation(float tx, float ty, float tz) 
    {
       return  { 1.f, 0.f, 0.f, 0.f,    
                 0.f, 1.f, 0.f, 0.f,     
                 0.f, 0.f, 1.f, 0.f,    
                 tx , ty,  tz,  1.f}  ; 
    }    

    GeoConfig(unsigned num)
    {
        for (int i = 0; i < num; i++)
        {
            int nx = static_cast<int>(i / 400);
            int ny = static_cast<int>((i - 400 * nx) / 20);
            int nz = i - 400 * nx - 20 * ny;

            float tx = nx * 50 - 475;
            float ty = ny * 50 - 475;
            float tz = nz * 50 - 475;
             
            if( i == 0 )
            {
                transforms.push_back(MakeTranslation(0.5,0.5,0.5));  
            } 
            else
            {
                transforms.push_back(MakeTranslation(tx,ty,tz));  
            }

        }
    }
};








//int NUM_PHOTON = 1000;
int NUM_PHOTON = 100;

const char* PREFIX = getenv("PREFIX"); 
const char* CMAKE_TARGET = "UseOptiXFan" ; 


// Error handling extracted from SDK/sutil/sutil.{h,cpp}
struct APIError
{   
    APIError( RTresult c, const std::string& f, int l ) 
        : code( c ), file( f ), line( l ) {}
    RTresult     code;
    std::string  file;
    int          line;
};

#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw APIError( code, __FILE__, __LINE__ );           \
  } while(0)


void reportErrorMessage( const char* message )
{
    std::cerr << "OptiX Error: '" << message << "'\n";
}

void handleError( RTcontext context, RTresult code, const char* file, int line)
{
    const char* message;
    char s[2048];
    rtContextGetErrorString(context, code, &message);
    sprintf(s, "%s\n(%s:%d)", message, file, line);
    reportErrorMessage( s );
}

#define SUTIL_CATCH( ctx ) catch( APIError& e ) {           \
    handleError( ctx, e.code, e.file.c_str(), e.line );     \
  }                                                         \
  catch( std::exception& e ) {                              \
    reportErrorMessage( e.what() );                         \
    exit(1);                                                \
  }



const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" )
{
    std::stringstream ss ; 
    ss << install_prefix
       << "/ptx/"
       << cmake_target
       << "_generated_"
       << cu_stem
       << cu_ext
       << ".ptx" 
       ;   
    std::string path = ss.str();
    return strdup(path.c_str()); 
}

void createContext(RTcontext *context)
{
    RT_CHECK_ERROR(rtContextCreate(context));
    RT_CHECK_ERROR(rtContextSetRayTypeCount(*context, 1));
    RT_CHECK_ERROR(rtContextSetEntryPointCount(*context, 1));
    RT_CHECK_ERROR(rtContextSetPrintEnabled( *context, 1 ) );
    RT_CHECK_ERROR(rtContextSetPrintBufferSize( *context, 4096 ) );
}

void createSource(RTcontext *context)
{
    const char* ptx = PTXPath( PREFIX, CMAKE_TARGET, "point_source" ) ; 
    std::cout << " createSource: ptx " << ptx << std::endl ;  

    RTprogram ray_gen_program;
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(*context, ptx, "point_source", &ray_gen_program));
    RT_CHECK_ERROR(rtContextSetRayGenerationProgram(*context, 0, ray_gen_program));

    RTprogram exception_program;
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(*context, ptx, "exception", &exception_program));
    RT_CHECK_ERROR(rtContextSetExceptionProgram(*context, 0, exception_program));

    RTprogram miss_program;
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(*context, ptx, "miss", &miss_program));
    RT_CHECK_ERROR(rtContextSetMissProgram(*context, 0, miss_program));


    RTvariable source_pos;
    RT_CHECK_ERROR(rtContextDeclareVariable(*context, "source_pos", &source_pos));
    RT_CHECK_ERROR(rtVariableSet3f(source_pos, 0.0f, 0.0f, 0.0f));
}




void createHitBuffer(RTcontext* context, RTbuffer *output_id)
{
    rtBufferCreate(*context, RT_BUFFER_OUTPUT, output_id);
    rtBufferSetFormat(*output_id, RT_FORMAT_UNSIGNED_INT);
    rtBufferSetSize1D(*output_id, NUM_PHOTON);
    RTvariable output_id_var;
    rtContextDeclareVariable(*context, "output_id", &output_id_var);
    rtVariableSetObject(output_id_var, *output_id);
}

void createMaterial(RTcontext* context, RTmaterial *material)
{
    const char* ptx = PTXPath( PREFIX, CMAKE_TARGET, "simple_dom" ) ; 
    std::cout << " createMaterial " << ptx << std::endl ; 

    RTprogram closest_hit ;
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(*context, ptx, "closest_hit", &closest_hit));

    RT_CHECK_ERROR(rtMaterialCreate(*context, material));
    RT_CHECK_ERROR(rtMaterialSetClosestHitProgram(*material, 0, closest_hit));
}

void createSphere(RTcontext* context, RTgeometry *sphere)
{
    const char *ptx = PTXPath( PREFIX, CMAKE_TARGET, "sphere");

    RT_CHECK_ERROR(rtGeometryCreate(*context, sphere));
    RT_CHECK_ERROR(rtGeometrySetPrimitiveCount(*sphere, 1u));

    RTprogram bounds ;
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(*context, ptx, "bounds", &bounds));
    RT_CHECK_ERROR(rtGeometrySetBoundingBoxProgram(*sphere, bounds));

    RTprogram intersect ;;
    RT_CHECK_ERROR(rtProgramCreateFromPTXFile(*context, ptx, "intersect", &intersect));
    RT_CHECK_ERROR(rtGeometrySetIntersectionProgram(*sphere, intersect));

    RTvariable sphere_param;
    RT_CHECK_ERROR(rtGeometryDeclareVariable(*sphere, "sphere_param", &sphere_param));
    float sphere_loc[4] = {0.f, 0.f, 0.f, 15.f};
    RT_CHECK_ERROR(rtVariableSet4fv(sphere_param, sphere_loc));
}



/*    
createGeometry
--------------

top                      (Group)
    assembly             (Group) 
       xform             (Transform)
           perxform      (GeometryGroup)
              pergi      (GeometryInstance)  
              accel      (Acceleration)   
*/  

void createGeometry(RTcontext* context, const GeoConfig& cfg)
{
  RTgeometry sphere;
  createSphere(context, &sphere); 

  RTmaterial material;
  createMaterial(context, &material);

  RTacceleration accel ;
  RT_CHECK_ERROR(rtAccelerationCreate(*context, &accel));
  RT_CHECK_ERROR(rtAccelerationSetBuilder(accel, "Trbvh"));

  unsigned num_instances = cfg.transforms.size() ;


  RTacceleration assembly_accel ;
  RT_CHECK_ERROR(rtAccelerationCreate(*context, &assembly_accel));
  RT_CHECK_ERROR(rtAccelerationSetBuilder(assembly_accel, "Trbvh"));

  RTgroup assembly ;
  RT_CHECK_ERROR(rtGroupCreate(*context, &assembly));
  RT_CHECK_ERROR(rtGroupSetChildCount(assembly, num_instances ));
  RT_CHECK_ERROR(rtGroupSetAcceleration(assembly, assembly_accel));

  for (int instance_idx = 0; instance_idx < num_instances; instance_idx++)
  {
      RTtransform xform ; 
      RT_CHECK_ERROR(rtTransformCreate(*context, &xform));
      int transpose = 1 ; 
      const std::array<float,16>& arr = cfg.transforms[instance_idx] ; 
      const float* matrix = arr.data() ; 
      const float* inverse = NULL ;  
      RT_CHECK_ERROR(rtTransformSetMatrix(xform, transpose, matrix, inverse )); 

      RTgeometryinstance pergi ;
      RT_CHECK_ERROR(rtGeometryInstanceCreate(*context, &pergi));
      RT_CHECK_ERROR(rtGeometryInstanceSetGeometry(pergi, sphere));
      RT_CHECK_ERROR(rtGeometryInstanceSetMaterialCount(pergi, 1u));
      RT_CHECK_ERROR(rtGeometryInstanceSetMaterial(pergi, 0, material));

      RTgeometrygroup perxform;
      RT_CHECK_ERROR(rtGeometryGroupCreate(*context, &perxform));
      RT_CHECK_ERROR(rtGeometryGroupSetChildCount(perxform, 1));
      RT_CHECK_ERROR(rtGeometryGroupSetChild(perxform, 0, pergi));
      RT_CHECK_ERROR(rtGeometryGroupSetAcceleration(perxform, accel));

      RT_CHECK_ERROR(rtTransformSetChild(xform, perxform));

      RT_CHECK_ERROR(rtGroupSetChild(assembly, instance_idx, xform));
  }

  RTacceleration top_accel ;
  RT_CHECK_ERROR(rtAccelerationCreate(*context, &top_accel));
  RT_CHECK_ERROR(rtAccelerationSetBuilder(top_accel, "Trbvh"));

  RTgroup top ;
  RT_CHECK_ERROR(rtGroupCreate(*context, &top));
  RT_CHECK_ERROR(rtGroupSetChildCount(top, 1));
  RT_CHECK_ERROR(rtGroupSetChild(top, 0, assembly));
  RT_CHECK_ERROR(rtGroupSetAcceleration(top, top_accel));

  RTvariable top_object;
  RT_CHECK_ERROR(rtContextDeclareVariable(*context, "top_object", &top_object));
  RT_CHECK_ERROR(rtVariableSetObject(top_object, top));
}



int main(int argc, char *argv[])
{
  assert( PREFIX && "must define PREFIX envvar " );  

  GeoConfig cfg(NUM_DOM);  
  RTcontext context = 0;
  try
  {
    createContext(&context);
    createSource(&context);

    RTbuffer output_id ;
    createHitBuffer(&context, &output_id);

    createGeometry(&context, cfg);

    std::cout << "[ context validation " << __FILE__ << ":" << __LINE__ << std::endl ; 
    RT_CHECK_ERROR(rtContextValidate(context));
    std::cout << "] context validation " << __FILE__ << ":" << __LINE__ << std::endl ; 

    unsigned entry_point_index = 0u ; 
    RTsize width = NUM_PHOTON ; 
    RT_CHECK_ERROR(rtContextLaunch1D(context, entry_point_index, width));

    // read output buffer
    void *output_id_ptr;
    RT_CHECK_ERROR(rtBufferMap(output_id, &output_id_ptr));
    unsigned int *output_id_data = (unsigned int *)output_id_ptr;
    for (int i = 0; i < NUM_PHOTON; i++)
    {
        std::cout << output_id_data[i] <<  " " ; 
        if(i % 50 == 0 ) std::cout << std::endl;
    }
    std::cout << std::endl ; 

    RT_CHECK_ERROR(rtBufferUnmap(output_id));

    // clean up
    RT_CHECK_ERROR(rtContextDestroy(context));
    return (0);
  }
  SUTIL_CATCH(context)
}

