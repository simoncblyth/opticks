/* 
   Intersection of Assimp and OptiX

   Some aspects inspired by 
       /usr/local/env/cuda/optix/OppositeRenderer/OppositeRenderer/RenderEngine/scene/Scene.h 
*/

#ifndef OPTIXSCENE_H
#define OPTIXSCENE_H

#include <GLUTDisplay.h>
#include <sutil.h>
#include <MeshScene.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>

class OptiXProgram ; 
class OptiXAssimpGeometry ; 


//class OptiXScene  : public SampleScene 
class OptiXScene  : public MeshScene 
{
public:
    OptiXScene();
    virtual ~OptiXScene();

    optix::Context getContext();

    void setProgram(OptiXProgram* program);

    void setGeometry(OptiXAssimpGeometry* geometry);

public:
   // From SampleScene
   void   initScene( InitialCameraData& camera_data );

   void   trace( const RayGenCameraData& camera_data );

   void   doResize( unsigned int width, unsigned int height );

   void   setDimensions( const unsigned int w, const unsigned int h );

   optix::Buffer getOutputBuffer();


private:

   unsigned int m_width  ;

   unsigned int m_height ;

   OptiXProgram* m_program ;

   OptiXAssimpGeometry* m_geometry ;

};


#endif

