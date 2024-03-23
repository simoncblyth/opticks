optix7sdk-vi(){ vi $BASH_SOURCE ; }
optix7sdk-env(){ echo -n ; }
optix7sdk-usage(){ cat << EOU 
optix7sdk review
=================


sutil::loadScene
    uses tinygltf to load model data and uploads 

    Q: Where is triangle vertices/indices ? buffers/meshes ?

    The WaterBottle.gltf has one buffer of 149412 bytes with 5 bufferViews
    of byte lengths : 20392 + 30588 + 40784 + 30588 + 27060 

    2549*2*4 = 20392   2549:VEC2:FLOAT             TEXCOORD_0
    2549*3*4 = 30588   2549:VEC3:FLOAT             NORMAL 
    2549*4*4 = 40784   2549:VEC4:FLOAT             TANGENT (apparently tangents often float4, why?)
    2549*3*4 = 30588   2549:VEC3:FLOAT             POSITION

    13530*2  = 27060   13530:SCALAR:UNSIGNED_SHORT INDICES?
    13530 % 3 == 0, 13530//3 = 4510 (number of triangles)

* UNSIGNED_SHORT INDICES IMPLY ONLY UP TO 0xffff//3 = 21845 ~21k triangles


sutil::Scene::addBuffer
    uploads data and appends CUdeviceptr into m_buffers



Grokking mesh handling
-------------------------

sutil/Scene.cpp::

     466             mesh->indices.push_back( bufferViewFromGLTF<uint32_t>( model, scene, gltf_primitive.indices ) );
     467             mesh->material_idx.push_back( gltf_primitive.material );
     468             std::cerr << "\t\tNum triangles: " << mesh->indices.back().count / 3 << std::endl;

* buffer_view.data is the CUdeviceptr 


Examples
----------

optixTriangle
   single tri only, no instancing
   optixTriangle.h : empty HitGroup data

optixBoundValues
optixCallablePrograms
optixCompileWithTasks
optixCurves
optixCustomPrimitive
optixCutouts
optixDemandLoadSimple
optixDemandTexture
optixDenoiser
optixDynamicGeometry
optixDynamicMaterials
optixHair
optixHello
optixMeshViewer
optixModuleCreateAbort
optixMotionGeometry
optixMultiGPU
optixNVLink
optixOpticalFlow
optixPathTracer
optixRaycasting
optixSimpleMotionBlur
optixSphere
optixVolumeViewer
optixWhitted





EOU
}

