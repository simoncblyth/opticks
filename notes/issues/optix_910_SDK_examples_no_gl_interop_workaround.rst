optix_910_SDK_examples_no_gl_interop_workaround
===============================================


Build the SDK examples with::

    cd /usr/local/optix/OptiX_910/SDK
    cat INSTALL-LINUX.txt

Follow the INSTALL-LINUX.txt instructions, briefly::

    mkdir build
    cd build
    ccmake ..    ##  some keypresses and typing "Debug"  needed as instructed

    make
    cd bin

    ./optixWhitted
    ./optixWhitted --no-gl-interop


Many but not all of the examples exit with error::

    Caught exception: GL interop is only available on display device, please
    use display device for optimal performance.  Alternatively you can disable GL
    interop with --no-gl-interop and run with degraded performance.

Using that option succeeds to get the examples to work.



Googling that error string yields forum post from dhart (NVIDIA OptiX expert)::

* https://forums.developer.nvidia.com/t/nvidia-optix-8-0-0-ray-trace-gl-interop-exception/324483

Which essentially says that disabling gl interop is only relevant to performance of
interactive apps, not to compute performance::

    Hi @ehteshamimaad,

    Have you tried using the Nvidia Control Panel to set your A3000 to be the
    default GPU for optixPathTracer? You can set the preferred GPU either globally
    or on a per-application basis.

    Does the sample run fine when using --no-gl-interop? You just want to make sure
    you’re getting optimal performance, and/or see what the performance difference
    is?

    To clarify, the “degraded performance” that the sample is referring to does not
    mean that ray tracing will slow down. What it means is that the application has
    to copy the framebuffer from your Nvidia GPU to the display GPU every frame.
    This is a relatively fast operation and may or may not be noticeable. The
    framebuffer copy will not prevent being able to achieve 60fps, and chances are
    it will not matter at all if your render kernel takes longer than what is
    required for 60fps, meaning it takes longer than 16 milliseconds (if you use a
    high samples-per-pixel setting, for example). Moreover, this only applies to
    interactive applications, and does not apply when writing your ray tracing
    results to an image file, since the framebuffer copy to the host machine is
    unavoidable.

    I’m only explaining in case the ‘degraded performance’ warning was accidentally
    scaring you more than it should. Internally we’ll generally just use
    --no-gl-interop fairly often as needed, and there aren’t many cases where it
    matters. If we’re doing careful benchmarking on a display GPU, for example,
    then yeah we’ll try to make sure GL interop is working. But actually more often
    than not, I prefer benchmarking on a non-display GPU because it’s faster and
    the timings are more stable - because the display GPU is always doing other
    things, display stuff, and not just the ray tracing or compute for your app.

    –
    David.


Despite this, the examples in earlier versions of OptiX and CUDA did not have this issue.
Also not all examples need that option to work.


The examples that did not require "--no-gl-interop" can be extracted from the below history::

     1016  ./optixWhitted
     1017  echo $CUDA_VISIBLE_DEVICES
     1018  ./optixWhitted --no-gl-interop
     1019  l
     1020  ./optixVolumeViewer
     1021  ./optixVolumeViewer --no-gl-interop
     1022  l
     1023  ./optixTriangle
     1024  l
     1025  ./optixSphere
     1026  l
     1027  ./optixSimpleMotionBlur
     1028  ./optixSimpleMotionBlur --no-gl-interop
     1029  l
     1030  ./optixRibbons
     1031  l
     1032  ./optixRaycasting
     1033  open output.ppm
     1034  l
     1035  ./optixPathTracer
     1036  ./optixPathTracer --no-gl-interop
     1037  l
     1038  ./optixOpacityMicromap
     1039  l
     1040  ./optixOpticalFlow
     1041  ./optixNeuralTexture
     1042  ./optixNeuralTexture --no-gl-interop
     1043  l
     1044  ./optixMultiGPU
     1045  l
     1046  ./optixMotionGeometry
     1047  ./optixMotionGeometry  --no-gl-interop
     1048  l
     1049  ./optixModuleCreateAbort
     1050  ./optixModuleCreateAbort --no-gl-interop
     1051  l
     1052  ./optixMeshViewer
     1053  ./optixMeshViewer --no-gl-interop
     1054  l
     1055  ./optixHair
     1056  l
     1057  ./optixDynamicMaterials
     1058  ./optixDynamicMaterials --no-gl-interop
     1059  l
     1060  ./optixDynamicGeometry
     1061  nvidia-smi
     1062  history
     1063  ./optixDynamicGeometry
     1064  pwd
     1065  ./optixDynamicGeometry --no-gl-interop
     1066  pwd
     1067  l
     1068  ./optixDenoiser
     1069  #./optixDenoiser ## requires inputs
     1070  ./optixCustomPrimitive
     1071  l
     1072  ./optixCustomCache
     1073  l
     1074  ./optixCurves
     1075  l
     1076  ./optixConsole
     1077  l
     1078  ./optixCompileWithTasks
     1079  ./optixCompileCancel
     1080  ./optixCompileCancel --no-gl-interop
     1081  ./optixClusterUnstructuredMesh
     1082  ./optixClusterUnstructuredMesh --no-gl-interop
     1083  l
     1084  ./optixClusterStructuredMesh
     1085  ./optixClusterStructuredMesh --no-gl-interop
     1086  l
     1087  ./optixCallablePrograms
     1088  ./optixCallablePrograms  --no-gl-interop
     1089  l
     1090  ./optixBoundValues
     1091  ./optixBoundValues --no-gl-interop
     1092  l
     1093  history
    A[blyth@localhost bin]$ 





