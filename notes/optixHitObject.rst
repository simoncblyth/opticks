optixHitObject
===============


OptiX_Programming_Guide_9.1.0.pdf/8.2.3 Hit objects
-----------------------------------------------------

OptiX does not have an explicit object representation of the data associated with a hit during
ray tracing. Instead, functions such as optixGetRayTmin and optixGetInstanceId return
values corresponding to the current hit. Collectively, this data is called the hit object. The hit
object is available during traversal and shading. It represents either the currently intersected
object or the closest intersected object, if any. Several of the hit object values are initialized at
the beginning of traversal, while others are set when a hit is accepted by the any hit program.


The hit object’s role in the ray-tracing process can be divided into four major phases:

1. A call to optixTrace — The current hit object is saved if there is one. A new hit object is
   created

2. Traversal — The hit object is modified and set based on what is intersected and the results
   of any hit calls.

3. Shading — The hit object is available to the closest-hit or miss programs to read.

4. The return to the caller — If the previous hit object was saved, it is now restored.


Because there is no explicit representation of the hit object, there can only ever be one hit
object active at a time. If a secondary ray is traced during shading, the current hit object will
be saved so that it may be restored later.
With the introduction of optixTraverse and optixInvoke , an application must be able to
interact with the hit object that is produced by traversal and consumed by shading. For the
purposes of this chapter, the hit object available during traversal and shading will be referred
to as the incoming hit object while the one available outside of traversal and shading will be
referred to as the outgoing hit object. Both the incoming and outgoing hit objects can be live at
the same time, but there is only ever one of each at a time. Accessor functions for the
outgoing hit object are distinguished from the incoming hit object with the inclusion of the
string “HitObject” in the device side API functions. For example, optixGetRayTmin accesses
the incoming hit object, while optixHitObjectGetRayTmin accesses the outgoing hit object.
There are three types of hit objects:

* hit
* miss
* nop (“no operation” — neither a hit nor miss will be executed if optixInvoke is called)

You can query the type of hit object through the following functions:

• optixHitObjectIsHit
• optixHitObjectIsMiss
• optixHitObjectIsNop

There are several functions that can set the outgoing hit object:

+--------------------------+-----------------+-----------------+-----------------+
| Function                 | Produces hit    |  Produces miss  |  Produces nop   |
|                          | (outgoing)      |  (outgoing)     |  (outgoing)     |
+==========================+=================+=================+=================+
| optixTraverse            |   YES           |      YES        |                 |
+--------------------------+-----------------+-----------------+-----------------+
| optixMakeHitObject       |   YES           |                 |                 |
+--------------------------+-----------------+-----------------+-----------------+
| optixMakeMissHitObject   |                 |     YES         |                 |
+--------------------------+-----------------+-----------------+-----------------+
| optixMakeNopHitObject    |                 |                 |    YES          |
+--------------------------+-----------------+-----------------+-----------------+
| optixInvoke              |                 |                 |    YES          |
+--------------------------+-----------------+-----------------+-----------------+
| optixTrace               |                 |                 |    YES          |
+--------------------------+-----------------+-----------------+-----------------+


Since there can be only one outgoing hit object at a time, any function that sets it will
overwrite the existing hit object.

Shading through the invocation of the outgoing hit object (i.e. calling optixInvoke )
“consumes” the outgoing hit object and the outgoing hit object is set to nop. You can consider
optixTrace as effectively a pair of calls to optixTraverse and optixInvoke , which
overwrites the outgoing hit object and subsequently sets it to nop after the invocation of
shading.

Note that after certain calls such as optixTraverse , the hit object can only be of type hit or
miss, so shaders can be written with a single comparison to determine if there is a hit.
Listing 8.2 shows such a comparison::

    optixTraverse( ... );
    if( optixHitObjectIsHit() )
        printf( "Ray hit" );
    else
        printf( "Ray missed" );


For a given function’s scope, there is always a single outgoing hit object which is initialized to
a nop hit object. The function’s scope also extends to any inlined functions. In other words, if
a function is inlined it does not maintain its own scope but rather becomes part of the caller’s
scope.

A nop hit object has no data associated with it and queries against it will return zero.

Note that the following hit object accessors will return undefined values when the hit object is
a miss:

• optixHitObjectGetInstanceId
• optixHitObjectGetInstanceIndex
• optixHitObjectGetPrimitiveIndex
• optixHitObjectGetTransformListHandle
• optixHitObjectGetSbtGASIndex
• optixHitObjectGetHitKind
• optixHitObjectGetAttribute_0 ... 7

After an outgoing hit object is set, the values in the hit object will remain unchanged until a
subsequent call creates a new outgoing hit object.

Setting the outgoing hit object will not affect the incoming hit object. For example, calling
optixTraverse inside a closest-hit shader will not affect the results of the incoming hit object
associated with that closest-hit shader. Note that in the regions of a function where the
incoming hit object is not used, OptiX does not need to save or restore the components of that
hit object, with an improvement in performance as a result.

The role of incoming and outgoing hit objects in function optixTraverse and their context in
the closest-hit program can be summarized as follows.

__ch__
{
     // The incoming hit object is set by trace/invoke and is accessible throughout the function.
     float tmax = optixGetRayTmax(); Returns the incoming hit object tmax.

     // The outgoing hit object is set to NOP at the start of the function.
     // Any access to a NOP-value outgoing hit object returns 0.
     assert( optixHitObjectGetRayTmax() == 0 );
     optixTrace()
     {
          // Actions during the traversal phase:
          * Save the incoming hit object.
          * Set the incoming hit object.
          * Traverse.

          // Actions during the invocation phase:
          * Invoke shading (closest-hit and miss programs).
          * Set the outgoing hit object to NOP.
          * Restore the incoming hit object.
      }


      // The incoming hit object is preserved across calls to optixTrace.
      assert( optixGetRayTmax() == tmaxIncoming );

      // The outgoing hit object is still NOP.
      assert( optixHitObjectGetRayTmax() == 0 );
}


When the traversal and shader invocation phases are performed separately with
optixTraverse and optixInvoke, the hit object plays the same role as in optixTrace :


__ch__
{
     // The incoming hit object is set by trace/invoke and is accessible throughout the function.
     float tmax = optixGetRayTmax(); Returns the incoming hit object tmax.

     // The outgoing hit object is set to NOP at the start of the function.
     // Any access to an outgoing NOP-value hit object returns 0.
     assert( optixHitObjectGetRayTmax() == 0 );

     optixTraverse() {
         // Actions taken by the function:
         * Save the incoming hit object.
         * Set the incoming hit object.
         * Traverse.
         * Set the outgoing hit object to the results of the traversal.
         * Restore the incoming hit object.
     }

     // The incoming hit object is preserved across calls to optixTraverse.
     assert( optixGetRayTmax() == tmaxIncoming );

     // You can access the outgoing hit object set by optixTraverse.
     float tmaxOutgoing = optixHitObjectGetRayTmax();

     optixInvoke {
          // Actions taken by the function:
          * Save the incoming hit object.
          * Set the incoming hit object from the outgoing hit object.
          * Invoke shading (closest-hit and miss programs).
          * Set the outgoing hit object to NOP.
          * Restore the incoming hit object.
     }

     // The incoming hit object is preserved across calls to optixInvoke.
     assert( optixGetRayTmax() == tmaxIncoming );
     // The outgoing hit object is set to NOP by calling optixInvoke.
     assert( optixHitObjectGetRayTmax() == 0 );
}



8.2.4 Coherence hints
----------------------


Using the pattern described in the previous section as a basis, an application can further
inform reordering with coherence hints. Coherence hints are represented as an integer from
which some number of bits are incorporated in the reordering key that the system computes
from a hit object. When numCoherenceHintBitsFromLSB is zero, no user hint bits are used
and coherenceHint is ignored.

Here is the full optixReorder signature::

    void optixReorder(
        unsigned int coherenceHint,
        unsigned int numCoherenceHintBitsFromLSB );

In optixReorder, NumCoherenceBitsFromLSB specifies the number of bits that
optixReorder should take into account, starting from the least significant bit.

There is a convenience version of optixReorder where the coherenceHint does not need to be
specified. It is equivalent to calling optixReorder with numCoherenceHintBitsFromLSB and
coherenceHint set to zero.::

     void optixReorder();



The final key used for reordering is then composed from the following components, in
descending order of priority:

1. Shader ID stored in the hit object
2. Coherence hint, with the most significant hint bit having highest priority
3. Spatial information stored in the hit object

With this key design, an application can use coherence hints to incorporate knowledge about
execution divergence into the reordering that isn’t already represented in the hit object. One
example of this is branches within hit shader code, which are often based on material
parameters — whether the material is emissive, has a clear coat layer, casts a shadow ray, etc.
An application may inspect these parameters and include them in the coherence hint before
reordering and executing the shader. One way to store material parameter information is as
root constants in the shader table. A hit object provides a convenient method to read data
from the shader table entry that the hit object represents.

Here is an example pattern that uses the hit object and optixReorder::

     optixTraverse( sceneIAS, ..., payload );

     unsigned int materialFlags = *reinterpret_cast<unsigned int*>(optixHitObjectGetSbtDataPointer() );
                             // Inspect the material flags encoded as a root constant
                             // at offset 0 in the Shader table entry of our hit.
                             // Note this requires all closest-hit and miss programs
                             // to encode this integer.

     optixReorder( materialFlags, 4 );
                             // Reorder by hit point, using material flags as
                             // additional coherence hints. Assumes we have
                             // four bits worth of flags.


     optixInvoke( payload );
                            // Invoke shading. Thanks to reordering with coherence hints
                            // this will be coherent both with respect to which shader is
                            // executed as well as which path is taken through the shaders.




Coherence hints can also be used to reorder threads based on control flow in the
ray-generation shader itself. In the simple loop depicted in Listing 8.6, threads in a warp can
be expected to stay on the same iteration without optixReorder (for example, first i=0, then
i=1). With optixReorder, after executing the call, the value of increment variable i can be
different among threads. This depends on the result of the reordering, which in turn depends
on the contents of the hit object. If this leads to divergence, it is suitable to add a coherence
hint based on i. It is often beneficial to make sure that the entire warp finishes the loop at the
same time.::

    // Listing 8.6
    for( unsigned int i = 0; i < 4; ++i ) {
        ...
        unsigned int coherenceHints = i == 3 ? 1 : 0;
        optixReorder( coherenceHints, 1 );
        ...
    }


A related situation can be found in path tracers or multi-bounce reflections. In those
processes, it is often highly effective to include in the coherence hint some additional
information about whether the main loop will terminate. On top of executing hit shaders
coherently, this has the effect of compacting threads into warps for the next iteration of the
loop. Hence, the number of inactive lanes from threads that exit the loop early is reduced.
This benefits both the execution of the loop itself, as well as the utilization of the RT Core
hardware, which operates more effectively the more threads per warp are active at the time
optixTrace is called.

Listing 8.7 is a pseudocode example of shader reordering based on coherence hints::

     // Listing 8.7
     while( ... ) {    // Loop over light bounces
       optixTraverse( sceneIAS, ..., payload );
       float albedo = *reinterpret_cast<float*>(optixHitObjectGetSbtDataPointer() );
       bool done = russianRoulette( payload, albedo ) || bounceCount >= maxBounces;
       unsigned int coherenceHints = done ? 1 : 0;
       optixReorder( coherenceHints, 1 );
       optixInvoke( payload );
       if( done ) break ;
              // Because we included the "done" flag in the coherence hints, chances are
              // good that all threads in the warp will make the same decision about
              // whether or not to break out of the loop.
    }


Sometimes, predicting the exact control flow is impossible at the point where a coherence hint
is computed. In such situations, it is important to remember that it may be sufficient to have a
confident estimate rather than an exact value for the coherence hint. A confident estimate is
sufficient because reordering only affects performance, not correctness, so that an
approximate coherence hint is often better than none at all.


8.2.5 More ways to use the hit object
---------------------------------------

The hit object is the mechanism that enables the splitting of optixTrace into two phases and
by extension the combination of application-side and system-side knowledge to inform
reordering. It is worth noting, however, that hit object is a versatile tool in and of itself even
without support for reordering. It allows an application to implement concepts that were
difficult or impossible to achieve with the traditional raytracing programming model. Some
examples that have proven useful in practice include the following.

Not executing closest-hit shading after casting a ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some situations, it is useful to cast a ray without triggering shading, but while still
obtaining basic information about the hit. Shadow or ambient occlusion rays are common
examples, where material shading typically isn’t needed, but knowing the closest
intersection distance can be useful for filtering. The existing API provides the
OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT ray flag, but using it offers no way to obtain
information about the closest hit. The typical solution is to write a trivial closest-hit shader
that only fills the payload with the desired hit information. A hit object provides a simpler
and more efficient path: one can simply trace the ray using optixTraverse and then use
state functions to inspect the resulting hit object without ever calling optixInvoke . See
“The hit object’s state” (page 111).

::

   // Listing 8.8
   optixTraverse( sceneIAS, ..., payload );
   bool hitDistance = MAX_FLT;
   if( optixHitObjectIsHit() )
        hitDistance = optixHitObjectGetRayTmax();


Custom indexing into the shader table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shader table record that optixTrace invokes is found using a formula that takes into
account a number of factors from various sources, described in “Acceleration structures”
(page 89). The current formula provides some amount of flexibility, but it is fixed and
imposes certain constraints. A hit object may be modified to reference an arbitrary shader
record with optixHitObjectSetSbtRecordIndex . This opens new possibilities when it
comes to shader table organization. Altering the SBT index can be interesting particularly
in applications that only use a single combined any-hit and intersection shader across all
objects in the scene. In such cases, all instances can refer to the same shader table range
used only to trigger any-hit and intersection shaders, while the closest-hit or miss shader is
selected using a manually computed index.


8.3 Best practices
--------------------

8.3.1 When to use (and when not to use) reordering
---------------------------------------------------

The implementation of optixReorder is highly optimized but its execution is not without
cost. Because the benefits of reordering depend on the work that is executed after reordering,
there are situations where reordering is not advisable.

For example, hit shaders for shadow or ambient occlusion rays are typically trivial, which
means that spending additional cycles to extract coherence for such shaders is usually not
worth the cost. The same is true for cases where rays are very coherent to begin with, like
primary rays.

Furthermore, work following optixReorder may not actually benefit from hit coherence.
Consider that reordering threads with respect to their ray hit location means giving up on the
2D screen-space locality that physical threads have by default. This can reduce the
performance of reading or writing data indexed by the launch index, which can turn
reordering into a net loss when the rest of the workload is particularly cheap.

Reordering is most beneficial in situations that exhibit non-trivial hit shading (irrespective of
whether the shading code lives in the ray-generation or closest-hit shader), paired with at
least moderate divergence in the ray distribution. Reflection regimes are a typical example,
especially when the reflections are glossy or diffuse. In general, the more numerous and the
more complex the shaders, and the more object-space or world-space data they access
(textures, vertex attributes, environment probes, etc.), the more potential there is for
reordering to improve performance. Note, however, that a high shader count isn’t necessarily
required. Using coherence hints to decrease data divergence can yield performance gains
even in applications that use only a single shader across the entire scene. Scenarios like
multi-bounce reflections or full-fledged path tracers increase the potential even further,
because additional execution divergence comes from the main loop itself. As indicated in the
examples earlier, incorporating information about the loop state into coherence hints can
yield additional performance improvements.

Furthermore, when multiple rays are traced in a loop, it can be important to reorder in a way
that is optimized towards subsequent rays. For example, if multiple rays are traced from the
same origin with cheap shading, such as in an ambient occlusion shader, reordering after each
ray may increase divergence in the ray origin for the next ray. Instead, it can be better to
reorder once before the loop based on the common origin. In a path tracing loop, it can make
sense to reorder for each radiance ray, but not for the one-off shadow ray from which the light
path is not continued.

Generally speaking, when and where SER is beneficial varies by application and depends on
a number of factors. The API can be integrated into most engines with low engineering effort,
which makes it easy to experiment with the feature. A good starting point is to replace
traditional optixTrace calls with optixTraverse / optixReorder / optixInvoke equivalent,
which is often enough to show some initial performance gains. From there, further
optimizations like coherence hints, live-state reduction, etc., can be explored.


8.3.2 Optimizing warp coherence
--------------------------------

Most developers familiar with GPU programming are accustomed to the concept of warp
coherence and it is one of the most effective metrics for SER optimization. Warp coherence,
that is, the number of threads active in a warp on average, is a good proxy for the general
execution coherence improvements that an application will achieve by using optixReorder .

8.3.3 Optimizing live state
----------------------------

Reordering threads across execution units on the GPU also requires migrating their live state
through the memory system. Live state consists of any variables that are defined before
reordering and used after reordering; that is, any variables that must persist across an
optixReorder call. The smaller that state is, the lower the overhead of the optixReorder call.
Applications should therefore strive to reduce live state as much as possible.

Manually optimizing for live state can be challenging, mainly for two reasons. First, unless
the shader code is very simple, it is hard to spot variables that are live across an
optixReorder call just by inspecting the shader code. Second, the compiler can often
eliminate much of the live-state memory traffic, which might lead to the developer
optimizing for variables that do not affect performance to begin with.

Applications should target OptiX-IR (page 64) and use CUDA’s __restrict__ keyword
wherever possible. OptiX-IR provides more state information to the compiler than PTX,
which improves the accuracy of the compiler’s live-state analysis. This, in turn, enables
additional optimizations and in many cases reduces migrated state. The
__restrict__ keyword also carries additional information in OptiX-IR modules, enabling
the compiler to avoid live-state migration caused by function calls. Note that the
__restrict__ keyword improves live-state analysis in OptiX-IR only; it is ignored in PTX.
For more information about the role of the __restrict__ keyword, see the CUDA
documentation. 1


1. https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__restrict__#restrict



8.3.4 Using coherence hint bits judiciously
--------------------------------------------

As discussed in previous sections, coherence hints are an important mechanism that the
developer can use to insert application-side knowledge into the reordering. The more
coherence-hint bits are used, the higher the granularity at which an application can control
reordering. However, using more coherence-hint bits also reduces the resolution at which the
hit object can be taken into account. Developers should keep this trade-off in mind and not
use more coherence-hint bits than necessary. In particular, numCoherenceHintBitsFromLSB
should only be set to the number of coherence-hint bits that are actually relevant, and no
higher.


8.3.5 Tailoring payload types to invoked shaders
-------------------------------------------------

When replacing a call to optixTrace with its hit object equivalents, it is most straightforward
to use the same payload type in both optixTraverse and optixInvoke . However, it is not
mandatory that the payload types used in these functions match. The only requirement is that
the payload type used at each call site matches the expectations in the shaders that are
actually executed. It is not unusual that the payload requirements between shaders invoked
from optixTraverse (any-hit and intersection) and those invoked from optixInvoke
(closest-hit and miss) can differ. A frequent example is that any-hit shaders require a smaller
payload than closest-hit, because any-hit shaders are only used for simple alpha testing. In
cases like these, an application should use different actual payload types — ideally combined
with a payload type as described in the API reference. Using different payload types
guarantee that register pressure from unnecessary payload fields is avoided as much as
possible, which can help increase overall performance. (See “Interaction with payload
semantic types” (page 113).)


8.4 API Reference
------------------

8.4.1 Querying optixReorder behavior
-------------------------------------

The optixReorder API is available on all raytracing-capable GPUs, but its behavior depends
on the GPU architecture. In most cases, an application can ignore this fact, and can simply use
optixReorder as if reordering was always supported. This works because thread reordering
affects performance, but not correctness. optixReorder acting as a no-op on previous
generation GPUs can therefore be viewed as an implementation of reordering that just
happens to reproduce its input order (producing no benefit, but also not incurring any cost).

However, there are cases where it can make sense to make higher-level decisions based on the
behavior of optixReorder . For example, an application might have a legacy path that
performs manual shader binning, which it would like to enable for GPUs that do not support
SER reordering. The behavior of optixReorder can be queried using
optixDeviceContextGetProperty .

::

     // Listing 8.9
     unsigned int SERFlags = 0;
     optixDeviceContextGetProperty(
          s_context,
          OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING,
          &SERFlags, sizeof( unsigned int ) );

          // Check to see if this device can support reordering.
          if( SERFlags == OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING_FLAG_NONE ) {
              std::cout << "Device does not support SER";
          }


8.4.1.1 optixTraverse
~~~~~~~~~~~~~~~~~~~~~~

Similar to optixTrace , but does not invoke the closest-hit or miss shader. Instead, it
overwrites the current outgoing hit object with the results of traversing the ray. The outgoing
hit object may be invoked at some later point with optixInvoke or used with optixReorder .
The outgoing hit object can also be queried through various functions such as
optixHitObjectIsHit or optixHitObjectGetAttribute_0 .

::

    // Listing 8.10
    template <typename... Payload>
    void optixTraverse(
        OptixTraversableHandle handle,
        float3 rayOrigin,
        float3 rayDirection,
        float  tmin,
        float  tmax,
        float  rayTime,
        OptixVisibilityMask visibilityMask,
        unsigned int rayFlags,
        unsigned int SBToffset,
        unsigned int SBTstride,
        unsigned int missSBTIndex,
        Payload&... payload );


    // Listing 8.11
    template <typename... Payload>
    void optixTraverse(
        OptixPayloadTypeID     type,
        OptixTraversableHandle handle,
        float3 rayOrigin,
        float3 rayDirection,
        float  tmin,
        float  tmax,
        float  rayTime,
        OptixVisibilityMask visibilityMask,
        unsigned int rayFlags,
        unsigned int SBToffset,
        unsigned int SBTstride,
        unsigned int missSBTIndex,
        Payload&... payload );


8.4.1.2 optixMakeHitObject
--------------------------

Constructs an outgoing hit object from the hit object data provided. The traverseData needs
to be collected from a previous hit. This hit object will now become the current outgoing hit
object and will overwrite the current outgoing hit object.

::

    //Listing 8.12
    template <typename... RegAttributes>
    void optixMakeHitObject(
        OptixTraversableHandle handle,
        float3 rayOrigin,
        float3 rayDirection,
        float tmin,
        float rayTime,
        unsigned int rayFlags,
        OptixTraverseData traverseData,
        const OptixTraversableHandle* transforms,
        unsigned int numTransforms );


::

     205 /// Constructs an outgoing hit object from the hit object data provided. The traverseData needs to be collected from a previous hit
     206 /// object using #optixHitObjectGetTraverseData.
     207 /// This hit object will now become the current outgoing hit object and will overwrite the current outgoing hit object.
     208 ///
     209 /// \param[in] handle
     210 /// \param[in] rayOrigin
     211 /// \param[in] rayDirection
     212 /// \param[in] tmin
     213 /// \param[in] rayTime
     214 /// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
     215 /// \param[in] traverseData
     216 /// \param[in] transforms
     217 /// \param[in] numTransforms
     218 ///
     219 /// Available in RG, CH, MS, CC
     220 static __forceinline__ __device__ void optixMakeHitObject( OptixTraversableHandle        handle,
     221                                                            float3                        rayOrigin,
     222                                                            float3                        rayDirection,
     223                                                            float                         tmin,
     224                                                            float                         rayTime,
     225                                                            unsigned int                  rayFlags,
     226                                                            OptixTraverseData             traverseData,
     227                                                            const OptixTraversableHandle* transforms,
     228                                                            unsigned int                  numTransforms );
     229 



     260 /// Serializes the current outgoing hit object which allows to recreate it at a later
     261 /// point using #optixMakeHitObject.
     262 ///
     263 /// \param[out] data
     264 ///
     265 /// Available in RG, CH, MS, CC, DC
     266 static __forceinline__ __device__ void optixHitObjectGetTraverseData( OptixTraverseData* data );








8.4.1.7 The hit object’s state
-------------------------------

Retrieve information from the current outgoing hit object. An implied nop hit object is always
assumed to exist even if there are no calls to optixTraverse , optixMakeHitObject ,
optixMakeMissHitObject , or optixMakeNopHitObject .

::

    void optixHitObjectGetTraverseData(
    OptixTraverseData* data );
    bool optixHitObjectIsMiss();
    bool optixHitObjectIsHit();
    bool optixHitObjectIsNop();
    unsigned int optixHitObjectGetInstanceId();
    unsigned int optixHitObjectGetInstanceIndex();
    unsigned int optixHitObjectGetPrimitiveIndex();
    unsigned int optixHitObjectGetTransformListSize();
    OptixTraversableHandle optixHitObjectGetTransformListHandle(unsigned int index );
    unsigned int optixHitObjectGetSbtGASIndex();
    unsigned int optixHitObjectGetHitKind();
    float3 optixHitObjectGetWorldRayOrigin();
    float3 optixHitObjectGetWorldRayDirection();
    float optixHitObjectGetRayTmin();
    float optixHitObjectGetRayTmax();
    float optixHitObjectGetRayTime();
    unsigned int optixHitObjectGetRayFlags();
    unsigned int optixHitObjectGetAttribute_0();
    unsigned int optixHitObjectGetAttribute_1();
    unsigned int optixHitObjectGetAttribute_2();
    unsigned int optixHitObjectGetAttribute_3();
    unsigned int optixHitObjectGetAttribute_4();
    unsigned int optixHitObjectGetAttribute_5();
    unsigned int optixHitObjectGetAttribute_6();
    unsigned int optixHitObjectGetAttribute_7();
    unsigned int optixHitObjectGetSbtRecordIndex();
    CUdeviceptr optixHitObjectGetSbtDataPointer();
    OptixTraversableHandle optixHitObjectGetGASTraversableHandle();
    void optixHitObjectGetTriangleVertexData( float3 data[3] );
    void optixHitObjectGetLinearCurveVertexData( float4 data[2] );
    void optixHitObjectGetQuadraticBSplineVertexData( float4 data[3] );
    void optixHitObjectGetQuadraticBSplineRocapsVertexData( float4 data[3] );
    void optixHitObjectGetCubicBSplineVertexData( float4 data[4] );

    void optixHitObjectGetCubicBSplineRocapsVertexData( float4 data[4] );
    void optixHitObjectGetCatmullRomVertexData( float4 data[4] );
    void optixHitObjectGetCatmullRomRocapsVertexData( float4 data[4] );
    void optixHitObjectGetCubicBezierVertexData( float4 data[4] );
    void optixHitObjectGetCubicBezierRocapsVertexData( float4 data[4] );
    void optixHitObjectGetRibbonVertexData( float4 data[3] );
    float3 optixHitObjectGetRibbonNormal( float2 ribbonParameters );
    void optixHitObjectGetSphereData( float4 data[1] );
    void optixHitObjectGetWorldToObjectTransformMatrix( float m[12] );
    void optixHitObjectGetObjectToWorldTransformMatrix( float m[12] );
    float3 optixHitObjectTransformPointFromWorldToObjectSpace( float3 point );
    float3 optixHitObjectTransformVectorFromWorldToObjectSpace( float3 vec );
    float3 optixHitObjectTransformNormalFromWorldToObjectSpace( float3 normal );
    float3 optixHitObjectTransformPointFromObjectToWorldSpace( float3 point );
    float3 optixHitObjectTransformVectorFromObjectToWorldSpace( float3 vec );
    float3 optixHitObjectTransformNormalFromObjectToWorldSpace( float3 normal );
    unsigned int optixHitObjectGetClusterId();
    float2 optixHitObjectGetTriangleBarycentrics();
    float optixHitObjectGetCurveParameter();
    float2 optixHitObjectGetRibbonParameters();





8.4.2 optixReorder
------------------

Reorders threads based on the current outgoing hit object and a coherence hint value.
Parameter NumCoherenceHintBits indicates how many of the least significant bits of
CoherenceHint should be considered during reordering, with a maximum of 16.
Applications should set this to the lowest value required to represent all possible values in
coherenceHint . For best performance, all threads should provide the same value for
numCoherenceHintBits::

    // Listing 8.19
    void optixReorder(
    unsigned int coherenceHint,
    unsigned int numCoherenceHintBits );


Reordering will consider information in the hit object and coherence hint with the following
priority:

1. Shader ID stored in the hit object
2. Coherence hint, with the most significant hint bit having highest priority
3. Spatial information stored in the hit object

This ordering implies that optixReorder will first attempt to group threads whose hit object
references the same shader ID. (Miss shaders and nop hit objects are grouped separately).
Within each of these groups, it will attempt to order threads by the value of their coherence
hints. Within ranges of equal coherence hints, it will also attempt to maximize locality in 3D
space of the ray hit (if any).

You can use optixReorder without arguments as a convenience function that supplies 0 for
numCoherenceHintBits thereby ignoring any values in coherenceHint . Only information in
the current outgoing hit object will be considered when reordering.




OPTICKS THOUGHTS
------------------

Default spatial understanding for coherence based on where the ray tracing hit is
BUT with photon simulation the appropriate spatial info is not at the boundary but
along the path of they ray still in the bulk where the scatter or reemission happens.

How to handle ?

* update hit object to inform default coherence of photon position ?
* add spatial

often not get to the boundary - so the correct spatial info
     is where the photon is (eg where scatter, happended)


   
API THOUGHTS
------------

1. optixInvoke consumes the outgoing hit/miss from optixTraverse yielding outgoing nop
2. higher level optixTrace internally does optixTraverse+optixInvoke - so it yields the same outgoing nop that optixInvoke does





Question 1 : photon simulation hit object spatial correction to scatter/reemit position ?
------------------------------------------------------------------------------------------

How is optixMakeHitObject used ? I am wordering for Opticks optical photon
simulation. The default coherence reordering uses spatial information from the
hit - but that information will often be incorrect for optical photon
simulation because the photon often does not get propagated to the ray trace
boundary - but rather it gets moved to the position of scatter or reemission ?

CRITIQUE::

    YES - BUT THE NEXT RAY WILL START FROM THE PHOTON POSITION ?

Can the spatial info used by optix reordering be controlled by correcting
spatial info with MakeHitObject ?  How to get the transforms ?


     205 /// Constructs an outgoing hit object from the hit object data provided. The traverseData needs to be collected from a previous hit
     206 /// object using #optixHitObjectGetTraverseData.
     207 /// This hit object will now become the current outgoing hit object and will overwrite the current outgoing hit object.
     208 ///
     209 /// \param[in] handle
     210 /// \param[in] rayOrigin
     211 /// \param[in] rayDirection
     212 /// \param[in] tmin
     213 /// \param[in] rayTime
     214 /// \param[in] rayFlags       really only 16 bits, combination of OptixRayFlags
     215 /// \param[in] traverseData
     216 /// \param[in] transforms
     217 /// \param[in] numTransforms
     218 ///
     219 /// Available in RG, CH, MS, CC
     220 static __forceinline__ __device__ void optixMakeHitObject( OptixTraversableHandle        handle,
     221                                                            float3                        rayOrigin,
     222                                                            float3                        rayDirection,
     223                                                            float                         tmin,
     224                                                            float                         rayTime,
     225                                                            unsigned int                  rayFlags,
     226                                                            OptixTraverseData             traverseData,
     227                                                            const OptixTraversableHandle* transforms,
     228                                                            unsigned int                  numTransforms );
     229 



     260 /// Serializes the current outgoing hit object which allows to recreate it at a later
     261 /// point using #optixMakeHitObject.
     262 ///
     263 /// \param[out] data
     264 ///
     265 /// Available in RG, CH, MS, CC, DC
     266 static __forceinline__ __device__ void optixHitObjectGetTraverseData( OptixTraverseData* data );





GEMINI::

    Yes, you can directly control and override the spatial information used by
    OptiX shader execution reordering (optixReorder) by supplying a corrected
    rayOrigin (and adjusted ray parameters) to optixMakeHitObject.

    In standard rendering, optixReorder relies on the spatial data encapsulated in
    the current hit object to cluster neighboring threads into spatial bins. For an
    optical photon simulation (like Opticks) where a photon scatters or re-emits in
    the bulk medium, the geometric boundary intersection stored in the default hit
    object is physically irrelevant. By using optixMakeHitObject to inject the
    actual scatter or re-emission coordinate as the new rayOrigin, you force
    optixReorder to group threads based on where the photons are currently
    interacting in the volume, drastically reducing warp divergence.

    How to Retrieve the Transforms

    To properly construct or modify a hit object that preserves the transform
    hierarchy (such as instance matrices and motion/static transforms), you need to
    feed the active transform list into the transforms and numTransforms arguments
    of optixMakeHitObject.

    You can extract these directly from the current execution context using OptiX
    device helper functions:

    unsigned int numTransforms = optixGetTransformListSize();
    OptixTraversableHandle transforms[8]; // Ensure your array is large enough for your max nesting depth
    for (unsigned int i = 0; i < numTransforms && i < 8; ++i) {
        transforms[i] = optixGetTransformListHandle(i);
    }

    When your photon undergoes bulk scattering or re-emission, you can override
    the hit object state like this:


    // 1. Get existing traversal state and transform list
    OptixTraverseData traverseData;
    optixHitObjectGetTraverseData(&traverseData);

    unsigned int numTransforms = optixGetTransformListSize();
    OptixTraversableHandle transforms[4];
    for(unsigned int i = 0; i < numTransforms && i < 4; ++i) {
        transforms[i] = optixGetTransformListHandle(i);
    }

    // 2. Define your new photon state (e.g., the scatter/re-emission point)
    float3 newRayOrigin = scatterPosition; 
    float3 newRayDirection = sampledScatterDirection;
    float  newTmin = 0.0f;
    float  newRayTime = 0.0f;
    unsigned int newRayFlags = OPTIX_RAY_FLAG_NONE;

    // 3. Rebuild the hit object with corrected spatial data
    optixMakeHitObject(
        topTraversableHandle, 
        newRayOrigin, 
        newRayDirection, 
        newTmin, 
        newRayTime, 
        newRayFlags, 
        traverseData, 
        transforms, 
        numTransforms
    );

    // 4. Reorder threads based on this corrected spatial/material context
    optixReorder(coherenceHint, numCoherenceHintBits);



    By doing this, subsequent execution steps and reordering calls will treat
    newRayOrigin as the anchor point for spatial coherence, aligning threads that
    are experiencing similar local volume conditions.

    Would you like to explore how to combine these corrected hit objects with
    custom coherenceHint bitmasks to further optimize warp execution for different
    physics processes like Rayleigh scattering versus boundary reflection?



