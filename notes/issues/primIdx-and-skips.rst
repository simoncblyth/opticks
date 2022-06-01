primIdx-and-skips
====================

Summary 
---------

The identity/naming machinery is working just fine the problem
is consistency on making dynamic geometry selections. Specifically the problem
is using results from a dynamically changed geometry with the unchanged geometry
as reference. The inconsistency gives mis-namings.  

To avoid the bookkeeping kludges it is necessary that when a dynamic prim selection 
is done changing the geometry that the cfbase of the dst geometry is changed 
and the changed geometry is saved to it and all outputs are then associated 
with that changed geometry "cfbase".   Then have to only be concerned with a single CFBase. 

BUT thats quite a rigmarole. 


Consider using CSGOptiXSimtraceTest and CSGOptiXSimTest together
-------------------------------------------------------------------

It is problematic to apply dynamic prim selection twice by both those executables.
Much clearer to do the prim selection for the skips once when translating geometry
yielding a single CSGFoundry.   

Dynamic prim selection without saving the CSGFoundry of the modified geometry 
can be useful for render performance scanning to find geometry bottlenecks 
but it is just not appropriate when wishing to run multiple executables over the same geometry 
and do detailed analysis of the results. In this situation its vital to have a more constant 
CSGFoundry geometry folder that is read by multiple executables including python analysis
machinery. 


How to act on that and avoid having to kludge to keep consistent ?
---------------------------------------------------------------------

CSGOptiXSimtraceTest.cc::

     78     qs->save(); // uses SGeo::LastUploadCFBase_OutDir to place outputs into CFBase/ExecutableName folder sibling to CSGFoundry   
     79 
     80 
     81     // TODO: move save control to SEvt not QSim/QEvent 
     82     // for example CPU only tests need to save too, so it makes no sense for them to reach up to QUDARap to control that 
     83     const char* dir = QEvent::DefaultDir();
     84     cx->fr.save(dir);
     85 
     86     return 0 ;
     87 }


::

    0263 void QSim::save() const
     264 {
     265     event->save();
     266 }

    569 void QEvent::save() const
    570 {
    571     const char* dir = DefaultDir();
    572     LOG(info) << "DefaultDir " << dir ;
    573     save(dir);
    574 }

    038 const char* QEvent::FALLBACK_DIR = "$TMP" ;
     39 const char* QEvent::efaultDir()
     40 {
     41     const char* dir_ = SGeo::LastUploadCFBase_OutDir();
     42     const char* dir = dir_ ? dir_ : FALLBACK_DIR  ;
     43     return dir ;
     44 }


    008 /**
      9 SGeo::SetLastUploadCFBase
     10 ---------------------------
     11 
     12 Canonically invoked from CSGFoundry::upload with CSGFoundry::getOriginCFBase
     13 
     14 **/
     15 
     16 void SGeo::SetLastUploadCFBase(const char* cfbase)
     17 {
     18     LAST_UPLOAD_CFBASE = cfbase ? strdup(cfbase) : nullptr ;
     19 }
     20 const char* SGeo::LastUploadCFBase()
     21 {
     22     return LAST_UPLOAD_CFBASE ;
     23 }
     24 
     25 /**
     26 SGeo::LastUploadCFBase_OutDir
     27 ------------------------------
     28 
     29 This provides a default output directory to QEvent::save
     30 which is within the last uploaded CFBase/ExeName
     31 
     32 **/
     33 const char* SGeo::LastUploadCFBase_OutDir()
     34 {
     35     const char* cfbase = LastUploadCFBase();
     36     if(cfbase == nullptr) return nullptr ;
     37     const char* exename = SProc::ExecutableName();
     38     const char* outdir = SPath::Resolve(cfbase, exename, DIRPATH );
     39     return outdir ;
     40 }




Try to simply write the dynamic geometry to CFBaseAlt so have consistency
------------------------------------------------------------------------------

HMM but that means the outputs should really be there in Alt together with the actual geometry::

    epsilon:1 blyth$ pwd
    /Users/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1

    epsilon:1 blyth$ l
    total 0
    0 drwxr-xr-x  3 blyth  staff   96 Jun  1 14:40 CSG_GGeo_Alt
    0 drwxr-xr-x  4 blyth  staff  128 Jun  1 14:40 .
    0 drwxr-xr-x  8 blyth  staff  256 May 24 15:45 CSG_GGeo
    0 drwxr-xr-x  3 blyth  staff   96 Mar  4 14:06 ..
    epsilon:1 blyth$ 
    epsilon:1 blyth$ 
    epsilon:1 blyth$ l CSG_GGeo/
    total 0
    0 drwxr-xr-x   4 blyth  staff  128 Jun  1 14:40 ..
    0 drwxr-xr-x  14 blyth  staff  448 May 30 15:38 CSGOptiXSimTest
    0 drwxr-xr-x  13 blyth  staff  416 May 29 20:35 CSGOptiXSimtraceTest
    0 drwxr-xr-x   8 blyth  staff  256 May 24 15:45 .
    0 drwxr-xr-x  20 blyth  staff  640 May 20 16:44 CSGFoundry
    0 drwxr-xr-x   3 blyth  staff   96 Mar 16 17:57 CSGIntersectSolidTest
    0 drwxr-xr-x   2 blyth  staff   64 Mar  4 14:23 CSGOptiXSimulateTest
    0 drwxr-xr-x   3 blyth  staff   96 Mar  4 14:04 CSGOptiXRenderTest
    epsilon:1 blyth$ l CSG_GGeo_Alt/
    total 0
    0 drwxr-xr-x   3 blyth  staff   96 Jun  1 14:40 .
    0 drwxr-xr-x   4 blyth  staff  128 Jun  1 14:40 ..
    0 drwxr-xr-x  11 blyth  staff  352 Jun  1 14:31 CSGFoundry
    epsilon:1 blyth$ 
    epsilon:1 blyth$ 


Kludge it with symbolic links, for now::

    epsilon:CSG_GGeo_Alt blyth$ ln -s ../CSG_GGeo/CSGOptiXSimtraceTest
    epsilon:CSG_GGeo_Alt blyth$ ln -s ../CSG_GGeo/CSGOptiXSimTest


cachegrab.sh::

    261 elif [ "$(uname)" == "Darwin" ]; then
    262 
    263     echo $cxs_msg Darwin $(pwd) LINENO $LINENO
    264 
    265     
    266     if [ "${cxs_arg}" == "grab" ]; then
    267         echo $cxs_msg grab LINENO $LINENO 
    268         EXECUTABLE=$bin       source cachegrab.sh grab
    269         CGREL=CSG_GGeo_Alt EXECUTABLE=CSGFoundry source cachegrab.sh grab
    270         ## NASTY MIXED CFBase THATS ONLY WORKING DUE TO KLUDGE SYMBOLIC LINKS IN CSG_GGeo_Alt/CSGOptiXSimtraceTest  
    271     else
    272         echo $cxs_msg cxs_arg $cxs_arg LINENO $LINENO
    273         CGREL=CSG_GGeo_Alt EXECUTABLE=$bin       source cachegrab.sh env
    274         ## NASTY MIXED CFBase THATS ONLY WORKING DUE TO KLUDGE SYMBOLIC LINKS IN CSG_GGeo_Alt/CSGOptiXSimtraceTest  
    275         
    276         cxs_dumpvars "FOLD CFBASE CGREL" after cachegrab.sh env
    277         
 


Trace detail of where the mis-naming happens
----------------------------------------------------

ana/feature.py::

    cf.primIdx_meshname_dict()

CSG/CSGFoundry.py::

    278     def meshIdx(self, primIdx):
    279         """
    280         """
    281         assert primIdx < len(self.prim)
    282         midx = self.prim[primIdx].view(np.uint32)[1,1]
    283         return midx 
    284         
    285     def primIdx_meshname_dict(self):
    286         """
    287         See notes/issues/cxs_2d_plotting_labels_suggest_meshname_order_inconsistency.rst
    288         """
    289         d = {}
    290         for primIdx in range(len(self.prim)):
    291             midx = self.meshIdx (primIdx)      # meshIdx method with contiguous primIdx argumnet
    292             assert midx < len(self.meshname)
    293             mnam = self.meshname[midx]
    294             d[primIdx] = mnam
    295             #print("CSGFoundry:primIdx_meshname_dict primIdx %5d midx %5d meshname %s " % (primIdx, midx, mnam))
    296         pass
    297         return d
    298         



DONE : work out way to handle prim skips with proper identity : JUST NEED TO ARRANGE CONSISTENTLY KEEPING RESULTS TOGETHER WITH THE CORRESPONDING GEOMETRY
-------------------------------------------------------------------------------------------------------------------------------------------------------------

From below, the primIdx in the OptiX machinery is the flat contiguous index 
from the uploaded CSGFoundry geometry. The CSGCopy just passes over the meshIdx::


    156 void CSGCopy::copySolidPrim(AABB& solid_bb, int dPrimOffset, const CSGSolid* sso )
    157 {
    158     unsigned dump_ = Dump(sSolidIdx);
    159     bool dump_prim = ( dump_ & 0x2 ) != 0u ;
    160 
    161     for(int primIdx=sso->primOffset ; primIdx < sso->primOffset+sso->numPrim ; primIdx++)
    162     {
    163          const CSGPrim* spr = src->getPrim(primIdx);
    164          unsigned meshIdx = spr->meshIdx() ;
    165          unsigned repeatIdx = spr->repeatIdx() ;
    166          bool selected = elv == nullptr ? true : elv->is_set(meshIdx) ;
    167          if( selected == false ) continue ;
    168 
    169          unsigned numNode = spr->numNode()  ;  // not envisaging node selection, so this will be same in src and dst 
    170          unsigned dPrimIdx_global = dst->getNumPrim() ;            // destination numPrim prior to prim addition
    171          unsigned dPrimIdx_local = dPrimIdx_global - dPrimOffset ; // make the PrimIdx local to the solid 
    172 
    173          CSGPrim* dpr = dst->addPrim(numNode, -1 );
    174          if( elv == nullptr ) assert( dpr->nodeOffset() == spr->nodeOffset() );
    175 
    176          dpr->setMeshIdx(meshIdx);
    177          dpr->setRepeatIdx(repeatIdx);
    178          dpr->setPrimIdx(dPrimIdx_local);
    179 
    180          AABB prim_bb = {} ;
    181          copyPrimNodes(prim_bb, spr );
    182          dpr->setAABB( prim_bb.data() );
    183          //dpr->setAABB( spr->AABB() );  // will not be so with selection 
    184 


  

HMM: CSG_GGeo_Convert just passes across all names independent of skips 
----------------------------------------------------------------------------

* YES: but this is just fine : the CSGPrim references the meshIdx, and do not remove meshIdx as change geometry : just treat those as absolute
* the issue is using old geometry with results from a new geometry, the new geometry having had the CSG::CopySelect skips applied


::

    076 void CSG_GGeo_Convert::init()
     77 {
     78     ggeo->getMeshNames(foundry->meshname);
     79     // ggeo->getBoundaryNames(foundry->bndname);   // boundary names now travel with the NP bnd.names 
     80     ggeo->getMergedMeshLabels(foundry->mmlabel);

    1022 void GGeo::getMeshNames(std::vector<std::string>& meshNames) const
    1023 {
    1024      m_meshlib->getMeshNames(meshNames);
    1025 }

    812 void GMeshLib::getMeshNames(std::vector<std::string>& meshNames) const
    813 {
    814     meshNames.clear();
    815     unsigned numMeshes = getNumMeshes();
    816     for(unsigned midx=0 ; midx < numMeshes ; midx++)
    817     {
    818         const char* mname = getMeshName(midx);
    819         meshNames.push_back(mname);
    820     }
    821 }


    0261 CSGSolid* CSG_GGeo_Convert::convertSolid( unsigned repeatIdx )
     262 {
     ...
     297     for(unsigned primIdx=0 ; primIdx < numPrim ; primIdx++)
     298     {
     299         unsigned meshIdx   = comp->getMeshIndex(primIdx);   // from idxBuffer aka lvIdx 
     300         const char* mname = foundry->getName(meshIdx);      //  
     301         bool cxskip = SGeoConfig::IsCXSkipLV(meshIdx);
     302 
     303         LOG(LEVEL) << " cxskip " << cxskip << " meshIdx " << meshIdx << " mname " << mname ;
     304         if(cxskip)
     305         {
     306             LOG(error) << " cxskip " << cxskip << " meshIdx " << meshIdx << " mname " << mname ;
     307             continue ;
     308         }
     309 
     310         CSGPrim* prim = convertPrim(comp, primIdx);
     311         bb.include_aabb( prim->AABB() );
     312 
     313         unsigned sbtIdx = prim->sbtIndexOffset() ;  // from CSGFoundry::addPrim
     314         //assert( sbtIdx == primIdx  );    // HMM: not with skips
     315         assert( sbtIdx == solidPrimChk  );
     316 
     317         prim->setRepeatIdx(repeatIdx);
     318         prim->setPrimIdx(primIdx);
     319 
     320         solidPrimChk += 1 ;
     321     } 
     322     // NB when SGeoConfig::IsCXSkipLV skips are used the primIdx set by CSGPrim::setPrimIdx will not be contiguous   
     323     // Q: Does the OptiX identity machinery accomodate this assigned primIdx  ?
     324     // A: I think the answer is currently NO 
     325     //    
     326     //    The value returned from optixGetPrimitiveIndex is the 0-based index of the bbox within the GAS plus a bias 
     327     //    that is passed into the GAS and currently comes from CSGSolid so->primOffset which is just the number of 
     328     //    primitives so far collected. 
     329     //  
     330     

::

    072 /**
     73 GAS_Builder::MakeCustomPrimitivesBI_11N
     74 -----------------------------------------
     75 
     76 References to bbox array from CSGPrimSpec copyied into the BI
     77 
     78 Creates buildInput using device refs of pre-uploaded aabb for all prim (aka layers) of the Solid
     79 and arranges for separate SBT records for each prim.
     80 
     81 Added primitiveIndexOffset to CSGPrimSpec in attempt to get identity info 
     82 regarding what piece of geometry is intersected/closesthit. 
     83 
     84 **/
     85 
     86 BI GAS_Builder::MakeCustomPrimitivesBI_11N(const CSGPrimSpec& ps)
     87 {
     88     assert( ps.device == true );
     89     assert( ps.stride_in_bytes % sizeof(float) == 0 );
     90 
     91     BI bi = {} ;
     92     bi.mode = 1 ;
     93     bi.flags = new unsigned[ps.num_prim];
     94     for(unsigned i=0 ; i < ps.num_prim ; i++) bi.flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ;
     95 
     96     // http://www.cudahandbook.com/2013/08/why-does-cuda-cudeviceptr-use-unsigned-int-instead-of-void/ 
     97     // CUdeviceptr is typedef to unsigned long long 
     98     // uintptr_t is an unsigned integer type that is capable of storing a data pointer.
     99 
    100     bi.d_aabb = (CUdeviceptr) (uintptr_t) ps.aabb ;
    101     bi.d_sbt_index = (CUdeviceptr) (uintptr_t) ps.sbtIndexOffset ;
    102 
    103     bi.buildInput = {};
    104     bi.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    105     OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
    106     buildInputCPA.aabbBuffers = &bi.d_aabb ;
    107     buildInputCPA.numPrimitives = ps.num_prim  ;
    108     buildInputCPA.strideInBytes = ps.stride_in_bytes ;
    109     buildInputCPA.flags = bi.flags;                                  // flags per sbt record
    110     buildInputCPA.numSbtRecords = ps.num_prim ;                      // number of sbt records available to sbt index offset override. 
    111     buildInputCPA.sbtIndexOffsetBuffer  = bi.d_sbt_index ;           // Device pointer to per-primitive local sbt index offset buffer, Every entry must be in range [0,numSbtRecords-1]
    112     buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);     // Size of type of the sbt index offset. Needs to be 0,     1, 2 or 4    
    113     buildInputCPA.sbtIndexOffsetStrideInBytes = ps.stride_in_bytes ; // Stride between the index offsets. If set to zero, the offsets are assumed to be tightly packed.
    114     buildInputCPA.primitiveIndexOffset = ps.primitiveIndexOffset ;   // Primitive index bias, applied in optixGetPrimitiveIndex() see OptiX7Test.cu:__closesthit__ch
    115 



__closesthit__ch
-------------------

::

    402 extern "C" __global__ void __closesthit__ch()
    403 {   
    404     //unsigned instance_index = optixGetInstanceIndex() ;  0-based index within IAS
    405     unsigned instance_id = optixGetInstanceId() ;  // user supplied instanceId, see IAS_Builder::Build and InstanceId.h 
    406     unsigned prim_idx = optixGetPrimitiveIndex() ;  // GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    407     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_id & 0xffff ) ;
    408 


primitiveIndexOffset is crucial bias applied to what *optixGetPrimitiveIndex* returns
----------------------------------------------------------------------------------------

::

    epsilon:CSGOptiX blyth$ opticks-f primitiveIndexOffset
    ./CSGOptiX/GAS_Builder.cc:        << " ps.primitiveIndexOffset " << ps.primitiveIndexOffset
    ./CSGOptiX/GAS_Builder.cc:Added primitiveIndexOffset to CSGPrimSpec in attempt to get identity info 
    ./CSGOptiX/GAS_Builder.cc:    buildInputCPA.primitiveIndexOffset = ps.primitiveIndexOffset ;   // Primitive index bias, applied in optixGetPrimitiveIndex() see OptiX7Test.cu:__closesthit__ch
    ./CSGOptiX/GAS_Builder.cc:        << " buildInputCPA.primitiveIndexOffset " << buildInputCPA.primitiveIndexOffset
    ./CSG/CSGPrim.cc:CSGPrimSpec::primitiveIndexOffset
    ./CSG/CSGPrim.cc:    ps.primitiveIndexOffset = primIdx ;   
    ./CSG/CSGPrimSpec.cc:       << " primitiveIndexOffset " << std::setw(4) << primitiveIndexOffset
    ./CSG/CSGPrimSpec.h:    unsigned        primitiveIndexOffset ;   // offsets optixGetPrimitiveIndex() see GAS_Builder::MakeCustomPrimitivesBI_11N
    ./externals/rcs.bash:    519 /// plus the primitiveIndexOffset.
    ./externals/rcs.bash:     385     /// Sum of primitiveIndexOffset and number of primitive must not overflow 32bits.
    ./externals/rcs.bash:     386     unsigned int primitiveIndexOffset;
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/GAS_Builder.cc:    unsigned primitiveIndexOffset = i ; 
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/GAS_Builder.cc:    buildInputCPA.primitiveIndexOffset = primitiveIndexOffset ;  // Primitive index bias, applied in optixGetPrimitiveIndex()
    epsilon:opticks blyth$ 



::

    061 How to implement Prim selection ?
     62 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     63 
     64 Applying Prim selection based on meshIdx/lvIdx of each 
     65 Prim still requires to iterate over them all.
     66 Better to apply selection in one place only. 
     67 So where to apply prim selection ?
     68 
     69 CSGPrimSpec is too late as the prim array handled
     70 there needs to be memory contiguous.   
     71 This suggests addition of selected_prim to CSGFoundry::
     72 
     73     std::vector<CSGPrim>  prim ;
     74     std::vector<CSGPrim>  selected_prim ;
     75 
     76 Must also ensure no blind passing of primOffsets as they 
     77 will be invalid. 
     78 
     79 **/
     80 
     81 CSGPrimSpec CSGPrim::MakeSpec( const CSGPrim* prim0,  unsigned primIdx, unsigned numPrim ) // static 
     82 {
     83     const CSGPrim* prim = prim0 + primIdx ;
     84 
     85     CSGPrimSpec ps ;
     86     ps.aabb = prim->AABB() ;
     87     ps.sbtIndexOffset = prim->sbtIndexOffsetPtr() ;
     88     ps.num_prim = numPrim ;
     89     ps.stride_in_bytes = sizeof(CSGPrim);
     90     ps.primitiveIndexOffset = primIdx ;
     91 
     92     return ps ;
     93 }

::

    epsilon:CSG blyth$ opticks-f MakeSpec
    ./CSGOptiX/SBT.cc:Thoughts on how to implement Prim selection with CSGPrim::MakeSpec
    ./CSG/CSGPrim.cc:CSGPrim::MakeSpec
    ./CSG/CSGPrim.cc:CSGPrimSpec CSGPrim::MakeSpec( const CSGPrim* prim0,  unsigned primIdx, unsigned numPrim ) // static 
    ./CSG/tests/CSGPrimImpTest.cc:     CSGPrimSpec psa = CSGPrim::MakeSpec(prim.data(), 0, prim.size() ); 
    ./CSG/tests/CSGPrimImpTest.cc:     CSGPrimSpec ps0 = CSGPrim::MakeSpec(prim.data(), 0, h ); 
    ./CSG/tests/CSGPrimImpTest.cc:     CSGPrimSpec ps1 = CSGPrim::MakeSpec(prim.data(), h, h ); 
    ./CSG/tests/CSGPrimImpTest.cc:     CSGPrimSpec d_ps = CSGPrim::MakeSpec( d_prim, 0, num ); 
    ./CSG/tests/CUTest.cc:    CSGPrimSpec psd = CSGPrim::MakeSpec( d_prim,  primOffset, numPrim ); ;
    ./CSG/CSGPrim.h:    static CSGPrimSpec MakeSpec( const CSGPrim* prim0, unsigned primIdx, unsigned numPrim ) ; 
    ./CSG/CSGFoundry.cc:    CSGPrimSpec ps = CSGPrim::MakeSpec( prim.data(),  so->primOffset, so->numPrim ); ; 
    ./CSG/CSGFoundry.cc:    CSGPrimSpec ps = CSGPrim::MakeSpec( d_prim,  so->primOffset, so->numPrim ); ; 
    ./CSG/CSGPrimSpec.h:* Instances are created for a solidIdx by CSGFoundry::getPrimSpec using CSGPrim::MakeSpec
    epsilon:opticks blyth$ 


    1065 CSGPrimSpec CSGFoundry::getPrimSpecHost(unsigned solidIdx) const
    1066 {
    1067     const CSGSolid* so = solid.data() + solidIdx ;
    1068     CSGPrimSpec ps = CSGPrim::MakeSpec( prim.data(),  so->primOffset, so->numPrim ); ;
    1069     ps.device = false ;
    1070     return ps ;
    1071 }
    1072 CSGPrimSpec CSGFoundry::getPrimSpecDevice(unsigned solidIdx) const
    1073 {
    1074     assert( d_prim );
    1075     const CSGSolid* so = solid.data() + solidIdx ;  // get the primOffset from CPU side solid
    1076     CSGPrimSpec ps = CSGPrim::MakeSpec( d_prim,  so->primOffset, so->numPrim ); ;
    1077     ps.device = true ;
    1078     return ps ;
    1079 }





optixGetPrimitiveIndex : returns primitive index within build array plus the primitiveIndexOffset
---------------------------------------------------------------------------------------------------

::

    513 /// For a given OptixBuildInputTriangleArray the number of primitives is defined as
    514 /// (OptixBuildInputTriangleArray::indexBuffer == nullptr) ? OptixBuildInputTriangleArray::numVertices/3 :
    515 ///                                                          OptixBuildInputTriangleArray::numIndices/3;
    516 ///
    517 /// For a given OptixBuildInputCustomPrimitiveArray the number of primitives is defined as
    518 /// numAabbs.  The primitive index returns is the index into the corresponding build array
    519 /// plus the primitiveIndexOffset.
    520 ///
    521 /// In Intersection and AH this corresponds to the currently intersected primitive.
    522 /// In CH this corresponds to the primitive index of the closest intersected primitive.
    523 /// In EX with exception code OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT corresponds 
            to the active primitive index. Returns zero for all other exceptions.
    524 static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex();



optixGetInstanceId : returns OptixInstance::instanceId of intersected instance
--------------------------------------------------------------------------------

::

    527 /// Returns the OptixInstance::instanceId of the instance within the top level acceleration structure associated with the current intersection.
    528 ///
    529 /// When building an acceleration structure using OptixBuildInputInstanceArray each OptixInstance has a user supplied instanceId.
    530 /// OptixInstance objects reference another acceleration structure.  During traversal the acceleration structures are visited top down.
    531 /// In the Intersection and AH programs the OptixInstance::instanceId corresponding to the most recently visited OptixInstance 
            is returned when calling optixGetInstanceId().
    532 /// In CH optixGetInstanceId() returns the OptixInstance::instanceId when the hit was recorded with optixReportIntersection.
    533 /// In the case where there is no OptixInstance visited, optixGetInstanceId returns ~0u
    534 static __forceinline__ __device__ unsigned int optixGetInstanceId();


optixGetInstanceIndex : returns 0-based index within the IAS
---------------------------------------------------------------

::

    536 /// Returns the zero-based index of the instance within its instance acceleration structure associated with the current intersection.
    537 ///
    538 /// In the Intersection and AH programs the index corresponding to the most recently visited OptixInstance is returned when calling optixGetInstanceIndex().
    539 /// In CH optixGetInstanceIndex() returns the index when the hit was recorded with optixReportIntersection.
    540 /// In the case where there is no OptixInstance visited, optixGetInstanceId returns 0
    541 static __forceinline__ __device__ unsigned int optixGetInstanceIndex();


TODO : compare optixGetInstanceId with optixGetInstanceIndex 
-------------------------------------------------------------

* currently I think they should be giving the same thing  
* if so : it means that there is a full 32 bits per instance going free (actually 31 bits as ~0u means not-an-instance)
* can use this for packed gas_idx/sensor_type/sensor_index without needing 
  to do a lookup into an identity array from the instance index 
* one downside is would need to occupy the last of the quad2 PRD slots 

DONE : added set_iindex to quad2 and machinery to populate it in CSGOptiX7.cu 


