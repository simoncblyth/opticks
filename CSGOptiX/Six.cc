#include <iostream>

#include "PLOG.hh"
#include "SSys.hh"
#include "SStr.hh"
#include "SPath.hh"
#include "NP.hh"
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "OpticksCSG.h"

#include "Opticks.hh"
// TODO: replace use of Opticks here (solid selection, emm)

#include "Params.h"

#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"

#include "Frame.h"
#include "Six.h"

    
Six::Six(const Opticks* ok_, const char* ptx_path_, const char* geo_ptx_path_, Params* params_)
    :
    ok(ok_), 
    solid_selection(ok->getSolidSelection()),
    emm(ok->getEMM()),
    context(optix::Context::create()),
    material(context->createMaterial()),

    d_pixel(nullptr),
    d_isect(nullptr),
    d_photon(nullptr),
 
    params(params_),
    ptx_path(strdup(ptx_path_)),
    geo_ptx_path(strdup(geo_ptx_path_)),
    entry_point_index(0u),
    optix_device_ordinal(0u),
    foundry(nullptr),
    pindex(SSys::getenvint("PINDEX", 0))
{
    initContext();
    initPipeline(); 
    initFrame(); 
    updateContext(); 
}

void Six::initContext()
{
    LOG(info) << " pindex " << pindex ; 
    context->setRayTypeCount(1);
    context->setPrintEnabled(true);
    context->setPrintBufferSize(1024);
    context->setPrintLaunchIndex(pindex); 
    context->setEntryPointCount(1);
}

void Six::initPipeline()
{
    LOG(info); 
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path , "raygen" ));
    context->setMissProgram(          entry_point_index, context->createProgramFromPTXFile( ptx_path , "miss" ));
    material->setClosestHitProgram(   entry_point_index, context->createProgramFromPTXFile( ptx_path , "closest_hit" ));
}

void Six::initFrame()
{
    pixel_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, params->width, params->height);
    context["pixel_buffer"]->set( pixel_buffer );  // formerly pixels_buffer
    d_pixel = (uchar4*)pixel_buffer->getDevicePointer(optix_device_ordinal); 

    isect_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, params->width, params->height);
    context["isect_buffer"]->set( isect_buffer );   // formerly posi_buffer
    d_isect = (float4*)isect_buffer->getDevicePointer(optix_device_ordinal); 

    photon_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_USER, params->width, params->height);
    photon_buffer->setElementSize( sizeof(quad4) ); 
    context["photon_buffer"]->set( photon_buffer );  // formerly isect_buffer
    d_photon = (quad4*)photon_buffer->getDevicePointer(optix_device_ordinal); 

    LOG(info) 
        << " params->width " << params->width
        << " params->height " << params->height
        << " optix_device_ordinal " << optix_device_ordinal
        ;
}

void Six::updateContext()
{
    LOG(info); 
    context[ "tmin"]->setFloat( params->tmin );  
    context[ "eye"]->setFloat( params->eye.x, params->eye.y, params->eye.z  );  
    context[ "U"  ]->setFloat( params->U.x, params->U.y, params->U.z  );  
    context[ "V"  ]->setFloat( params->V.x, params->V.y, params->V.z  );  
    context[ "W"  ]->setFloat( params->W.x, params->W.y, params->W.z  );  
    context[ "radiance_ray_type"   ]->setUint( 0u );  
    context[ "cameratype"   ]->setUint( params->cameratype );  
    context[ "raygenmode"   ]->setUint( params->raygenmode ); 
}

/**
Six::createContextInputBuffer
-------------------------------

Create shim optix::Buffer from CUDA buffer pointers 

The setDevicePointer fails for output buffers due to OptiX limitation::

    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: 
    Function "RTresult _rtBufferSetDevicePointer(RTbuffer, int, void *)" caught exception: 
    rtBufferSetPointer() not allowed for output or input/output buffers)

The optix-pdf (OptiX 5.0) p82 documents this limitation::

    Calling rtBufferSetDevicePointer on output or input/output buffers is not allowed.

Worked around this limitation by generalizing Frame to work with externally 
provided CUDA device pointers obtained from the real OptiX < 7 optix::Buffer  

**/

template<typename T>
void Six::createContextInputBuffer( T* d_ptr, unsigned num_item, const char* name )
{
    LOG(info) << name << " " << d_ptr << ( d_ptr == nullptr ? " EMPTY " : "" ); 

    // NB OptiX limitation the setDevicePointer throws exception for non-INPUT buffers
    unsigned type = RT_BUFFER_INPUT ; 
    RTformat format = RT_FORMAT_USER ; 
    optix::Buffer buffer = context->createBuffer( type, format, num_item );
    buffer->setElementSize( sizeof(T) ); 
    if(d_ptr)
    {
        buffer->setDevicePointer(optix_device_ordinal, d_ptr ); 
    }
    context[name]->set( buffer );
}

template void Six::createContextInputBuffer( CSGNode*,  unsigned, const char* ) ; 
template void Six::createContextInputBuffer( qat4*,     unsigned, const char* ) ; 
template void Six::createContextInputBuffer( float*,    unsigned, const char* ) ; 


void Six::setFoundry(const CSGFoundry* foundry_) 
{
    foundry = foundry_ ; 
    createGeom(); 
}
    
void Six::createGeom()
{
    LOG(info) << "[" ; 
    createContextInputBuffers(); 
    createGAS(); 
    createIAS(); 
    LOG(info) << "]" ; 
}


/**
Six::createContextInputBuffers : Shim optix::Buffer facade from CUDA device pointers
--------------------------------------------------------------------------------------

NB the CSGPrim prim_buffer is not here as that is specific 
to each geometry "solid"

These CSGNode float4 planes and qat4 inverse transforms are 
here because those are globally referenced.

This is equivalent to foundry upload with SBT.cc in OptiX 7 

NB no significant uploading is done here, 
the CSGFoundry buffers having already been uploaded to device.  

**/
void Six::createContextInputBuffers()
{
    createContextInputBuffer<CSGNode>( foundry->d_node, foundry->getNumNode(), "node_buffer" ); 
    createContextInputBuffer<qat4>(    foundry->d_itra, foundry->getNumItra(), "itra_buffer" ); 
    createContextInputBuffer<float4>(  foundry->d_plan, foundry->getNumPlan(), "plan_buffer" ); 
}


/**
Six::createGeometry
----------------------

Invoked from Six::createGAS_Standard for each compound solid index. 
Creates optix::Geometry corresponding to each CSGSolid referencing 
its CSGPrim via its *prim_buffer*  

**/

optix::Geometry Six::createGeometry(unsigned solid_idx)
{
    const CSGSolid* so = foundry->solid.data() + solid_idx ; 
    unsigned primOffset = so->primOffset ;  
    unsigned numPrim = so->numPrim ; 
    CSGPrim* d_pr = foundry->d_prim + primOffset ; 

    LOG(info) 
        << " solid_idx " << std::setw(3) << solid_idx
        << " numPrim " << std::setw(3) << numPrim 
        << " primOffset " << std::setw(3) << primOffset
        << " d_pr " << d_pr
        ;

    optix::Geometry solid = context->createGeometry();
    solid->setPrimitiveCount( numPrim );
    solid->setBoundingBoxProgram( context->createProgramFromPTXFile( geo_ptx_path , "bounds" ) );
    solid->setIntersectionProgram( context->createProgramFromPTXFile( geo_ptx_path , "intersect" ) ) ; 

    optix::Buffer prim_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER, numPrim );
    prim_buffer->setElementSize( sizeof(CSGPrim) ); 
    prim_buffer->setDevicePointer(optix_device_ordinal, d_pr ); 
    solid["prim_buffer"]->set( prim_buffer );
 
    return solid ; 
}

void Six::createGAS()
{
    if( solid_selection.size() == 0  )
    {
        createGAS_Standard();  
    }
    else
    {
        createGAS_Selection();  
    }
}

void Six::createGAS_Standard()
{
    unsigned num_solid = foundry->getNumSolid();   
    LOG(info) << "num_solid " << num_solid ;  

    for(unsigned i=0 ; i < num_solid ; i++)
    {
        if(ok->isEnabledMergedMesh(i))
        {
            LOG(info) << " create optix::Geometry solid/ mm " << i ; 
            optix::Geometry solid = createGeometry(i); 
            solids[i] = solid ;  
        }
        else
        {
            LOG(error) << " emm skipping " << i ; 
        }
    }
}

void Six::createGAS_Selection()
{
    unsigned num_solid = solid_selection.size() ;   
    LOG(info) << "num_solid " << num_solid ;  
    for(unsigned i=0 ; i < num_solid ; i++)
    {
        unsigned solidIdx = solid_selection[i] ; 
        optix::Geometry solid = createGeometry(solidIdx); 
        solids[solidIdx] = solid ;  
    }
}


optix::Geometry Six::getGeometry(unsigned solid_idx) const 
{
    unsigned count = solids.count(solid_idx); 
    assert( count <= 1 ) ; 
    if( count == 0 ) LOG(fatal) << " FAILED to find solid_idx " << solid_idx ; 
    return solids.at(solid_idx); 
}


void Six::createIAS()
{
    if( solid_selection.size() == 0 )
    {
        createIAS_Standard(); 
    }
    else
    {
        createIAS_Selection(); 
    }
}

void Six::createIAS_Standard()
{
    unsigned num_ias = foundry->getNumUniqueIAS() ; 
    for(unsigned i=0 ; i < num_ias ; i++)
    {
        unsigned ias_idx = foundry->ias[i]; 
        optix::Group ias = createIAS(ias_idx); 
        groups.push_back(ias); 
    }
}

void Six::createIAS_Selection()
{
    unsigned ias_idx = 0 ; 
    optix::Group ias = createSolidSelectionIAS( ias_idx, solid_selection ); 
    groups.push_back(ias); 
}

optix::Group Six::createIAS(unsigned ias_idx)
{
    unsigned tot_inst = foundry->getNumInst(); 
    unsigned num_inst = foundry->getNumInstancesIAS(ias_idx, emm); 
    LOG(info) 
        << " ias_idx " << ias_idx
        << " tot_inst " << tot_inst
        << " num_inst " << num_inst 
        << " emm(hex) " << std::hex << emm << std::dec
        ; 
    assert( num_inst > 0); 

    std::vector<qat4> inst ; 
    foundry->getInstanceTransformsIAS(inst, ias_idx, emm ); 
    assert( inst.size() == num_inst );  

    optix::Group ias = createIAS(inst); 
    return ias ; 
}

/**
Six::createSolidSelectionIAS
------------------------------

Non-standard "presentation" geometry.

**/

optix::Group Six::createSolidSelectionIAS(unsigned ias_idx, const std::vector<unsigned>& solid_selection)
{
    unsigned num_select = solid_selection.size() ; 
    assert( num_select > 0 ); 
    float mxe = foundry->getMaxExtent(solid_selection); 


    std::vector<qat4> inst ; 
    unsigned ins_idx = 0 ; 
    unsigned middle = num_select/2 ; 

    for(unsigned i=0 ; i < num_select ; i++)
    {
        unsigned gas_idx = solid_selection[i] ; 
        int ii = int(i) - int(middle) ; 

        qat4 q ; 
        q.setIdentity(ins_idx, gas_idx, ias_idx );
        q.q3.f.x = 2.0*mxe*float(ii) ;   

        inst.push_back(q); 
        ins_idx += 1 ; 
    }

    optix::Group ias = createIAS(inst); 
    return ias ; 
}



/**
Six::createIAS
---------------

group
    xform
        perxform
            pergi
    xform
        perxform
            pergi
    xform
        perxform
            pergi
    ...

**/

optix::Group Six::createIAS(const std::vector<qat4>& inst )
{
    unsigned num_inst = inst.size() ;
    LOG(info) << " num_inst " << num_inst ; 

    const char* accel = "Trbvh" ; 
    optix::Acceleration instance_accel = context->createAcceleration(accel);
    optix::Acceleration group_accel  = context->createAcceleration(accel);

    optix::Group group = context->createGroup();
    group->setChildCount( num_inst );
    group->setAcceleration( group_accel );  

    for(unsigned i=0 ; i < num_inst ; i++)
    {
        const qat4& qc = inst[i] ; 

        unsigned ins_idx,  gas_idx, ias_idx ;
        qc.getIdentity( ins_idx,  gas_idx, ias_idx ); 

        /*
        if( ins_idx != i )
        {
            LOG(info) 
                << " i " << i  
                << " ins_idx " << ins_idx  
                << " gas_idx " << gas_idx  
                << " ias_idx " << ias_idx  
                ;
        }        
        //assert( ins_idx == i );   when emm skipping this doesnt match, ins_idx didnt account for emm perhaps  
        */

        const float* qcf = qc.cdata(); 
        qat4 q(qcf);        // copy to clear identity before passing to OptiX
        q.clearIdentity(); 

        optix::Transform xform = context->createTransform();
        bool transpose = true ; 
        xform->setMatrix(transpose, q.data(), 0); 
        group->setChild(i, xform);

        // here referencing the GAS into the IAS with gas_idx
        optix::GeometryInstance pergi = createGeometryInstance(gas_idx, ins_idx); 
        optix::GeometryGroup perxform = context->createGeometryGroup();
        perxform->addChild(pergi); 
        perxform->setAcceleration(instance_accel) ; 
        xform->setChild(perxform);
    }
    return group ;
}

optix::GeometryInstance Six::createGeometryInstance(unsigned gas_idx, unsigned ins_idx)
{
    //LOG(info) << " gas_idx " << gas_idx << " ins_idx " << ins_idx  ;   
    optix::Geometry solid = getGeometry(gas_idx); 
    optix::GeometryInstance pergi = context->createGeometryInstance() ;
    pergi->setMaterialCount(1);
    pergi->setMaterial(0, material );
    pergi->setGeometry(solid);
    pergi["identity"]->setUint(ins_idx);
    return pergi ; 
}

void Six::setTop(const char* spec)
{
    char c = spec[0]; 
    assert( c == 'i' || c == 'g' );  
    int idx = atoi( spec + 1 );  

    LOG(info) << "spec " << spec ; 
    if( c == 'i' )
    {
        assert( idx < int(groups.size()) ); 
        optix::Group grp = groups[idx]; 
        context["top_object"]->set( grp );
    }
    else if( c == 'g' )
    {
        assert( idx < int(solids.size()) ); 

        optix::GeometryGroup gg = context->createGeometryGroup();
        gg->setChildCount(1);
     
        unsigned identity = 1u + idx ;  
        optix::GeometryInstance pergi = createGeometryInstance(idx, identity); 
        gg->setChild( 0, pergi );
        gg->setAcceleration( context->createAcceleration("Trbvh") );

        context["top_object"]->set( gg );
    }
}

void Six::launch(unsigned width, unsigned height, unsigned depth)
{
    LOG(info) << "[ width " << width << " height " << height << " depth " << depth  ; 
    context->launch( entry_point_index , width, height, depth );  
    LOG(info) << "]" ; 
}

