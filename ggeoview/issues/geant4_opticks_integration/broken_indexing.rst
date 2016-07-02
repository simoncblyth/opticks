Broken Indexing
================

Changes to opticksop-/OpIndexer for CG4 have broken Op indexing::

    delta:env blyth$ op 
    ...
    [2016-May-29 14:45:22.505882]:info: App::indexEvt WITH_OPTIX
    [2016-May-29 14:45:22.505967]:info: NumpyEvt::createHostIndexBuffers  flat true num_photons 500000 num_records 5000000 m_maxrec 10
    [2016-May-29 14:45:22.506132]:info: OpEngine::indexSequence proceeding  
    [2016-May-29 14:45:22.506245]:info: OpIndexer::indexSequenceInterop
    CUDA error at /Users/blyth/env/cuda/cudarap/CResource_.cu:40 code=11(cudaErrorInvalidValue) "cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags)" 
    delta:env blyth$ 

::

     37    void registerBuffer()
     38    {
     39        //printf("Resource::registerBuffer %d : %s \n", buffer_id, getFlagDescription() );
     40        checkCudaErrors( cudaGraphicsGLRegisterBuffer(&resource, buffer_id, flags) );
     41    }


Need to rearrange prep for indexing::

    1150 void App::indexSequence()
    1151 {
    1152     if(!m_ope)
    1153     {
    1154         LOG(warning) << "App::indexSequence NULL OpEngine " ;
    1155         return ;
    1156     }
    1157     
    1158     //m_evt->prepareForIndexing();  // stomps on prior recsel phosel buffers, causes CUDA error with Op indexing, but needed for G4 indexing  
    1159     
    1160     m_ope->indexSequence();
    1161     LOG(info) << "App::indexSequence DONE" ;
    1162 }   


