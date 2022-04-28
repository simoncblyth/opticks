/**
CXRaindropTest
======================

**/

#include <cuda_runtime.h>
#include <algorithm>
#include <csignal>
#include <iterator>

#include "scuda.h"
#include "sqat4.h"
#include "stran.h"
#include "NP.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "SProc.hh"
#include "OpticksGenstep.h"

#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "SOpticksResource.hh"

#include "CSGFoundry.h"
#include "CSGGenstep.h"
#include "CSGOptiX.h"
#include "RG.h"

#include "QBnd.hh"
#include "QSim.hh"
#include "QEvent.hh"
#include "SEvent.hh"


const char* OutDir(const Opticks& ok, const char* cfbase, const char* name)
{
    int optix_version_override = CSGOptiX::_OPTIX_VERSION(); 
    const char* out_prefix = ok.getOutPrefix(optix_version_override);   
    // out_prefix includes values of envvars OPTICKS_GEOM and OPTICKS_RELDIR when defined
    const char* default_outdir = SPath::Resolve(cfbase, name, out_prefix, DIRPATH );  
    const char* outdir = SSys::getenvvar("OPTICKS_OUTDIR", default_outdir );  

    LOG(info) 
        << " optix_version_override " << optix_version_override
        << " out_prefix [" << out_prefix << "]" 
        << " cfbase " << cfbase 
        << " default_outdir " << default_outdir
        << " outdir " << outdir
        ;

    return outdir ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv ); 
    ok.configure(); 
    ok.setRaygenMode(RG_SIMULATE) ; // override --raygenmode option 

    const char* EXECUTABLE = SProc::ExecutableName();
    const char* cfbase = SOpticksResource::CFBase(); 
    const char* outdir = OutDir(ok, cfbase, EXECUTABLE );    
    ok.setOutDir(outdir); 
    ok.writeOutputDirScript(outdir) ; // writes CSGOptiXSimulateTest_OUTPUT_DIR.sh in PWD 
    // TODO: relocate script writing and dir mechanices into SOpticksResource or SOpticks or similar 
    ok.dumpArgv(EXECUTABLE); 

    // HMM: note use of the standard OPTICKS_KEY geocache , TODO: add for CF?
    const char* idpath = ok.getIdPath(); 
    const char* rindexpath = SPath::Resolve(idpath, "GScintillatorLib/LS_ori/RINDEX.npy", 0 );  


    // BASIS GEOMETRY : LOADED IN ORDER TO RAID ITS BOUNDARIES FOR MATERIALS,SURFACES

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    if(fd->hasMeta()) LOG(info) << "fd.meta\n" << fd->meta ; 
    LOG(info) << fd->descComp(); 

    // fabricate some additional boundaries from the basis ones for this simple test 

    std::vector<std::string> specs = { "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Air///Water" } ;  
    NP* optical_plus = nullptr ; 
    NP* bnd_plus = nullptr ; 
    QBnd::Add( &optical_plus, &bnd_plus, fd->optical, fd->bnd, specs );
    LOG(info) << std::endl << QBnd::DescOptical(optical_plus, bnd_plus)  ; 

    // qudarap does not depend on CSG so cannot consoldate (eg uploading "fd") here unless
    // rearrange to uploading a string, NP map : or do some protocal base gymnastics 

    QSim<float>::UploadComponents(fd->icdf, bnd_plus, optical_plus, rindexpath ); 


    const char* cfbase_local = SSys::getenvvar("CFBASE_LOCAL") ; assert(cfbase_local) ; 
    LOG(fatal) << "MIXING CSGFoundry combining basis cfbase with cfbase_local "; 
    std::cout << std::setw(20) << "cfbase" << ":" << cfbase << std::endl ; 
    std::cout << std::setw(20) << "cfbase_local" << ":" << cfbase_local  << std::endl ; 


    // load raindrop geometry and customize to use the boundaries added above 

    CSGFoundry* fdl = CSGFoundry::Load(cfbase_local, "CSGFoundry") ; 
    fdl->setOpticalBnd(optical_plus, bnd_plus);    // instanciates bd CSGName using bndplus.names
    fdl->saveOpticalBnd();                         // DIRTY: persisting new bnd and optical into the source directory : ONLY APPROPRIATE IN SMALL TESTS
    fdl->setPrimBoundary( 0, specs[0].c_str() ); 
    fdl->setPrimBoundary( 1, specs[1].c_str() );   // HMM: notice these boundary changes are not persisted 
    std::cout << "fdl.detailPrim " << std::endl << fdl->detailPrim() ; 
    fdl->upload(); 


    float4 ce = make_float4( 0.f, 0.f, 0.f, 100.f );   // TODO: this should come from the geometry 


    // HMM : WOULD BE BETTER FOR CONSISTENCY TO HAVE SINGLE UPLOAD API

    CSGOptiX cx(&ok, fdl );   // QSim QEvent instanciated here 
    cx.setComposition(ce); 

    QEvent* event = cx.event ; 
    event->setGensteps(SEvent::MakeTorchGensteps());     

    cx.simulate();  
    cudaDeviceSynchronize(); 

    const char* odir = SPath::Resolve(cfbase_local, EXECUTABLE, DIRPATH ); 


    // TODO: consolidated saving using QEvent method with standardized naming 
    // and using the max
    
    NP* p = event->getPhotons() ; 
    NP* f = event->getRecords() ; 
    NP* r = event->getRec() ;
 
    LOG(info) << " p " << ( p ? p->sstr() : "-" ) ; 
    LOG(info) << " f " << ( f ? f->sstr() : "-" ) ; 
    LOG(info) << " r " << ( r ? r->sstr() : "-" ) ; 
    LOG(info) << " odir " << odir ; 

    if(p) p->save(odir, "p.npy"); 
    if(f) f->save(odir, "f.npy"); 
    if(r) r->save(odir, "r.npy"); 
 

    return 0 ; 
}
