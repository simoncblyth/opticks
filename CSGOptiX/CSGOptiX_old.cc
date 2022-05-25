

void CSGOptiX::setComposition()   // TODO: replace with setFrame 
{
    setComposition(SSys::getenvvar("MOI", "-1")); 
}
void CSGOptiX::setComposition(const char* moi) // TODO: replace this with setFrame
{
    moi = moi_ ? strdup(moi_) : "-1" ;  

    int midx, mord, iidx ;  // mesh-index, mesh-ordinal, instance-index
    foundry->parseMOI(midx, mord, iidx,  moi );  

    float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 

    // m2w and w2m are initialized to identity, the below call may populate them 
    qat4* m2w = qat4::identity() ; 
    qat4* w2m = qat4::identity() ; 
    int rc = foundry->getCenterExtent(ce, midx, mord, iidx, m2w, w2m ) ;

    LOG(info) 
        << " moi " << moi 
        << " midx " << midx << " mord " << mord << " iidx " << iidx 
        << " rc [" << rc << "]" 
        << " ce (" << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ") " 
        << " m2w (" << *m2w << ")"    
        << " w2m (" << *w2m << ")"    
        ; 

    assert(rc==0); 
    setComposition(ce, m2w, w2m );   // establish the coordinate system 
}

void CSGOptiX::setComposition(const float4& v, const qat4* m2w, const qat4* w2m )
{
    glm::vec4 ce(v.x, v.y, v.z, v.w); 
    setComposition(ce, m2w, w2m ); 
}


/**

CSGOptiX::snapSimtraceTest
---------------------------

TODO: eliminate this, instead just use normal QEvent::save  

Saving data for 2D cross sections, used by tests/CSGOptiXSimtraceTest.cc 

The writing of pixels and frame photon "fphoton.npy" are commented 
here as they are currently not being filled by OptiX7Test.cu:simtrace
Although they could be reinstated the photons.npy array is more useful 
for debugging as that can be copied from remote to laptop enabling local analysis 
that gives flexible python "rendering" with tests/CSGOptiXSimtraceTest.py 

**/


void CSGOptiX::snapSimtraceTest() const    // only from CSGOptiXSimtraceTest.cc
{
    const char* outdir = SEventConfig::OutFold();

    event->setMeta( foundry->meta.c_str() ); 

    // HMM: QEvent rather than here ?  
    event->savePhoton( outdir, "photons.npy");   // this one can get very big 
    event->saveGenstep(outdir, "genstep.npy");  
    event->saveMeta(   outdir, "fdmeta.txt" ); 

    savePeta(          outdir, "peta.npy");   

    if(metatran) metatran->save(outdir, "metatran.npy");

}


void CSGOptiX::savePeta(const char* fold, const char* name) const  // TODO: ELIMINATE USING sframe TO HOLD THE INFO
{
    const char* path = SPath::Resolve(fold, name, FILEPATH) ; 
    LOG(info) << path ; 
    NP::Write(path, (float*)(&peta->q0.f.x), 1, 4, 4 );
}

void CSGOptiX::setMetaTran(const Tran<double>* metatran_ ) // only from CSGOptiXSimtraceTest.cc TODO: ELIMINATE USING sframe 
{
    metatran = metatran_ ; 
}


/**
CSGOptiX::setCEGS Center-Extent gensteps  TODO: ELIMINATE
-------------------------------------------------------------

From cegs vector into peta quad4 

HMM: seems peta is not uploaded ?  Just for python consumption, eg tests/CSGOptiXSimtraceTest.py 

**/



    /*
    peta->q2.f.x = ce.x ;   // moved from q1   
    peta->q2.f.y = ce.y ; 
    peta->q2.f.z = ce.z ; 
    peta->q2.f.w = ce.w ; 
    */



void CSGOptiX::setCEGS(const std::vector<int>& cegs)
{
    assert( cegs.size() == 7 );   // use QEvent::StandardizeCEGS to convert 4 to 7  

    peta->q0.i.x = cegs[0] ;  // ix0   these are after standardization
    peta->q0.i.y = cegs[1] ;  // ix1
    peta->q0.i.z = cegs[2] ;  // iy0 
    peta->q0.i.w = cegs[3] ;  // iy1

    peta->q1.i.x = cegs[4] ;  // iz0
    peta->q1.i.y = cegs[5] ;  // iz1 
    peta->q1.i.z = cegs[6] ;  // num_photons
    peta->q1.i.w = 0 ;     // TODO: gridscale according to ana/gridspec.py 
}


