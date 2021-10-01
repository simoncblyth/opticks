
#include "NP.hh"
#include "SPath.hh"
#include "OPTICKS_LOG.hh"

#include "QTex.hh"
#include "QTexMaker.hh"
#include "QTexLookup.hh"

const char* FOLD = "$TMP/QTexLookupTest" ; 

NP* make_array(unsigned ni, unsigned nj, unsigned nk)
{
    std::vector<float> src ; 
    for(unsigned i=0 ; i < ni ; i++)
    for(unsigned j=0 ; j < nj ; j++)
    for(unsigned k=0 ; k < nk ; k++)
    {
        float val = float(i*100 + j*10 + k) ;  
        src.push_back(val); 
    }

    NP* a = NP::Make<float>( ni, nj, nk ); 
    a->read(src.data()) ; 
    a->set_meta<unsigned>("hd_factor", 10); 

    return a ; 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const NP* origin = make_array( 5, 10, 4 ); 
    char filterMode = 'P' ; 
    unsigned hd_factor = origin->get_meta<unsigned>("hd_factor"); 
    assert( hd_factor > 0 ); 

    QTex<float4>* tex = QTexMaker::Make2d_f4(origin, filterMode ); 
    assert(tex); 

    tex->setHDFactor(hd_factor); 
    tex->uploadMeta(); 

    QTexLookup<float4> look(tex); 
    NP* out = look.lookup(); 

    const char* fold = SPath::Resolve(FOLD); 
    int rc = SPath::MakeDirs(fold); 
    assert( rc == 0); 

    out->save(fold, "lookup.npy"); 
    origin->save(fold, "origin.npy"); 

    return 0 ; 
}



