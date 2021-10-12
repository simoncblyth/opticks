#include "NP.hh"
#include "QTex.hh"
#include "QTexMaker.hh"
#include "OPTICKS_LOG.hh"

//struct QTex<float4> ; 


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

    const NP* a = make_array( 5, 10, 4 ); 
    char filterMode = 'P' ; 
    unsigned hd_factor = a->get_meta<unsigned>("hd_factor"); 
    assert( hd_factor > 0 ); 

    QTex<float4>* tex = QTexMaker::Make2d_f4(a, filterMode ); 
    assert(tex); 

    tex->setHDFactor(hd_factor); 
    tex->uploadMeta(); 

    // TODO: lookup checks on the GPU texture, like QScint 


    return 0 ; 
}


