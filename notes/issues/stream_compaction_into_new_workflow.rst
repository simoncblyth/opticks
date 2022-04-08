stream_compaction_into_new_workflow
=======================================

* basis is thrust::count_if thrust::copy_if
* old workflow implemented with a smorgasbord : OEvent, TBuf, TIsHit, CBufSpec, NPY
* new workflow : need focussed implementation using NP and quadrap/QU and keeping thrust hidden 


* https://github.com/NVIDIA/thrust
* https://forums.developer.nvidia.com/t/using-thrust-copy-if-with-a-parameter/119735



::

    686 unsigned OEvent::downloadHitsCompute(OpticksEvent* evt)
    687 {
    688     OK_PROFILE("_OEvent::downloadHitsCompute");
    689 
    690     NPY<float>* hit = evt->getHitData();
    691     LOG(LEVEL) << "into hit array :" << hit->getShapeString();
    692     CBufSpec cpho = m_photon_buf->bufspec();
    693     assert( cpho.size % 4 == 0 );
    694     cpho.size /= 4 ;    //  decrease size by factor of 4, increases cpho "item" from 1*float4 to 4*float4 
    695 
    696     bool verbose = m_dbghit ;
    697     TBuf tpho("tpho", cpho );
    698     unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, m_hitmask, verbose);
    699     // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    700     assert(hit->hasShape(nhit,4,4));
    701 
    702     OK_PROFILE("OEvent::downloadHitsCompute");
    703 
    704     LOG(LEVEL)
    705          << " nhit " << nhit
    706          << " hit " << hit->getShapeString()
    707          ;
    708 
    709     if(m_ok->isDumpHit())
    710     {
    711         unsigned maxDump = 100 ;
    712         NPho::Dump(hit, maxDump, "OEvent::downloadHitsCompute --dumphit,post,flgs" );
    713     }
    714 
    715     return nhit ;
    716 }


TIsHit.hh::

     52 struct TIsHit4x4 : public thrust::unary_function<float4x4,bool>
     53 {
     54     unsigned hitmask ;
     55 
     56     TIsHit4x4(unsigned hitmask_) : hitmask(hitmask_) {}
     57 
     58     __host__ __device__
     59     bool operator()(float4x4 photon)
     60     {
     61         tquad q3 ;
     62         q3.f = photon.q3 ;
     63         return ( q3.u.w & hitmask ) == hitmask ;
     64     }
     65 };
     66 



thrap/TBuf_.cu::

    213 unsigned TBuf::downloadSelection4x4(const char* name, NPY<float>* npy, unsigned hitmask, bool verbose) const
    214 {
    215     return downloadSelection<float4x4>(name, npy, hitmask, verbose);
    216 }
    217 

    259 template <typename T>
    260 unsigned TBuf::downloadSelection(const char* name, NPY<float>* selection, unsigned hitmask, bool verbose) const
    261 {
    ...


