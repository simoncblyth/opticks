okg4-with-nog4-option
=========================


Context
-----------

Preferable to use the same executable but disable G4 running 
for high photon count tests of GPU side changes such as RTX mode
which has no effect on G4 : so dont want to repeat G4 propagaation, 
and pay the heavy price.


Added "--nog4propagate" but unclear how to use in OKG4Mgr 
----------------------------------------------------------- 

* took the easiest route : to still boot g4 just skipping m_g4->propagate()

* considerable changes to OpticksRun were needed to cope without m_g4evt

* also the ana/profile.py needed a rewrite to cope with getting info from two tagdirs 



::

    185 void OKG4Mgr::propagate_()
    186 {
    187     bool align = m_ok->isAlign();
    188 
    189     if(m_generator && m_generator->hasGensteps())   // TORCH
    190     {
    191          NPY<float>* gs = m_generator->getGensteps() ;
    192          m_run->setGensteps(gs);
    193 
    194          if(align)
    195              m_propagator->propagate();
    196 
    197 
    198          m_g4->propagate();
    199     }
    200     else   // no-gensteps : G4GUN or PRIMARYSOURCE
    201     {
    202          NPY<float>* gs = m_g4->propagate() ;
    203 
    204          if(!gs) LOG(fatal) << "CG4::propagate failed to return gensteps" ;
    205          assert(gs);
    206 
    207          m_run->setGensteps(gs);
    208     }
    209 
    210     if(!align)
    211         m_propagator->propagate();
    212 }



