where_mask_running
====================

hmm : how to fast forward to debug single photon ?
------------------------------------------------------

* on GPU almost no point, as it dont help much with debugging, 
  can just dump with pindex

* on CPU it would be useful 


where mask running
~~~~~~~~~~~~~~~~~~~~

Running on a subselection, picked via a where-mask of indices.
Apply mask to emitconfig photons, and to the rng inputs.

::

    161 CSource* CGenerator::initInputPhotonSource()
    162 {
    163     LOG(info) << "CGenerator::initInputPhotonSource " ;
    164     NPY<float>* inputPhotons = m_hub->getInputPhotons();
    165     NPY<float>* inputGensteps = m_hub->getInputGensteps();
    166     GenstepNPY* gsnpy = m_hub->getGenstepNPY();
    167 
    168     assert( inputPhotons );
    169     assert( inputGensteps );
    170     assert( gsnpy );
    171 
    172     setGensteps(inputGensteps);
    173     setDynamic(false);
    174 
    175     int verbosity = m_ok->isDbgSource() ? 10 : 0 ;
    176     CInputPhotonSource* cips = new CInputPhotonSource( m_ok, inputPhotons, gsnpy, verbosity) ;
    177 
    178     setNumG4Event( cips->getNumG4Event() );
    179     setNumPhotonsPerG4Event( cips->getNumPhotonsPerG4Event() );
    180 
    181     CSource* source  = static_cast<CSource*>(cips);
    182     return source ;
    183 }

::

    013 void CPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
     14 {
     15     m_source->GeneratePrimaryVertex(event);
     16 }


    153 void CInputPhotonSource::GeneratePrimaryVertex(G4Event *evt)
    154 {
    155     unsigned n = m_tranche->tranche_size(m_gpv_count) ;
    156     SetNumberOfParticles(n);
    157     assert( m_num == int(n) );
    158     for (G4int i = 0; i < m_num; i++)
    159     {
    160         unsigned pho_index = m_tranche->global_index( m_gpv_count,  i) ;
    161         G4PrimaryVertex* vertex = convertPhoton(pho_index);
    162         evt->AddPrimaryVertex(vertex);
    163         collectPrimary(vertex);
    164     }
    165     m_gpv_count++ ;
    166 }
          




