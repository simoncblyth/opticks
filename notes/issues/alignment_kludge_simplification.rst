alignment_kludge_simplification
=================================

Overview
-----------

Current triple-whammy-kludge approach to getting 
perfect alignment is too complicated.  

::

    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero


Aligned bi-simulation
----------------------

To keep two simulations aligned it is necessary to arrange the "zipping" 
together of "code-locations" at which RNGs are consumed and the RNGs values
in a way that matches both locations and values between the two simulations.

At every "to_boundary" Opticks step (called bounce within Opticks) RNGs
for scattering and absorption are thrown and consumed in order to decide whether
to scatter/absorb/sail thru to the boundary. Standard Geant4 does not do this, 
it retains the G4VProcess interaction length RNG from step to step, and only 
throws again, in G4VDiscreteProcess::PostStepDoIt for the winning process. 

In order to make Geant4 RNG consumption follow a pattern closer to that 
of Opticks a devious ClearNumberOfInteractionLengthLeft is used at the end of every step
which clears the RNGs for scattering and absorption, forcing 
throwing of RNGs for every step.  

::

   CProcessManager::ClearNumberOfInteractionLengthLeft( m_ctx._process_manager, *m_ctx._track, *m_ctx._step );


Normally this every step clearing succeeds to make 
the pattern of RNG consumption the same for the two simulations. However certain
conditions involving Geant4 ZeroSteps break the alignment.


G4 Conditions that need special handling (each different) to keep the simulations aligned
-------------------------------------------------------------------------------------------

ZeroStep
~~~~~~~~~~~

ZeroStep are a common G4 condition, occuring for every reflection, that leads to mis-alignment
unless a jump back rewinding of the RNG sequence is done, such that the 
Geant4 decision gets "repeated" keeping matched to Opticks, despite burning the ZeroStep. 

::

    316 void CRandomEngine::postStep()
    317 {
    318     if(m_ctx._noZeroSteps > 0)
    319     {
    320         int backseq = -m_current_step_flat_count ;
    321         bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ;
    322 
    323         LOG(error) << "CRandomEngine::postStep"
    324                    << " _noZeroSteps " << m_ctx._noZeroSteps
    325                    << " backseq " << backseq
    326                    << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
    327                    ;
    328 
    329         if( dbgnojumpzero )
    330         {
    331             LOG(fatal) << "CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero " ;
    332         }
    333         else
    334         {
    335             jump(backseq);
    336         }
    337     }


ZeroStep immediately after FresnelReflection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ZeroStep following on directly from a FresnelReflection is a rare G4 condition, 
that requires different handling than the standard ZeroStep to retain alignment.

Q: is _prior_prior_boundary_status always StepTooSmall in this case ?


dbgnojumpzero 
    skip the rewind   

dbgskipclearzero
    skip clearing of the interactionlength RNG
     
dbgkludgeflatzero
    first "RNG" consumption following the rare condition yields a peek-back 
    value (actually this value is never used: so it could just be zero)
    and the sequence cursor stays unmoved.



::

    210 double CRandomEngine::flat()
    211 {
    212     if(!m_internal) m_location = CurrentProcessName();
    213     assert( m_current_record_flat_count < m_curand_nv );
    214 
    215     bool kludge = m_dbgkludgeflatzero
    216                && m_current_step_flat_count == 0
    217                && m_ctx._boundary_status == StepTooSmall
    218                && m_ctx._prior_boundary_status == FresnelReflection
    219                ;
    220 
    221     double v = kludge ? _peek(-2) : _flat() ;
    222 
    223     if( kludge )
    224     {
    225         LOG(info) << " --dbgkludgeflatzero  "
    226                   << " first flat call following boundary status StepTooSmall after FresnelReflection yields  _peek(-2) value "
    227                   << " v " << v
    228                  ;
    229         // actually the value does not matter, its just OpBoundary which is not used 
    230     }
    231 
    232     m_flat = v ;
    233     m_current_record_flat_count++ ;  // (*lldb*) flat 
    234     m_current_step_flat_count++ ;
    235 
    236     return m_flat ;
    237 }



Fixed u_boundary_burn simplification ?
-----------------------------------------

OpBoundary "u_boundary_burn" RNG is actually not used by Opticks OR Geant4,
as there is no length associated with the boundary process, thus 
perhaps ir would be simplify life to detect G4UniformRand from 
OpBoundary process and always provide a constant value (eg zero).

Suspect that taking RNG out of the loop for OpBoundary, ie always 
providing zero for OpBoundary interaction length RNG.

Difficulty is need to distinguish OpBoundary RNG calls for the 
interaction length from other meaningful ones.


  


