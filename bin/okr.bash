okr-source(){ echo $BASH_SOURCE ; }
okr-vi(){ vi $(okr-source) ; }
okr-env(){ echo -n ; }
okr-usage(){  cat << \EOU

Opticks Refs, thinking about reconstruction using GPUs
========================================================


Reconstruction Possibilities
------------------------------

~/opticks_refs/IceCube_Eike_Middell_diplom_final.pdf
    Reconstruction of Cascade-Like Events in IceCube

~/opticks_refs/Chroma-ProjectX-2099.pdf 
    "A sufficiently fast Monte Carlo is indistinguishable 
    from a maximum likelihood reconstruction algorithm."

    * But "sufficiently fast" 


~/opticks_refs/Stanley_Seibert_abstract_MWS_APR12-2012-000787.pdf
   
    * http://absimage.aps.org/image/APR12/MWS_APR12-2012-000787.pdf
    * http://meetings.aps.org/Meeting/APR12/Session/T11.5

    We demonstrate the feasibility of event reconstruction—including position,
    direction, en- ergy and particle identification—in water Cherenkov detectors
    with a purely Monte Carlo-based method. Using a fast optical Monte Carlo
    package we have written, called Chroma, in combination with several variance
    reduction techniques, we can estimate the value of a likelihood function for an
    arbitrary event hypothesis. **The likelihood can then be maximized over the
    parameter space of interest using a form of gradient descent designed for
    stochastic functions**. Although slower than more traditional reconstruction
    algorithms, this completely Monte Carlo-based technique is universal and can be
    applied to a detector of any size or shape, which is a major advantage during
    the design phase of an experiment. As a specific example, we focus on
    reconstruction results from a simulation of the 200 kiloton water Cherenkov far
    detector option for LBNE.

    * http://meetings.aps.org/Meeting/APR12/PersonIndex/2314


Monte Carlo-based Reconstruction in Water Cherenkov Detectors using Chroma
    * https://www.researchgate.net/publication/258593416_Monte_Carlo-based_Reconstruction_in_Water_Cherenkov_Detectors_using_Chroma


https://link.springer.com/content/pdf/10.1140%2Fepjc%2Fs10052-017-5380-x.pdf

    Cherenkov and scintillation light separation in organic liquid scintillators

https://arxiv.org/pdf/1610.02029.pdf

    Experiment to Demonstrate Separation of Cherenkov and Scintillation Signals





EOU
}





