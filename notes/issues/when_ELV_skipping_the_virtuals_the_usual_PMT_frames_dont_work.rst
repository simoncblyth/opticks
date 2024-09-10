when_ELV_skipping_the_virtuals_the_usual_PMT_frames_dont_work
==============================================================


This works::

    ELV=t:HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual,mask_PMT_20inch_vetosMask_virtual MOI=sTarget:0:-1 ~/o/cx.sh

   
This does too, must have been pilot error::

    ELV=filepath:/tmp/elv.txt MOI=sTarget:0:-1 ~/o/cx.sh

    P[blyth@localhost sysrap]$ grep \# /tmp/elv.txt 
    #HamamatsuR12860sMask_virtual
    #NNVTMCPPMTsMask_virtual
    #mask_PMT_20inch_vetosMask_virtual





