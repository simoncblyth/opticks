#!/bin/bash -l 

skipsolidname=NNVTMCPPMTsMask_virtual,HamamatsuR12860sMask_virtual,mask_PMT_20inch_vetosMask_virtual

OGeo=INFO OpticksDbg=INFO  FLIGHT=RoundaboutZX PERIOD=8 EMM=~5, PVN=lLowerChimney_phys flight.sh --rtx 1 --cvd 1 --skipsolidname $skipsolidname

