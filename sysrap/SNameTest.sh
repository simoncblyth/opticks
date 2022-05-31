#!/bin/bash -l 
usage(){ cat << EOU

QTYPE=C SNameTest virtual 


Default match type is QTYPE=S meaning START::

    epsilon:sysrap blyth$ SNameTest NNVTMCPPMTsMask_virtual0x
    id.desc()
    SName::desc numName 141 name[0] sTopRock_domeAir0x578ca70 name[-1] uni10x5832ff0
     findIndex                                                        NNVTMCPPMTsMask_virtual0x count   1 idx 110
     findIndices                                                        NNVTMCPPMTsMask_virtual0x idxs.size   1 SName::QTypeLabel START
    descIndices
     110 : NNVTMCPPMTsMask_virtual0x5f5f900


QTYPE=C uses CONTAIN name matching::

    epsilon:sysrap blyth$ QTYPE=C SNameTest virtual 
    id.desc()
    SName::desc numName 141 name[0] sTopRock_domeAir0x578ca70 name[-1] uni10x5832ff0
     findIndex                                                                          virtual count   0 idx  -1
     findIndices                                                                          virtual idxs.size   3 SName::QTypeLabel CONTAIN
    descIndices
     110 : NNVTMCPPMTsMask_virtual0x5f5f900
     117 : HamamatsuR12860sMask_virtual0x5f50d40
     134 : mask_PMT_20inch_vetosMask_virtual0x5f62e40


QTYPE=E uses EXACT name matching::

    epsilon:sysrap blyth$ QTYPE=E SNameTest virtual
    id.desc()
    SName::desc numName 141 name[0] sTopRock_domeAir0x578ca70 name[-1] uni10x5832ff0
     findIndex                                                                          virtual count   0 idx  -1
     findIndices                                                                          virtual idxs.size   0 SName::QTypeLabel EXACT
    descIndices

    epsilon:sysrap blyth$ QTYPE=E SNameTest NNVTMCPPMTsMask_virtual0x5f5f900
    id.desc()
    SName::desc numName 141 name[0] sTopRock_domeAir0x578ca70 name[-1] uni10x5832ff0
     findIndex                                                 NNVTMCPPMTsMask_virtual0x5f5f900 count   1 idx 110
     findIndices                                                 NNVTMCPPMTsMask_virtual0x5f5f900 idxs.size   1 SName::QTypeLabel EXACT
    descIndices
     110 : NNVTMCPPMTsMask_virtual0x5f5f900

    epsilon:sysrap blyth$ QTYPE=E SNameTest NNVTMCPPMTsMask_virtual0x5f5f90
    id.desc()
    SName::desc numName 141 name[0] sTopRock_domeAir0x578ca70 name[-1] uni10x5832ff0
     findIndex                                                  NNVTMCPPMTsMask_virtual0x5f5f90 count   1 idx 110
     findIndices                                                  NNVTMCPPMTsMask_virtual0x5f5f90 idxs.size   0 SName::QTypeLabel EXACT
    descIndices


EOU
}


SNameTest $* 





