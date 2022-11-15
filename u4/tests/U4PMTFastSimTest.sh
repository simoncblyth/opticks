#!/bin/bash -l 

export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1

export GEOM=hamaLogicalPMT

U4PMTFastSimTest

