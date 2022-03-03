#!/bin/bash -l 

export X4PhysicalVolume=INFO
export GInstancer=INFO

# comment the below to create an all global geometry, uncomment to instance the PMT volume 
#export GInstancer_instance_repeat_min=25

G4OKPMTSimTest



