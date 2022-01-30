#!/bin/bash -l 

export X4SolidMaker_SphereWithThetaSegment_theta_start=0.25    # inputs are multiples of pi 
export X4SolidMaker_SphereWithThetaSegment_theta_delta=0.50

export THIRDLINE="theta_start $X4SolidMaker_SphereWithThetaSegment_theta_start theta_delta $X4SolidMaker_SphereWithThetaSegment_theta_delta "

## theta_start:0    theta_delta:0.25    upwards 90 degree fan centered on +ve Z-axis
## theta_start:0.25 theta_delta:0.25    bow-tie above the z=0 plane
## theta_start:0.5  theta_delta:0.25    bow-tie under the z=0 plane
## theta_start:0.75 theta_delta:0.25    downwards 90 degree fan centered on -ve Z-axis
## theta_start:1    theta_delta:0.25    some kinda mess : just a radial line 


