#!/bin/bash -l 

export BOpticksResource=INFO


export GEOM=J004
source $(dirname $BASH_SOURCE)/../../bin/GEOM_.sh  
env | grep $GEOM

OpticksTest 

