#!/bin/bash -l 

#export QBnd=INFO

test=fill_state
#test=water

export TEST=${TEST:-$test}

QSimTest 
