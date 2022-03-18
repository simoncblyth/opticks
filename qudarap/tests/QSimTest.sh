#!/bin/bash -l 


test=fill_state
#test=water

export TEST=${TEST:-$test}

QSimTest 
