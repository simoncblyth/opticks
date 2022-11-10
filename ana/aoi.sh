#!/bin/bash -l 

name=aoi 

${IPYTHON:-ipython} --pdb -i $name.py 
