#!/bin/bash -l 

name=rsttable
${IPYTHON:-ipython} --pdb -i $name.py 
