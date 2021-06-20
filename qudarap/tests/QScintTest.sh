#!/bin/bash -l 

path=${BASH_SOURCE}
path=${path/.sh}.py 

ipython -i $path




