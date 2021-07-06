#!/bin/bash 

pfx=tds3ip
#pfx=tds3gun

ipython --pdb -i evts.py -- --pfx $pfx --src natural 
