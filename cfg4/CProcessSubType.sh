#!/bin/bash -l 

g4-

ipython -i $(which enu.py) --  --hdr $(g4-dir)/source/processes/electromagnetic/utils/include/G4EmProcessSubType.hh --kls CEmProcessSubType
