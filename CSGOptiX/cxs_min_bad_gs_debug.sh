#!/usr/bin/env bash


#export QSim=INFO
#export QSim__SAVE_IGS_EVENTID=0
#export SEvt=INFO

#export TMP=/data1/blyth/tmp
#export TMP=/data2/blyth/tmp
#export TMP=/tmp/blyth/opticks
#mkdir -p $TMP
### MOVED THE CHANGE OF TMP TO LOWER LEVEL - USING VIP_LOCAL_TEMP

#export NPFold__save_DUMP=1
#export NP__save_DUMP=1


cxs_min.sh $*


