#!/usr/bin/env python
"""

Dump json structured like below, ordered by the int::

    {
        "ADVertiCableTraySurface": "25",
        "AdCableTraySurface": "12",
        "ESRAirSurfaceBot": "11",
        "ESRAirSurfaceTop": "10",
        "LegInDeadTubSurface": "44",
        "LegInIWSTubSurface": "19",
        ...

::

    epsilon:opticksdata blyth$ js.py export/DayaBay/GSurfaceLib/order.json
      1 : NearPoolCoverSurface 
      2 : NearDeadLinerSurface 
      3 : NearOWSLinerSurface 
      4 : NearIWSCurtainSurface 
      5 : SSTWaterSurfaceNear1 
      6 : SSTOilSurface 
      7 : lvPmtHemiCathodeSensorSurface 
      8 : lvHeadonPmtCathodeSensorSurface 
      9 : RSOilSurface 
     10 : ESRAirSurfaceTop 
     ...    


"""
import sys, json, os, logging
log = logging.getLogger(__name__)

js = json.load(file(sys.argv[1]))
print "\n".join(["%3s : %s " % ( kv[1], kv[0]) for kv in sorted(js.items(), key=lambda kv:int(kv[1]))])




