#!/usr/bin/env python
"""

::

    simon:pmt blyth$ ./polyconfig.py 
                 DEFAULT : {'verbosity': '0', 'resolution': '40', 'poly': 'IM'} 
               CONTAINER : {'containerscale': '4', 'container': '1', 'verbosity': '0', 'resolution': '40', 'poly': 'IM'} 
               lvPmtHemi : {'verbosity': '0', 'resolution': '40', 'poly': 'IM'} 
         lvPmtHemiVacuum : {'verbosity': '0', 'resolution': '40', 'poly': 'IM'} 
        lvPmtHemiCathode : {'threshold': '1', 'verbosity': '0', 'nominal': '7', 'coarse': '6', 'poly': 'DCS'} 
         lvPmtHemiBottom : {'threshold': '1', 'verbosity': '0', 'nominal': '7', 'coarse': '6', 'poly': 'DCS'} 
         lvPmtHemiDynode : {'verbosity': '3', 'resolution': '40', 'poly': 'IM'} 


"""

DEFAULT = "DEFAULT"
CONTAINER = "CONTAINER"
PYREX = "lvPmtHemi"    
VACUUM = "lvPmtHemiVacuum"   
CATHODE = "lvPmtHemiCathode"
BOTTOM = "lvPmtHemiBottom"
DYNODE = "lvPmtHemiDynode"

ALL = [DEFAULT, CONTAINER, PYREX, VACUUM, CATHODE, BOTTOM, DYNODE ]


class PolyConfig(object):
    """
    Common location for volume specific polygonization settings
    to avoid duplication between the GDML and detdesc branches.
    """

    def __init__(self, lvn):
        self.lvn = lvn 

    def _get(self, d):
        return d.get(self.lvn, d[DEFAULT])  


    # general settings

    _verbosity = {
          DYNODE:"0",
          DEFAULT:"0"
    }

    _poly = {
     #    CATHODE:"DCS",
     #    BOTTOM:"DCS",
         CONTAINER:"IM",
         DEFAULT:"IM"
    }

    verbosity = property(lambda self:self._get(self._verbosity))
    poly = property(lambda self:self._get(self._poly))


    # im settings

    _seeds = {
        CATHODE: "0,0,127.9,0,0,1",
         BOTTOM:"0,0,0,0,0,-1", 
         DEFAULT:None
     }

    _resolution = {
          BOTTOM:"150",
         DEFAULT:"40",
    }

    seeds = property(lambda self:self._get(self._seeds))
    resolution = property(lambda self:self._get(self._resolution))


    # dcs settings

    _nominal = {
         DEFAULT:"7",
    }
    _coarse = {
         DEFAULT:"6",
    }
    _threshold = {
         DEFAULT:"1",
    }

    nominal = property(lambda self:self._get(self._nominal))
    coarse = property(lambda self:self._get(self._coarse))
    threshold = property(lambda self:self._get(self._threshold))


    # mc settings

    _nx = {
         DEFAULT:"30"
    } 

    nx = property(lambda self:self._get(self._nx))

    # container settings
    #
    #      container="1" meta causes the NCSG deserialization 
    #      to adjust box size and position to contain contents (ie prior trees)
    #

    _container = dict(poly="IM",resolution="40", container="1", containerscale="4")

    def _get_meta(self):
        d = dict(verbosity=self.verbosity, poly=self.poly)
        if self.lvn == CONTAINER:
            d.update(self._container) 
        else:
            if d['poly'] == "IM":
                d.update(resolution=self.resolution) 
                seeds = self.seeds
                if seeds is not None:d.update(seeds=seeds)
            elif d['poly'] == "DCS":
                d.update(nominal=self.nominal, coarse=self.coarse, threshold=self.threshold) 
            elif d['poly'] == "MC":
                d.update(nx=self.nx)
            else:
                assert 0, d 
            pass
        pass
        return d
    meta = property(_get_meta)

    def __repr__(self):
        return "%20s : %s " % (self.lvn, self.meta )



if __name__ == '__main__':

    for lvn in ALL:
        pc = PolyConfig(lvn)
        print pc





