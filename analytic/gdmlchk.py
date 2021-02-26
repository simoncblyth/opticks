#!/usr/bin/env python
"""
gdmlchk.py
===========

::

    ipython --pdb -i gdmlchk.py 

"""
import os, sys, logging
from collections import OrderedDict as odict 
from fnmatch import fnmatch

log = logging.getLogger(__name__)

import numpy as np
import lxml.etree as ET
import lxml.html as HT

#np.seterr(all='raise')

try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None
pass


tostring_ = lambda _:ET.tostring(_)
exists_ = lambda _:os.path.exists(os.path.expandvars(_))
parse_ = lambda _:ET.parse(os.path.expandvars(_)).getroot()
fparse_ = lambda _:HT.fragments_fromstring(file(os.path.expandvars(_)).read())
pp_ = lambda d:"\n".join([" %30s : %f " % (k,d[k]) for k in sorted(d.keys())])
values_ = lambda _:np.fromstring(_, sep=" ")




class P(object):
    def __init__(self, property_, m):
        name = property_.attrib["name"]
        ref = property_.attrib["ref"]
        self.e = property_
        self.name = name
        self.ref = ref
        self.m = m 
        mname = m.name
        mname = mname[:mname.find("0x")]
        self.mname = mname
        self.title = "%s.%s" % (self.mname,self.name)

    def __str__(self):
        return "%30s :  %s" % (self.title, repr(self)) 

    def __repr__(self):
        return "    P %30s : %30s : %s  " % ( self.name, self.ref, self.m.name  )

class M(object):
    def __init__(self, material):
        name = material.attrib["name"]
        self.name = name
        pp = odict()
        for property_ in material.xpath(".//property"):
            p = P(property_, self)
            pp[p.name] = p 
        pass
        self.e = material
        self.pp = pp 

    def find_properties_with_ref(self, ref):
        return list(filter(lambda p:p.ref == ref, self.pp.values()))

    def num_properties_with__ref(self, ref):
        return len(self.find_properties_with_ref(ref))

    def __repr__(self):
        hdr = "M %30s : %d " % (self.name, len(self.pp))
        return "\n".join([hdr] + list(map(str,self.pp.values())) + [""])


class MM(object):
    def __init__(self, material_elements):
        """
        :param material_elements: from lxml parsed GDML tree
        """
        d = odict()
        for material in material_elements:
            m = M(material)
            d[m.name] = m
        pass
        self.d = d

    def find_materials_with_matrix_ref(self, matrix_ref):
        found = list(filter( lambda m:m.num_matrix_ref(matrix_ref) > 0, self.d.values()))
        return found

    def find_material_properties_with_matrix_ref(self, matrix_ref):
        mp = []
        for m in self.d.values():
            pp = m.find_properties_with_ref(matrix_ref)
            mp.extend(pp)
        pass
        return mp

    def __repr__(self):
        return "\n".join(map(str,self.d.values()))

    def __call__(self, arg):
        """
        :param arg: integer or name 
        :return m: M object  
        """
        try:
            iarg = int(arg)
        except ValueError:
            iarg = None
        pass
        items = self.d.items()
        values = list(map(lambda kq:kq[1], items))
        return self.d[arg] if iarg is None else values[iarg] 




class Q(object):
    """
    Using approx hc eV to nm conversion of 1240 as it seems that was done upstream, 
    hence using this approx value will actually be better as it should 
    give the measurement nm from LS group.

    Rather than using the more precise 1239.8418754199977 which 
    will give nm slightly off 
    """
    headings = tuple(["name", "ptitle", "shape", "eV[0]", "eV[-1]", "nm[0]", "nm[-1]", "len(vals)", "msg" ])
    fmt = "%40s : %40s : %10s : %10s : %10s : %10s : %10s : %10s : %s "  

    #hc_eVnm=1239.8418754199977  # h_Planck*c_light*1e12    
    hc_eVnm=1240.0 

    def __init__(self, name, v, sv, msg, mm):

        self.name = name  
        self.v = v 
        self.sv = sv 
        self.mm = mm 
        self.len = len(v)

        vals = v[:,1]
        mev = v[:,0]      # CLHEP SystemOfUnits megaelectronvolt = 1.  g4-cls SystemOfUnits  
        ev = 1.e6*v[:,0]    
        is_optical = ev.min() >= 1.0 and ev.max() <= 20.0    # reasonal energy range for optical property 
        is_zero_in_domain = np.any(ev == 0.) 

        if is_optical: msg += " optical, "
        if is_zero_in_domain: msg += " zero_in_domain, "

        if not is_zero_in_domain:
            nm = self.hc_eVnm/ev
        else:
            nm = np.zeros(len(v), dtype=v.dtype)
        pass

        a = np.zeros( v.shape, dtype=v.dtype )
        a[:,0] = nm[::-1]
        a[:,1] = vals[::-1]
        self.a = a
 
        self.msg = msg 
        self.mev = mev
        self.ev = ev
        self.nm = nm
        self.is_optical = is_optical
        self.is_zero_in_domain = is_zero_in_domain
        self.property = self.get_property(self.name)
        self.ptitle = self.property.title if not self.property is None else "-"

    def get_property(self, matrix_ref):
        props = self.mm.find_material_properties_with_matrix_ref(matrix_ref)
        assert len(props) < 2 
        return props[0] if len(props) == 1 else None

    def row(self):
        fmt_ = lambda f:"%10.4f" % f  
        return tuple([self.name, self.ptitle, str(self.v.shape), fmt_(self.ev[0]), fmt_(self.ev[-1]), fmt_(self.nm[0]), fmt_(self.nm[-1]), len(self.sv),self.msg])

    @classmethod
    def hdr(cls):
        return cls.fmt % cls.headings

    def __str__(self):
        return self.fmt % self.row()

    def __repr__(self):
        return self.fmt % self.row()

    def plot(self):
        fig, ax = plt.subplots(1)   
        a = self.a
        x = a[:,0]
        y = a[:,1]
        ax.plot(x, y)
        ax.text(x.max(),y.max(), self.ptitle, ha='right' )  
        print(self)
        print(a.shape)
        print(a)
        fig.show()
        return ax

 

class QQ(object):
    def __init__(self, matrix_elements, mm):
        """
        :param g: lxml parsed GDML tree
        """
        self.mm = mm
        d = odict()
        for matrix in matrix_elements:
            q = self.parse_matrix(matrix)
            d[q.name] = q
        pass
        self.d = d
        self._select_keys = None

    def parse_matrix(self, matrix):
        coldim = matrix.attrib["coldim"]
        assert coldim == "2"
        name = matrix.attrib["name"]
        #name = name[:name.find("0x")]

        sv = matrix.attrib["values"]
        v = values_(sv)

        if len(v) % 2 == 0:
            msg = ""
        else:
            v = v[:-1]
            msg = "truncated+trimmed?"
        pass
        v = v.reshape(-1,2)
        q = Q(name, v, sv, msg, self.mm)
        return q 

    def __call__(self, arg):
        """
        :param arg: integer or name 
        :return q: Q object  
        """
        try:
            iarg = int(arg)
        except ValueError:
            iarg = None
        pass
        items = self.sorted_items() 
        values = list(map(lambda kq:kq[1], items))
        return self.d[arg] if iarg is None else values[iarg] 

    def unselect(self):
        self._select_keys = None

    def select(self, keys_include=[], keys_exclude=[], optical=True):
        """
        :param keys_include: list of fnmatch patterns to include from key selection
        :param keys_exclude: list of fnmatch patterns to exclude from key selection
        :param optical: when true requires properties to have energy in optical range 1-20 eV
        :return select_keys_: list of selected keys

        Has side effect of setting self._select_keys which inflences
        other presentation of contents.  
        """
        filter_q_optical = lambda q: q.is_optical == optical    
        filter_k_include = lambda k: any(fnmatch(k, include_pattern) for include_pattern in keys_include)
        filter_k_exclude = lambda k: any(fnmatch(k, exclude_pattern) for exclude_pattern in keys_exclude)
        filter_func = lambda kq:filter_q_optical(kq[1]) and filter_k_include(kq[0]) and not filter_k_exclude(kq[0]) 

        items = self.sorted_items() 
        _select_items = list(filter(filter_func, items))
        _select_keys  = list(map(lambda kq:kq[0], _select_items))
        self._select_keys = _select_keys
        return self._select_keys

    def num_select(self):
        return len(self._select_keys)

    def nm_min_(self):
        items = self.selected_items()
        return np.array(list(map(lambda kq:kq[1].nm.min(), items)))
    def nm_min(self):
        return self.nm_min_().min()
    def nm_max_(self):
        items = self.selected_items()
        return np.array(list(map(lambda kq:kq[1].nm.max(), items)))
    def nm_max(self):
        return self.nm_max_().max()

    def find(self, arg):
        """
        :param arg: eg RINDEX
        :return qq: list of all Q objects starting with arg
        """
        items = self.sorted_items() 
        keys   = list(map(lambda kq:kq[0], items))
        ksel = filter(lambda k:k.startswith(arg), keys)
        return list(map(self, ksel))

    def sorted_items(self):
        items = list(self.d.items())
        items = list(sorted(items,key=lambda kq:kq[1].len, reverse=True))
        return items

    def selected_items(self):
        items = self.sorted_items()
        if not self._select_keys is None:
            items = list(filter(lambda kq:kq[0] in self._select_keys, items))
        pass
        return items 

    def __repr__(self):
        hdr =  Q.hdr()
        items = self.selected_items() 
        values = list(map(lambda kq:kq[1], items))
        nm_min = "nm_min:"+str(self.nm_min())
        nm_max = "nm_max:"+str(self.nm_max())
        return "\n".join(map(str,[hdr]+values+[hdr,nm_min,nm_max]))



    def plot(self):
        num = self.num_select()
        print("num:%s" % num)
        print("\n".join(self._select_keys))
        fig, axs = plt.subplots(num, sharex=True)   

        nm_max = self.nm_max()

        for i in range(num):
            q = self(i)
            a = q.a
            x = a[:,0]
            y = a[:,1]
            ymid = (y.min() + y.max())/2.
            ax = axs[i]
            ax.plot(x, y)
            ax.text(nm_max,ymid, q.ptitle, ha='right' )  
        pass
        fig.show()



if __name__ == '__main__':

   np.set_printoptions(suppress=True)

   #path = sys.argv[1]
   path = "/usr/local/opticks/opticksaux/export/juno2102/tds_ngt_pcnk_sycg_202102_v0.gdml"
   g = parse_(path) 

   mm = MM(g.xpath("//material"))
   qq = QQ(g.xpath("//matrix"), mm)

   print("all optical properties")
   qq.select(keys_include=["*"],optical=True)  
   print(qq)

   print("all properties without selection")
   qq.unselect() 
   print(qq)

   wild = "*ABSLENGTH*"
   print("all optical properties matching wildcard %s " % wild )
   qq.select(keys_include=[wild], optical=True)  
   print(qq)

   qq.plot() 

   #q = qq.find("PPOABSLENGTH")[0]
   #a = q.a



