#!/usr/bin/env python

import glob 

fsu = set(glob.glob("Use*"))
doc = set(list(map(lambda _:_[:-1], filter(lambda _:len(_) > 3 and _.startswith("Use"), open("README.rst","r").readlines()))))

undocumented = fsu - doc   
lost = doc - fsu 

print("\nundocumented\n","\n".join(list(undocumented)))
print("\nlost\n","\n".join(list(lost)))




