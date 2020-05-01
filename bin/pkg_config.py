#!/usr/bin/env python
"""
pkg_config.py
=================

This is to check that pkg-config finds packages 
in the expected manner.  See bin/oc.bash 


"""
from findpkg import Main

if __name__ == '__main__':
    Main(default_mode="pc")
  

