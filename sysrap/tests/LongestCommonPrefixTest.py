#!/usr/bin/env python

def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
        pass
    pass
    return s1


if __name__ == '__main__':
    m = "o one/a one/x one/y one/z".split()
    p = commonprefix(m)
    print(p)


