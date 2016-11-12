#!/usr/bin/env python
"""
Copied from env/doc/make_rst_table.py 


http://stackoverflow.com/questions/11347505/what-are-some-approaches-to-outputting-a-python-data-structure-to-restructuredte

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)


def make_rst_table(grid):
    max_cols = [max(out) for out in map(list, zip(*[[len(item) for item in row] for row in grid]))]
    rst = table_div(max_cols, 1)

    for i, row in enumerate(grid):
        header_flag = i == 0 or i == len(grid)-1
        rst += normalize_row(row,max_cols)
        if header_flag or row[0].strip()[-1] == "]":
            rst += table_div(max_cols, header_flag )
    return rst

def table_div(max_cols, header_flag=1):
    out = ""
    if header_flag == 1:
        style = "="
    else:
        style = "-"

    for max_col in max_cols:
        out += max_col * style + " "

    out += "\n"
    return out


def normalize_row(row, max_cols):
    """
    Padding to equalize cell string lengths 
    """
    r = ""
    for i, max_col in enumerate(max_cols):
        r += row[i] + (max_col  - len(row[i]) + 1) * " "

    return r + "\n"



def fmt(cell, prefix=""):
    if type(cell) is str or type(cell) is np.string_:
        s = str(cell) 
        if s == "key":
            return prefix  
        elif s.startswith(prefix):
            return s[len(prefix):]
        else:
            return s

    return "%5.2f" % cell
         

def recarray_as_rst(ra):
    """
    Expecting recarray with dtype of form: 

         dtype=[('key', 'S64'), ('X', '<f4'), ('Y', '<f4'), ('Z', '<f4'), ('T', '<f4'), ('A', '<f4'), ('B', '<f4'), ('C', '<f4'), ('R', '<f4')]

    ======================= ===== ===== ===== ===== ===== ===== ===== ===== 
    PmtInBox/torch          X     Y     Z     T     A     B     C     R     
    ======================= ===== ===== ===== ===== ===== ===== ===== ===== 
    [TO] BT SA               1.15  1.00  0.00  0.00  1.06  1.03  0.00  1.13 
    TO [BT] SA               1.15  1.00  1.06  0.91  1.06  1.03  0.00  1.13 
    TO BT [SA]               0.97  1.02  1.05  0.99  1.06  1.03  0.00  1.25 
    [TO] BT SD               0.91  0.73  0.56  0.56  0.98  1.09  0.56  0.88 
    TO [BT] SD               0.91  0.73  0.81  0.89  0.98  1.09  0.56  0.88 
    TO BT [SD]               0.99  0.83  0.97  0.99  0.98  1.09  0.56  0.89 
    [TO] BT BT SA            0.95  0.82  0.04  0.04  0.97  0.89  0.04  0.57 
    TO [BT] BT SA            0.95  0.82  0.70  0.50  0.97  0.89  0.04  0.57 
    TO BT [BT] SA            0.91  0.94  0.43  0.60  0.97  0.89  0.04  0.05 
    TO BT BT [SA]            0.93  0.87  0.04  0.35  0.97  0.89  0.04  0.72 
    ======================= ===== ===== ===== ===== ===== ===== ===== ===== 


    """
    prefix = os.path.commonprefix(map(str,ra.key))
    grid = []
    grid.append(map(lambda _:fmt(_,prefix),ra.dtype.names))
    for i in range(len(ra)):
        grid.append(map(lambda _:fmt(_,prefix),ra[i])) 
    pass
    return make_rst_table(grid)



def test_make_rst_table():
    print make_rst_table( [['Name', 'Favorite Food', 'Favorite Subject'],
                           ['Joe', 'Hamburgrs', 'I like things with really long names'],
                           ['Jill', 'Salads', 'American Idol'],
                           ['Sally', 'Tofu', 'Math']])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    path = os.path.expandvars("$TMP/stat.npy")
    log.info("path %s " % path )

    stat = np.load(path).view(np.recarray)
    print recarray_as_rst(stat)








    


