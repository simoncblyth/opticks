
import datetime


def timestamp():
    return int(datetime.datetime.now().strftime("%s")) 

def format_timestamp(v):
    return datetime.datetime.fromtimestamp(v).strftime("%c")

def format_kv(k,v):
    if k.startswith("memory"):return format_memory_size(v)
    if k.startswith("time"):return format_timestamp(v)
    return v

def format_memory_size(v):
    if v < 1e3:
        return '%.1f%s' % (v, ' ')
    elif v < 1e6:
        return '%.1f%s' % (v/1e3, 'K')
    elif v < 1e9:
        return '%.1f%s' % (v/1e6, 'M')
    else:
        return '%.1f%s' % (v/1e9, 'G')


