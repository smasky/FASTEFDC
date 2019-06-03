import numpy as np

def _init():
    global _global_dict
    _global_dict = {}

def set_value(name, value):
    _global_dict[name] = value

def get_value(name, defValue=None):
    return _global_dict[name]
#IC=55
#JC=15 #行
#KC=3  #层数
#LC=3514 #一头一尾
#LVC=3512 #真实网格
#IL=np.arange(LC)
#JL=np.arange(LC)
#LCT=np.arange(LC)
#LNC=np.arange(LC)
#LSC=np.arange(LC)
#LIJ=np.zeros((IC+1,JC+1))
