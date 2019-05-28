"""
    :author: Wu
    :modify: 20190528
    :url: https://github.com/smasky/FASTEFDC
    :copyright: © 2019 Smasky <492109831@qq.com>
    :license: MIT, see LICENSE for more details.
"""
import re
import numpy as np
def Read_cell(IC,JC):
    '''
        网格读取文件 IC行数 JC列数 默认文件名：cell.inp
    '''
    Cell=np.zeros((IC,JC))

    rull=r'\d{%d}'%(1)
    pattern=re.compile(rull)
    I=0
    with open('cell.inp','r') as f:
        for line in f:
            if('#' in line):
                pass
            else:
                string=line.lstrip().strip('\n').split('  ')
                result=np.array(pattern.findall(string[1]),dtype=np.int16)
                Cell[I]=result
                I+=1
    return Cell

def Read_dxdy(IC,JC):
    '''
        读取网格信息文件 IC行数 JC列数 默认文件名:dxdy.inp
    '''
    Dxp=np.zeros((IC,JC))
    Dyp=np.zeros((IC,JC))
    Hmp=np.zeros((IC,JC))
    Belv=np.zeros((IC,JC))
    Zbr=np.zeros((IC,JC))

    pattern=re.compile(r'\s+')
    
    with open('dxdy.inp','r') as f:
        for line in f:
            if('#' in line):
                pass
            else:
                string=line.lstrip().strip('\n')
                result=np.array(pattern.sub(' ',string).split(' '),dtype=np.float)
                I=int(result[1])
                J=int(result[0])
                Dxp[I,J]=result[2]
                Dyp[I,J]=result[3]
                Hmp[I,J]=result[4]
                Belv[I,J]=result[5]
                Zbr[I,J]=result[6]
        print(Dxp)
Read_dxdy(55,15)        

            
