"""
    :author: Wu
    :modify: 20190528
    :url: https://github.com/smasky/FASTEFDC
    :copyright: © 2019 Smasky <492109831@qq.com>
    :license: MIT, see LICENSE for more details.
"""
import re
import numpy as np
import math
import globalvalue as gb
#$from numba import jit
def Read_cell(IC,JC):
    '''
        网格读取文件 IC行数 JC列数 默认文件名：cell.inp
        CCell 倒置的网格
    '''

    Cell=np.zeros((IC,JC))
    CCell=np.zeros((IC,JC))
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
    for i in range(IC-1,-1,-1):
        CCell[i]=Cell[IC-i-1].copy()
    CCCell=np.zeros((IC+1,JC+1))
    for i in range(IC):
        CCCell[i+1][1:]=CCell[i]
    gb.set_value('Cell',CCell)
    gb.set_value('CCell',CCCell)

def Read_dxdy(IC,JC):
    '''
        读取网格信息文件 IC行数 JC列数 默认文件名:dxdy.inp
        return Dxp,Dyp,Hmp,Belv,Zbr
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
                I=int(result[1])-1
                J=int(result[0])-1
                Dxp[I,J]=result[2]
                Dyp[I,J]=result[3]
                Hmp[I,J]=result[4]
                Belv[I,J]=result[5]
                Zbr[I,J]=result[6]
    gb.set_value('Dxp',Dxp)
    gb.set_value('Dyp',Dyp)
    gb.set_value('Hmp',Hmp)
    gb.set_value('Belv',Belv)
    gb.set_value('Zbr',Zbr)

def Read_lxly(IC,JC):
    '''
        读取网格节点信息 IC行数，JC列数 默认文件名:lxly.inp
        return Lxp,Lyp,Cue,Cun,Cve,Cvn
    '''
    Lxp=np.zeros((IC,JC))
    Lyp=np.zeros((IC,JC))
    Cue=np.zeros((IC,JC))
    Cve=np.zeros((IC,JC))
    Cun=np.zeros((IC,JC))
    Cvn=np.zeros((IC,JC))

    pattern=re.compile(r'\s+')

    with open('lxly.inp','r') as f:
        for line in f:
            if('#' in line):
                pass
            else:
                string=line.lstrip().strip('\n')
                result=np.array(pattern.sub(' ',string).split(' '),dtype=np.float)
                I=int(result[1])
                J=int(result[0])
                Lxp[I,J]=result[2]
                Lyp[I,J]=result[3]
                ANG1=math.atan2(result[4],result[5])
                ANG2=math.atan2(result[6],result[7])
                ANG=(ANG1+ANG2)/2
                if(sign(ANG1)<=sign(ANG2)):
                    if(abs(ANG1)>1.57 or abs(ANG2)>1.57):
                        ANG=ANG+math.acos(-1)
                Cue[I,J]=math.cos(ANG)
                Cun[I,J]=math.sin(ANG)*-1
                Cve[I,J]=math.sin(ANG)
                Cvn[I,J]=math.cos(ANG)
        gb.set_value('Lxp',Lxp)
        gb.set_value('Lyp',Lyp)
        gb.set_value('Cue',Cue)
        gb.set_value('Cun',Cun)
        gb.set_value('Cve',Cve)
        gb.set_value('Cvn',Cvn)

def Read_Qser(Qnum,KC):
    '''
        读取流量边界条件 Qnum 流量边界个数 KC 层数 MQSER 时间序列的个数
        return:QSER MQSER
    '''
    pattern=re.compile(r'\s+')
    MQSER=np.arange(Qnum)
    QSER=np.zeros((1,KC+1))
    with open(r'qser.inp','r') as f:
        string=f.readlines()
    num=0
    for i in range(Qnum):
        while True:
            line=string[num]
            if('#' in line):
                pass
            else:
                break
            num+=1
        stringL=line.lstrip().strip('\n').split("\'")
        result=np.array(pattern.sub(' ',stringL[0].strip()).split(' '),dtype=np.float)

        lineNum=int(result[1])
        MQSER[i]=lineNum
        for j in range(lineNum):
            num+=1
            line=string[num]
            stringL=line.lstrip().strip('\n').strip()
            result=np.array(pattern.sub(' ',stringL).split(' '),dtype=np.float)
            QSER=np.vstack((QSER,result))
        num+=1
    gb.set_value('QSER',QSER)
    gb.set_value('MQSER',MQSER)

def Read_Pser(Pnum):
    '''
        读取压力边界条件 Pnum 压力边界个数 KC 层数 MPSER 个数 TAPSER 时间序列编号
        PSER 每一个序列对应的时间
        return:PSER MPSER
    '''
    pattern=re.compile(r'\s+')
    MPSER=np.zeros((1,Pnum))
    PSER=np.zeros((1,2))
    with open(r'C:\Users\sky\Desktop\EFDC-MPI-master\SampleModels\CurvilinearHarbourGrid\pser.inp','r') as f:
        string=f.readlines()
    num=0
    for i in range(Pnum):
        while True:
            line=string[num]
            if('#' in line):
                pass
            else:
                break
            num+=1
        stringL=line.lstrip().strip('\n').split("\'")
        result=np.array(pattern.sub(' ',stringL[0].strip()).split(' '),dtype=np.float)
        lineNum=int(result[0])
        MPSER[0][i]=lineNum
        for j in range(lineNum):
            num+=1
            line=string[num]
            stringL=line.lstrip().strip('\n').strip()
            result=np.array(pattern.sub(' ',stringL).split(' '),dtype=np.float)
            PSER=np.vstack((PSER,result))
        num+=1
    np.delete(PSER,0,axis=0)#0删不掉
    gb.set_value('PSER',PSER)
    gb.set_value('MPSER',MPSER)

def Read_Qinp(NQSIJ):
    with open('NQSIJ.inp') as f:
        string=f.readlines()
    num=0
    QINF=np.zeros((6,NQSIJ))
    QINF[:]=0
    LIJ=gb.get_value('LIJ')
    for line in string:
        if('#' in line):
            pass
        else:
            result=line.strip('\n').split(' ')
            IQS=int(result[0])
            QINF[0,num]=IQS
            JQS=int(result[1])
            QINF[1,num]=JQS
            QCON=result[2]
            QINF[2,num]=QCON
            QSFACTOR=result[3]
            QINF[3,num]=QSFACTOR
            QID=result[4]
            QINF[4,num]=QID
            QL=LIJ[IQS][JQS]
            QINF[5,num]=QL
            num+=1
    gb.set_value('QINF',QINF)
def Read_Pinp(NPSIJ):
    with open('NPSIJ.inp') as f:
        string=f.readlines()
    num=0
    PINF=np.zeros((6,NPSIJ))
    PINF[:]=0
    LIJ=gb.get_value('LIJ')
    for line in string:
        if('#' in line):
            pass
        else:
            result=line.strip('\n').split(' ')
            IPS=int(result[0])
            PINF[0,num]=IPS
            JPS=int(result[1])
            PINF[1,num]=JPS
            PCON=float(result[2])*9.81
            PINF[2,num]=PCON
            PID=result[3]
            PINF[3,num]=PID
            PL=LIJ[IPS][JPS]
            PINF[4,num]=PL
            PREND=result[4]
            PINF[5,num]=PREND
            num+=1
    gb.set_value('PINF',PINF)





##########################################################
#@jit
def sign(x2):
    if(x2<0):
        return -1
    elif(x2>0):
        return 1
    else:
        return 0

#Read_Pser(1)
#Read_Qser(1,3,4)
