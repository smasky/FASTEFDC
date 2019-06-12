"""
    :author: Wu
    :modify: 20190531
    :url: https://github.com/smasky/FASTEFDC
    :copyright: © 2019 Smasky <492109831@qq.com>
    :license: MIT, see LICENSE for more details.
工具类的文件 主要包括计算SUB SVB DXU DXV DYU DYV等
"""
from numba import jit
import numpy as np
import globalvalue as gb
import math
@jit
def Cal_LCT_IL_JL(IC,JC,LC):
    '''
    计算LCT,IL,JL LC网格个数 Dxp Dyp Hmp
    return:LCT IL JL LIJ LSC LNC
    '''
    Cell=gb.get_value('Cell')
    CCell=gb.get_value('CCell')
    Dxp=gb.get_value('Dxp')
    Dyp=gb.get_value('Dyp')
    Hmp=gb.get_value('Hmp')
    Belv=gb.get_value('Belv')
    Zbr=gb.get_value('Zbr')
    IL=gb.get_value('IL')
    JL=gb.get_value('JL')
    LCT=gb.get_value('LCT')
    LIJ=gb.get_value('LIJ')
    Dxpl=np.arange(LC,dtype=np.float)
    Dxpl[:]=1
    Dypl=np.arange(LC,dtype=np.float)
    Dypl[:]=1
    Hmpl=np.arange(LC,dtype=np.float)
    Hmpl[:]=0
    Belvl=np.arange(LC,dtype=np.float)
    Belvl[:]=0
    Zbrl=np.arange(LC,dtype=np.float)
    Zbrl[:]=0
    LNC=np.arange(LC)
    LSC=np.arange(LC)
    LNWC=np.arange(LC)
    LSEC=np.arange(LC)
    LSWC=np.arange(LC)
    LNEC=np.arange(LC)
    L=1
    for i in range(IC):
        for j in range(JC):
            Value=Cell[i][j]
            if(Value>0 and Value<9):
                LCT[L]=Value
                Dxpl[L]=Dxp[i][j]
                Dypl[L]=Dyp[i][j]
                Hmpl[L]=Hmp[i][j]
                Belvl[L]=Belv[i][j]
                Zbrl[L]=Zbr[i][j]
                IL[L]=i+1
                JL[L]=j+1
                LIJ[i+1][j+1]=L
                L+=1
    #print(L)
    L=1
    for i in range(LC-2):
        ii=IL[L]
        jj=JL[L]
        NValue=CCell[ii+1][jj]
        SValue=CCell[ii-1][jj]
        NWValue=CCell[ii+1][jj-1]
        SEValue=CCell[ii-1][jj+1]
        SWValue=CCell[ii-1][jj-1]
        NEValue=CCell[ii+1][jj+1]
        if(NValue==9):
            LNC[L]=LC-1
        else:
            LNC[L]=LIJ[ii+1][jj]
        if(SValue==9):
            LSC[L]=LC-1
        else:
            LSC[L]=LIJ[ii-1][jj]
        if(NWValue==9):
            LNWC[L]=LC-1
        else:
            LNWC[L]=LIJ[ii+1][jj-1]
        if(SEValue==9):
            LSEC[L]=LC-1
        else:
            LSEC[L]=LIJ[ii-1][jj+1]
        if(SWValue==9):
            LSWC[L]=LC-1
        else:
            LSWC[L]=LIJ[ii-1][jj-1]
        if(NEValue==9):
            LNEC[L]=LC-1
        else:
            LNEC[L]=LIJ[ii+1][jj+1]
        L+=1
    P=9.81*(Hmpl+Belvl)
    gb.set_value('LCT',LCT)
    gb.set_value('IL',IL)
    gb.set_value('JL',JL)
    gb.set_value('LIJ',LIJ)
    gb.set_value('LSC',LSC)
    gb.set_value('LNWC',LNWC)
    gb.set_value('LNEC',LNEC)
    gb.set_value('LSWC',LSWC)
    gb.set_value('LSEC',LSEC)
    gb.set_value('LNC',LNC)
    gb.set_value('Dxp',Dxpl)
    gb.set_value('Dyp',Dypl)
    gb.set_value('Hp',Hmpl)
    gb.set_value('Belv',Belvl)
    gb.set_value('Zbr',Zbrl)
    gb.set_value('P',P)

def Cal_Dx_Dy_HM(IC,JC,LC):
    '''
        计算Dxu Dyu Dxv Dyv Hu Hv
    '''
    Dxp=gb.get_value('Dxp')
    Dyp=gb.get_value('Dyp')
    LSC=gb.get_value('LSC')
    Hp=gb.get_value('Hp')
    Dxu=np.arange(LC,dtype=np.float)
    Dxu[:]=0
    Dxv=np.arange(LC,dtype=np.float)
    Dxv[:]=0
    Dyu=np.arange(LC,dtype=np.float)
    Dyu[:]=0
    Dyv=np.arange(LC,dtype=np.float)
    Dyv[:]=0
    Hu=np.arange(LC,dtype=np.float)
    Hu[:]=0
    Hv=np.arange(LC,dtype=np.float)
    Hv[:]=0
    for i in range(LC-2):
        LS=LSC[i+1]
        Dxu[i+1]=0.5*(Dxp[i+1]+Dxp[i])
        Dyu[i+1]=0.5*(Dyp[i+1]+Dyp[i])
        Dxv[i+1]=0.5*(Dxp[i+1]+Dxp[LS])
        Dyv[i+1]=0.5*(Dyp[i+1]+Dyp[LS])
        Hu[i+1]=0.5*(Dxp[i+1]*Dyp[i+1]*Hp[i+1]+Dxp[i]*Dyp[i]*Hp[i])/(Dxu[i+1]*Dyu[i+1])
        Hv[i+1]=0.5*(Dxp[i+1]*Dyp[i+1]*Hp[i+1]+Dxp[LS]*Dyp[LS]*Hp[LS])/(Dxv[i+1]*Dyv[i+1])
    Hu[0]=Hu[1]
    Hv[0]=Hv[1]
    Hu[-1]=Hu[-2]
    Hv[-1]=Hv[-2]
    Hpi=1.0/Hp
    Hui=1.0/Hu
    Hvi=1.0/Hv
    Hv1i=Hvi.copy()
    Hu1i=Hui.copy()
    gb.set_value('Dxu',Dxu)
    gb.set_value('Dxv',Dxv)
    gb.set_value('Dyu',Dyu)
    gb.set_value('Dyv',Dyv)
    gb.set_value('Hu',Hu)
    gb.set_value('Hv',Hv)
    gb.set_value('Hpi',Hpi)
    gb.set_value('Hui',Hui)
    gb.set_value('Hvi',Hvi)
    gb.set_value('Hv1i',Hv1i)
    gb.set_value('Hu1i',Hu1i)

@jit
def Cal_Sub_Svb(IC,JC,LC):
    '''
        计算Sub Svb
    '''
    L=1
    KC=gb.get_value('KC')
    Sub=np.arange(LC)
    Svb=np.arange(LC)
    IL=gb.get_value('IL')
    JL=gb.get_value('JL')
    LCT=gb.get_value('LCT')
    Cell=gb.get_value('CCell')
    L=1
    for i in range(LC-2):
        ii=IL[L]
        jj=JL[L]
        LCTN=LCT[L]
        if(Cell[ii][jj-1]==5):
            Sub[L]=1
        else:
            Sub[L]=0
        if(Cell[ii-1][jj]==5):
            Svb[L]=1
        else:
            Svb[L]=0
        L+=1
    Sub[0]=0
    Svb[0]=0
    Sub[-1]=0
    Svb[-1]=0
    ################################
    #边界
    LBERC=gb.get_value('LBERC')
    LBNRC=gb.get_value('LBNRC')
    PINF=gb.get_value('PINF')
    NPSIJ=gb.get_value('NPSIJ')
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    LIJ=gb.get_value('LIJ')
    SPB=gb.get_value('SPB')
    SAAX=np.arange(LC,dtype=np.float)
    SAAX[:]=1
    SAAY=np.arange(LC,dtype=np.float)
    SAAY[:]=1
    for i in range(NPSIJ):
        PREND=PINF[5,i]
        ii=int(PINF[0,i])
        jj=int(PINF[1,i])
        L=LIJ[ii][jj]
        SPB[L]=0
        if(PREND==0):#东
            Svb[L]=0
            Svb[L-1]=0
            SAAX[L]=0
            SAAY[L]=0
        elif(PREND==1):#南
            Sub[L]=0
            Svb[L]=0
            LN=LNC[L]
            Sub[LN]=0
            SAAX[L]=0
            SAAY[L]=0
            SAAX[LN]=0
            SAAY[LN]=0
        elif(PREND==3):#北
            Sub[L]=0
            LS=LSC[L]
            SAAX[L]=0
            SAAY[L]=0
            Sub[LS]=0
        elif(PREND==2):#西
            Sub[L]=0
            Svb[L]=0
            Svb[L+1]=0
            SAAX[L]=0
            SAAY[L]=0
            SAAX[L+1]=0
            SAAY[L+1]=0
    QINF=gb.get_value('QINF')
    NQSIJ=gb.get_value('NQSIJ')
    NBCS=0
    for i in  range(NQSIJ):
        L=int(QINF[5,i])
        LBERC[NBCS]=L
        if(Sub[L]>0.5):
            SAAX[L]=0
            SAAY[L]=0
        if(L<LC-4):
            if(Sub[L]<0.5 and (Sub[L+1]>0.5 and Sub[L+2]>0.5)):
                B=L+1
                LBERC[NBCS]=B
                SAAX[B]=0
                SAAY[B]=0
        if(L>1 and L<LC-2):
            if((Sub[L]>0.5 and Sub[L+1]<0.5) and (Sub[L-1]>0.5 and Sub[L-2]>0.5) ):
                B=L-1
                LBERC[NBCS]=B
                SAAX[B]=0
                SAAY[B]=0
                SAAX[L]=0
                SAAY[L]=0
        LBNRC[NBCS]=L
        if(Svb[L]<0.5):
            SAAX[L]=0
            SAAY[L]=0
        if(Svb[L]<0.5 and (Svb[LNC[L]]>0.5 and Svb[LNC[LNC[L]]]>0.5)):
            LBNRC[NBCS]=LNC[L]
            SAAX[LNC[L]]=0
            SAAY[LNC[L]]=0
        if(Svb[L]>0.5 and Svb[LNC[L]]<0.5 and (Svb[LSC[L]]>0.5 and Svb[LSC[LSC[L]]]>0.5)):
            LBNRC[NBCS]=LSC[L]
            SAAX[LNC[L]]=0
            SAAY[LNC[L]]=0
            SAAX[L]=0
            SAAY[L]=0
        NBCS+=1
    for i in range(NPSIJ):
        PREND=PINF[5,i]
        L=int(PINF[4,i])
        if(PREND==0):#东
            LBERC[NBCS]=L-1
            LBNRC[NBCS]=L
        elif(PREND==1):#南
            LBERC[NBCS]=L
            LBNRC[NBCS]=LNC[L]
        elif(PREND==3):#北
            LBERC[NBCS]=L
            LBNRC[NBCS]=LSC[L]
        elif(PREND==2):#北
            LBERC[NBCS]=L+1
            LBNRC[NBCS]=L
        NBCS+=1
    Subo=Sub.copy() #Sub origin
    Svbo=Svb.copy() #Svb origin
    ##########################边界
    gb.set_value('LBERC',LBERC)
    gb.set_value('LBNRC',LBNRC)
    gb.set_value('Sub',Sub)
    gb.set_value('Svb',Svb)
    gb.set_value('Subo',Subo)
    gb.set_value('Svbo',Svbo)
    gb.set_value('SAAX',SAAX)
    gb.set_value('SAAY',SAAY)


def Cal_reset_Dx_Dy_Rss(IC,JC,LC):
    '''
        根据网格关系重新计算DXU DXV DYU DYV
    '''
    Sub=gb.get_value('Sub')
    Svb=gb.get_value('Svb')
    Subo=gb.get_value('Subo')
    Svbo=gb.get_value('Svbo')
    Dxp=gb.get_value('Dxp')
    Dyp=gb.get_value('Dyp')
    Dxu=gb.get_value('Dxu')
    Dxv=gb.get_value('Dxv')
    Dyu=gb.get_value('Dyu')
    Dyv=gb.get_value('Dyv')
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    L=1
    for i in range(LC-2):
        #u方向的分量
        if(Sub[L]>0.5):
            Dxu[L]=0.5*(Dxp[L]+Dxp[L-1])
            Dyu[L]=0.5*(Dyp[L]+Dyp[L-1])
        elif(Sub[L]<0.5 and Sub[L+1]>0.5):
            Dxu[L]=Dxp[L]
            DDYDDDX=2.0*(Dyp[L+1]-Dyp[L])/(Dxp[L]+Dxp[L+1])
            Dyu[L]=Dyp[L]-0.5*Dxp[L]*DDYDDDX
        elif(Sub[L]<0.5 and Sub[L+1]<0.5):
            Dxu[L]=Dxp[L]
            Dyu[L]=Dyp[L]
        #v方向的分量
        LN=LNC[L]
        LS=LSC[L]
        if(Svb[L]>0.5):
            Dxv[L]=0.5*(Dxp[L]+Dxp[LS])
            Dyv[L]=0.5*(Dyp[L]+Dyp[LS])
        elif(Svb[L]<0.5 and Svb[LN]>0.5):
            DDXDDDY=2.0*(Dxp[LN]-Dxp[L])/(Dyp[L]+Dyp[LN])
            Dxv[L]=Dxp[L]-0.5*Dyp[L]*DDXDDDY
            Dyv[L]=Dxp[L]-0.5*Dyp[L]*DDXDDDY
        elif(Svb[L]<0.5 and Svb[LN]<0.5):
            Dxv[L]=Dxp[L]
            Dyv[L]=Dyp[L]
        L+=1
        ##########################
    gb.set_value('Dxu',Dxu)
    gb.set_value('Dxv',Dxv)
    gb.set_value('Dyu',Dyu)
    gb.set_value('Dyv',Dyv)
    #####################################
    '''
    计算RSSBCE RSSBCN RSSBCW RSSBCS
    '''
    #################################
    #计算Dxdi Dydj
    Dydi=np.arange(LC)
    Dydi[-1]=0
    Dxdj=np.arange(LC)
    Dxdj[-1]=0
    CCell=gb.get_value('CCell')
    IL=gb.get_value('IL')
    JL=gb.get_value('JL')
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        LS=LSC[L]
        ii=IL[L]
        jj=JL[L]
        if(CCell[ii][jj-1]==5):
            if(CCell[ii][jj+1]==5):
                Dydi[L]=Dyu[L+1]-Dyu[L]
            else:
                DDYDDDX=2.0*(Dyp[L]-Dyp[L-1])/(Dxp[L]+Dxp[L-1])
                DYUP1=Dyp[L]+0.5*DDYDDDX*Dxp[L]
                Dydi[L]=DYUP1-Dyu[L]
        else:
            if(CCell[ii][jj+1]==5):
                DDYDDDX=2.0*(Dyp[L+1]-Dyp[L])/(Dxp[L+1]+Dxp[L])
                DYUM1=Dyp[L]-0.5*DDYDDDX*Dxp[L]
                Dydi[L]=Dyu[L]-DYUM1
            else:
                Dydi[L]=0
        if(CCell[ii-1][jj]==5):
            if(CCell[ii+1][jj]==5):
                Dxdj[L]=Dxv[LN]-Dxv[L]
            else:
                DDXDDDY=2.0*(Dxp[L]-Dxp[LS])/(Dyp[L]+Dyp[LS])
                DXVLN=Dxp[L]+0.5*DDXDDDY*Dyp[L]
                Dxdj[L]=DXVLN-Dxv[L]
        else:
            if(CCell[ii+1][jj]==5):
                DDXDDDY=2*(Dxp[LN]-Dxp[L])/(Dyp[LN]+Dyp[L])
                DXVLS=Dxp[L]-0.5*DDXDDDY*Dyp[L]
                Dxdj[L]=Dxv[L]-DXVLS
            else:
                Dxdj[L]=0
        L+=1
    gb.set_value('Dydi',Dydi)
    gb.set_value('Dxdj',Dxdj)
    ###########################################
    #初始化一波参数
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    Dxyu=np.arange(LC)
    Dxyu[-1]=0
    Dxyv=np.arange(LC)
    Dxyv[-1]=0
    Dxyp=np.arange(LC)
    Dxyp[-1]=0
    Hru=np.arange(LC)
    Hru[-1]=0
    Hrv=np.arange(LC)
    Hrv[-1]=0
    Hruo=np.arange(LC)
    Hruo[-1]=0
    Hrvo=np.arange(LC)
    Hrvo[-1]=0
    Hru=Sub*Hu*Dyu/Dxu
    Hru[0]=0
    Hru[-1]=0
    Hrv=Svb*Hv*Dxv/Dyv
    Hrv[0]=0
    Hrv[-1]=0
    Hruo=Subo*Dyu/Dxu
    Hruo[0]=0
    Hruo[-1]=0
    Hrvo=Svbo*Dxv/Dyv
    Hrvo[0]=0
    Hrvo[-1]=0
    Dxyu=Dxu*Dyu
    Dxyv=Dxv*Dyv
    Dxyp=Dxp*Dyp
    Hrxyv=np.arange(LC)
    Hrxyu=np.arange(LC)
    Hrxyv=Dxu/Dyu
    Hrxyv[-1]=0
    Hrxyv[0]=0
    Hrxyu=Dxv/Dyv
    Hrxyu[0]=0
    Hrxyu[-1]=0
    Sbx=np.arange(LC)
    Sbx[-1]=0
    Sby=np.arange(LC)
    Sby[-1]=0
    Sbx=0.5*Sub*Dyu
    Sby=0.5*Svb*Dxv
    Sbxo=Sbx.copy()
    Sbyo=Sby.copy()
    gb.set_value('Dxyu',Dxyu)
    gb.set_value('Dxyv',Dxyv)
    gb.set_value('Dxyp',Dxyp)
    gb.set_value('Hru',Hru)
    gb.set_value('Hrv',Hrv)
    gb.set_value('Hruo',Hruo)
    gb.set_value('Hrvo',Hrvo)
    gb.set_value('Hrxyv',Hrxyv)
    gb.set_value('Hrxyu',Hrxyu)
    gb.set_value('Sbx',Sbx)
    gb.set_value('Sby',Sby)
    gb.set_value('Sbxo',Sbxo)
    gb.set_value('Sbyo',Sbyo)
@jit
def Cal_UV_VU(IC,JC,KC,LC):
    '''
        计算Uv、Vu的值
    '''
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    LNWC=gb.get_value('LNWC')
    LNEC=gb.get_value('LNEC')
    LSWC=gb.get_value('LSWC')
    LSEC=gb.get_value('LSEC')
    UV=gb.get_value('UV')
    VU=gb.get_value('VU')
    Hp=gb.get_value('Hp')
    H1p=gb.get_value('H1p')
    U=gb.get_value('U')
    U1=gb.get_value('U1')
    V=gb.get_value('V')
    V1=gb.get_value('V1')
    Hvi=gb.get_value('Hvi')
    Hv1i=gb.get_value('Hv1i')
    Hui=gb.get_value('Hui')
    Hu1i=gb.get_value('Hu1i')
    U1V=UV.copy()
    V1U=VU.copy()
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        LS=LSC[L]
        LSE=LSEC[L]
        LNW=LNWC[L]
        LSW=LSWC[L]
        UV[L]=0.25*(Hp[LS])*(U[LSE,0]+U[LS,0])+Hp[L]*(U[L+1,0]+U[L,0])*Hvi[L]
        U1V[L]=0.25*(H1p[LS])*(U1[LSE,0]+U1[LS,0])+H1p[L]*(U1[L+1,0]+U1[L,0])*Hv1i[L]
        VU[L]=0.25*(Hp[L-1]*(V[LNW,0]+V[L-1,0])+Hp[L]*(V[LN,0]+V[L,0]))*Hui[L]
        V1U[L]=0.25*(H1p[L-1]*(V1[LNW,0]+V1[L-1,0])+H1p[L]*(V1[LN,0]+V1[L,0]))*Hu1i[L]
        L+=1
    gb.set_value('UV',UV)
    gb.set_value('U1V',U1V)
    gb.set_value('VU',VU)
    gb.set_value('V1U',V1U)

@jit
def Cal_UV_VU_2(IC,JC,KC,LC):
    '''
        计算Uv、Vu的值
    '''
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    LNWC=gb.get_value('LNWC')
    LNEC=gb.get_value('LNEC')
    LSWC=gb.get_value('LSWC')
    LSEC=gb.get_value('LSEC')
    UV=gb.get_value('UV')
    VU=gb.get_value('VU')
    Hp=gb.get_value('Hp')
    H1p=gb.get_value('H1p')
    U=gb.get_value('U')
    U1=gb.get_value('U1')
    V=gb.get_value('V')
    V1=gb.get_value('V1')
    Hvi=gb.get_value('Hvi')
    Hv1i=gb.get_value('Hv1i')
    Hui=gb.get_value('Hui')
    Hu1i=gb.get_value('Hu1i')
    U1V=UV.copy()
    V1U=VU.copy()
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        LS=LSC[L]
        LSE=LSEC[L]
        LNW=LNWC[L]
        LSW=LSWC[L]
        UV[L]=0.25*(Hp[LS])*(U[LSE,0]+U[LS,0])+Hp[L]*(U[L+1,0]+U[L,0])*Hvi[L]
        VU[L]=0.25*(Hp[L-1]*(V[LNW,0]+V[L-1,0])+Hp[L]*(V[LN,0]+V[L,0]))*Hui[L]
        L+=1
    gb.set_value('UV',UV)
    gb.set_value('U1V',U1V)
    gb.set_value('VU',VU)
    gb.set_value('V1U',V1U)

@jit
def Cal_init_STBXY(IC,JC,KC,LC):
    '''
        STBXY的初始化
    '''
    STBX=np.arange(LC,dtype=np.float)
    STBX[:]=1
    STBY=np.arange(LC,dtype=np.float)
    STBY[:]=1
    RITB1=1.0#
    RITB=0.0#
    CDLIMIT=0.5#三个系数
    gb.set_value('STBX',STBX)
    gb.set_value('STBY',STBY)
    gb.set_value('RITB1',RITB1)
    gb.set_value('RITB',RITB)
    gb.set_value('CDLIMIT',CDLIMIT)

@jit
def Cal_STBXY(IC,JC,LC):
    '''
        计算底部切应力 STBXY
    '''
    Dxp=gb.get_value('Dxp')
    Dyp=gb.get_value('Dyp')
    Zbr=gb.get_value('Zbr')
    Dxu=gb.get_value('Dxu')
    Dyv=gb.get_value('Dyv')
    U1=gb.get_value('U1')
    V1U=gb.get_value('V1U')
    U1V=gb.get_value('U1V')
    V1=gb.get_value('V1')
    LSC=gb.get_value('LSC')
    DT=gb.get_value('DT')
    CDLIMIT=gb.get_value('CDLIMIT')
    H1u=gb.get_value('H1u')
    H1v=gb.get_value('H1v')
    STBX=gb.get_value('STBX')
    STBY=gb.get_value('STBY')
    DZC=gb.get_value('DZC')
    L=1
    for i in range(LC-2):
        LS=LSC[L]
        ZBRATU=0.5*(Dxp[L-1]*Zbr[L-1]+Dxp[L]*Zbr[L])/Dxu[L]
        ZBRATV=0.5*(Dyp[LS]*Zbr[LS]+Dyp[L]*Zbr[L])/Dyv[L]
        UMAGTMP=math.sqrt(U1[L,0]*U1[L,0]+V1U[L]*V1U[L]+0.00000000000001)
        VMAGTMP=math.sqrt(U1V[L]*U1V[L]+V1[L,0]*V1[L,0]+0.00000000000001)
        CDMAXU=CDLIMIT*H1u[L]/(DT*UMAGTMP)
        CDMAXV=CDLIMIT*H1v[L]/(DT*VMAGTMP)
        HURTMP=max(ZBRATU,H1u[L])
        HVRTMP=max(ZBRATV,H1v[L])
        DZHUDZBR=1.0+0.5*DZC[0]*HURTMP/ZBRATU
        DZHVDZBR=1.0+0.5*DZC[0]*HVRTMP/ZBRATV
        A=(np.power((np.log(DZHUDZBR)),2))
        B=(np.power((np.log(DZHVDZBR)),2))
        STBX[L]=0.16/A
        STBY[L]=0.16/B
        STBX[L]=min(CDMAXU,STBX[L])
        STBY[L]=min(CDMAXV,STBY[L])
        L+=1
    gb.set_value('STBX',STBX)
    gb.set_value('STBY',STBY)
    ################################################
    TBX=gb.get_value('TBX')
    TBY=gb.get_value('TBY')
    VU=gb.get_value('VU')
    UV=gb.get_value('UV')
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    U=gb.get_value('U')
    V=gb.get_value('V')
    L=1
    AVCON1=0 #与AVCON成比例
    for i in range(LC-2):
        TBX[L]=(AVCON1/Hu[L]+STBX[L]*np.sqrt(VU[L]*VU[L]+U[L,0]*U[L,0]))*U[L,0]
        TBY[L]=(AVCON1/Hv[L]+STBY[L]*np.sqrt(UV[L]*UV[L]+V[L,0]*V[L,0]))*V[L,0]
        L+=1
    gb.set_value('TBX',TBX)
    gb.set_value('TBY',TBY)

@jit
def Cal_HDMF(IC,JC,LC,KC):
    '''
        计算水平扩散通量
    '''
    N=gb.get_value('N')
    Sub=gb.get_value('Sub')
    Svb=gb.get_value('Svb')
    if(N<15):
        ICORDYU=gb.get_value('ICORDYU')
        ICORDXV=gb.get_value('ICORDXV')
        LSC=gb.get_value('LSC')
        LNC=gb.get_value('LNC')
        L=1

        for i in range(LC-2):
            LS=LSC[L]
            LW=L-1
            if(Sub[L]<0.5 and Sub[LS]<0.5):
                ICORDYU[L]=0
            if(Sub[L]>0.5 and Sub[LS]>0.5):
                ICORDYU[L]=1
            if(Sub[L]<0.5 and Sub[LS]>0.5):
                ICORDYU[L]=2
            if(Sub[L]>0.5 and Sub[LS]<0.5):
                ICORDYU[L]=3
            ###############################
            if(Svb[L]<0.5 and Svb[LW]<0.5):
                ICORDXV[L]=0
            if(Svb[L]>0.5 and Svb[LW]>0.5):
                ICORDXV[L]=1
                if(Sub[L]<0.5):
                    ICORDXV[L]=3
            if(Svb[L]<0.5 and Svb[LW]>0.5):
                ICORDXV[L]=2
            if(Svb[L]>0.5 and Svb[LW]<0.5):
                ICORDXV[L]=3
            L+=1
        gb.set_value('ICORDXV',ICORDXV)
        gb.set_value('ICORDYU',ICORDYU)
        ######################################
    ICORDYU=gb.get_value('ICORDYU')
    ICORDXV=gb.get_value('ICORDXV')
    Dxu1=gb.get_value('Dxu1')
    Dyv1=gb.get_value('Dyv1')
    Dyu1=gb.get_value('Dyu1')
    Dxv1=gb.get_value('Dxv1')
    Sxy=gb.get_value('Sxy')
    Dxp=gb.get_value('Dxp')
    Dyp=gb.get_value('Dyp')
    U=gb.get_value('U')
    V=gb.get_value('V')
    Dxv=gb.get_value('Dxv')
    Dyu=gb.get_value('Dyu')
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        LS=LSC[L]
        LW=L-1
        for k in range(KC):
            Dxu1[L,k]=Sub[L+1]*(U[L+1,k]-U[L,k])/Dxp[L]
            Dyv1[L,k]=Svb[LN]*(V[LN,k]-V[L,k])/Dyp[L]
            if(ICORDYU[L]==1):
                Dyu1[L,k]=2.0*Svb[L]*(U[L,k]-U[LS,k])/(Dyu[L]+Dyu[LS])
            else:
                Dyu1[L,k]=0
            if(ICORDXV[L]==1):
                Dxv1[L,k]=2.0*Sub[L]*(V[L,k]-V[LW,k])/(Dxv[L]+Dxv[LW])
            else:
                Dxv1[L,k]=0
            Sxy[L,k]=Dyu1[L,k]+Dxv1[L,k]
        L+=1
    gb.set_value('Sxy',Sxy)
    gb.set_value('Dxu1',Dxu1)
    gb.set_value('Dyv1',Dyv1)
    gb.set_value('Dyu1',Dyu1)
    gb.set_value('Dxv1',Dxv1)
    #####################################
    AHD=gb.get_value('AHD')
    AHO=gb.get_value('AHO')
    AH=gb.get_value('AH')
    L=1
    if(AHD>0):
        for i in range(LC-2):
            for k in range(KC):
                TMPVAL=AHD*Dxp[L]*Dyp[L]
                DSQR=Dxu1[L,k]*Dxu1[L,k]+Dyv1[L,k]*Dyv1[L,k]+Sxy[L,k]*Sxy[L,k]/4
                AH[L,k]=AHO+TMPVAL*np.sqrt(DSQR)
            L+=1
    elif(N<10):
        for i in range(LC-2):
            for k in range(KC):
                    AH[L,k]=AHO
            L+=1
    gb.set_value('AH',AH)
###################################################
    FMDUX=gb.get_value('FMDUX')
    FMDUY=gb.get_value('FMDUY')
    FMDVX=gb.get_value('FMDVX')
    FMDVY=gb.get_value('FMDVY')
    Hp=gb.get_value('Hp')
    Dxu=gb.get_value('Dxu')
    Dyv=gb.get_value('Dyv')
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    L=1
    for i in range(LC-2):
        LS=LSC[L]
        LN=LNC[L]
        for k in range(KC):
            FMDUX[L,k]=(Dyp[L]*Hp[L]*AH[L,k]*Dxu1[L,k]-Dyp[L-1]*Hp[L-1]*AH[L-1,k]*Dxu1[L-1,k])*Sub[L]
            FMDUY[L,k]=(0.5*(Dxu[LN]+Dxu[L])*Hu[L]*(AH[LN,k]*Sxy[LN,k]-AH[L,k]*Sxy[L,k]))*Svb[LN]
            FMDVX[L,k]=0.5*(Dyv[L+1]+Dyv[L])*Hv[L]*(AH[L+1,k]*Sxy[L+1,k]-AH[L,k]*Sxy[L,k])*Sub[L+1]
            FMDVY[L,k]=(Dxp[L]*Hp[L]*AH[L,k]*Dyv1[L,k]-Dxp[LS]*Hp[LS]*AH[LS,k]*Dyv1[LS,k])*Svb[L]
        L+=1
    ###############################
    ###边界原因
    QINF=gb.get_value('QINF')
    PINF=gb.get_value('PINF')
    NQSIJ=gb.get_value('NQSIJ')
    NPSIJ=gb.get_value('NPSIJ')
    for i in range(NQSIJ):
        L=int(QINF[5,i])
        FMDUX[L,:]=0
        FMDUY[L,:]=0
        FMDVX[L,:]=0
        FMDVY[L,:]=0

    for i in range(NPSIJ):
        L=int(PINF[4,i])
        FMDUX[L,:]=0
        FMDUY[L,:]=0
        FMDVX[L,:]=0
        FMDVY[L,:]=0

    gb.set_value('FMDUX',FMDUX)
    gb.set_value('FMDUY',FMDUY)
    gb.set_value('FMDVX',FMDVX)
    gb.set_value('FMDVY',FMDVY)


def Cal_init_Tsxy(IC,JC,KC):
    '''
    '''
    pass
@jit
def Cal_Exp2T(IC,JC,LC,KC):
    '''
        计算显示右边项
    '''
    FCAXE=gb.get_value('FCAXE')
    FCAYE=gb.get_value('FCAYE')
    FCAX=gb.get_value('FCAX')
    FCAY=gb.get_value('FCAY')
    FXE=gb.get_value('FXE')
    FYE=gb.get_value('FYE')
    FCAXE[:]=0
    FXE[:]=0
    FYE[:]=0
    FCAYE[:]=0
    UHDY=gb.get_value('UHDY')
    VHDX=gb.get_value('VHDX')
    U=gb.get_value('U')
    V=gb.get_value('V')
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    FUHU=gb.get_value('FUHU')
    FVHU=gb.get_value('FVHU')
    FUHV=gb.get_value('FUHV')
    FVHV=gb.get_value('FVHV')
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        LS=LSC[L]
        for k in range(KC):
            UHC=0.5*(UHDY[L,k]+UHDY[LS,k])
            UHB=0.5*(UHDY[L,k]+UHDY[L+1,k])
            VHC=0.5*(VHDX[L,k]+VHDX[L-1,k])
            VHB=0.5*(VHDX[L,k]+VHDX[LN,k])

            FUHU[L,k]=max(UHB,0.0)*U[L,k]+min(UHB,0)*U[L+1,k]
            FVHU[L,k]=max(VHC,0)*U[LS,k]+min(VHC,0.0)*U[L,k]
            FVHV[L,k]=max(VHB,0.0)*V[L,k]+min(VHB,0.0)*V[LN,k]
            FUHV[L,k]=max(UHC,0.0)*V[L-1,k]+min(UHC,0.0)*V[L,k]
        L+=1
#########################################################################
#垂向对流项
    Dxyu=gb.get_value('Dxyu')
    Dxyv=gb.get_value('Dxyv')
    FWU=gb.get_value('FWU')
    FWV=gb.get_value('FWV')
    W=gb.get_value('W')
    L=1
    for i in range(LC-2):
        LS=LSC[L]
        for k in range(KC-1):
            WU=0.5*Dxyu[L]*(W[L,k]+W[L-1,k])
            WV=0.5*Dxyv[L]*(W[L,k]+W[LS,k])
            FWU[L,k]=max(WU,0.0)*U[L,k]+min(WU,0.0)*U[L,k+1]
            FWV[L,k]=max(WU,0.0)*V[L,k]+min(WV,0.0)*V[L,k+1]
        L+=1
###############################################################
###计算科氏力 FACXE
#########################################
    L=1
    FX=gb.get_value('FX')
    FY=gb.get_value('FY')
    for i in range(LC-2):
        LN=LNC[L]
        LS=LSC[L]
        for k in range(KC):
            FX[L,k]=(FUHU[L,k]-FUHU[L-1,k]+FVHU[LN,k]-FVHU[L,k])#缺FUHJ SAAX大概边界取0
            FY[L,k]=FUHV[L+1,k]-FUHV[L,k]+FVHV[L,k]-FVHV[LS,k]#缺FVHJ
        L+=1
    #######################
    ####边界处理
    QINF=gb.get_value('QINF')
    PINF=gb.get_value('PINF')
    SAAX=gb.get_value('SAAX')
    SAAY=gb.get_value('SAAY')
    NQSIJ=gb.get_value('NQSIJ')
    NPSIJ=gb.get_value('NPSIJ')
    LBERC=gb.get_value('LBERC')
    LBNRC=gb.get_value('LBNRC')
    for i in range(NQSIJ):
        L=int(QINF[5,i])
        FX[L,:]=SAAX[L]*FX[L,:]
        FY[L,:]=SAAY[L]*FY[L,:]

    for i in range(NPSIJ):
        L=int(PINF[4,i])
        FX[L,:]=SAAX[L]*FX[L,:]
        FY[L,:]=SAAY[L]*FY[L,:]
    for i in range(NPSIJ+NQSIJ):
        L=LBERC[i]
        LL=LBNRC[i]
        FX[L,:]=SAAX[L]*FX[L,:]
        FY[LL,:]=SAAY[LL]*FY[LL,:]
    ##########################
    ##################################
    #######水平动量扩散
    ##############################
    ###########计算FACXE
    L=1
    FMDUX=gb.get_value('FMDUX')
    FMDVX=gb.get_value('FMDVX')
    FMDUY=gb.get_value('FMDUY')
    FMDVY=gb.get_value('FMDVY')
    for i in range(LC-2):
        for k in range(KC):
            FX[L,k]=FX[L,k]-(FMDUX[L,k]+FMDUY[L,k])
            FY[L,k]=FY[L,k]-(FMDVX[L,k]+FMDVY[L,k])
        L+=1
    ########################################
    #计算FXE 外模态计算
    L=1
    DZC=gb.get_value('DZC')
    for i in range(LC-2):
        for k in range(KC):
            ###FCAXC
            FXE[L]=FXE[L]+FX[L,k]*DZC[k]
            FYE[L]=FYE[L]+FY[L,k]*DZC[k]
        L+=1
    #####################
    #内模块计算 覆盖了FX FY
    L=1
    for i in range(LC-2):
        for k in range(KC):
            if(k==0):
                FF=0
                FA=0
            else:
                FF=FWU[L,k-1]
                FA=FWV[L,k-1]
            FX[L,k]=FX[L,k]+SAAX[L]*(FWU[L,k]-FF)/DZC[k]
            FY[L,k]=FY[L,k]+SAAY[L]*(FWV[L,k]-FA)/DZC[k]
        L+=1
    ############################
    ###计算内部剪切力项
    Du=gb.get_value('Du')
    Dv=gb.get_value('Dv')
    DT=gb.get_value('DT')
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    SNLT=gb.get_value('SNLT')
    L=1
    for i in range(LC-2):
        Du[L,KC-1]=0
        Dv[L,KC-1]=0
        for k in range(KC-1):
            CDZF=DZC[k]*DZC[k+1]/(DZC[k]+DZC[k+1])
            Du[L,k]=CDZF*(Hu[L]*(U[L,k+1]-U[L,k]))/DT+(FCAX[L,k+1]-FCAX[L,k]+SNLT*(FX[L,k]-FX[L,k+1]))/Dxyu[L]
            Dv[L,k]=CDZF*(Hv[L]*(V[L,k+1]-V[L,k]))/DT+(FCAY[L,k+1]-FCAY[L,k+1]+SNLT*(FY[L,k]-FY[L,k+1]))/Dxyv[L]
        L+=1
    #######################
    gb.set_value('Du',Du)
    gb.set_value('Dv',Dv)
    gb.set_value('FX',FX)
    gb.set_value('FY',FY)
    gb.set_value('FXE',FXE)
    gb.set_value('FYE',FYE)
    gb.set_value('FWU',FWU)
    gb.set_value('FWV',FWV)
    gb.set_value('FUHU',FUHU)
    gb.set_value('FUHV',FUHV)
    gb.set_value('FVHU',FVHU)
    gb.set_value('FVHV',FVHV)

@jit
def Cal_QVS(IC,JC,LC,KC):
    '''
        读取流量数据
    '''
    QSUME=gb.get_value('QSUME')
    QSUM1E=gb.get_value('QSUM1E')
    QSUM=gb.get_value('QSUM')
    QINF=gb.get_value('QINF')
    NQSIJ=gb.get_value('NQSIJ')
    N=gb.get_value('N')
    MQSER=gb.get_value('MQSER')
    QSER=gb.get_value('QSER')
    DZC=gb.get_value('DZC')
    QSUM1E=QSUME.copy()
    QSUME[:]=0
    QSUM[:,:]=0
    time_day=N/86400.0

    for i in range(NQSIJ):
        L=int(QINF[5,i])
        ID=int(QINF[4,i])
        QCON=QINF[2,i]
        QFACTOR=QINF[3,i]
        EQ=np.sum(MQSER[:ID])#4+4
        BQ=EQ-MQSER[ID-1]#8-4
        for ii in range(BQ,EQ-1):
            time1=QSER[ii+1,0]
            time2=QSER[ii+2,0]
            if(time_day>=time1 and time_day<=time2):
                tt=ii+1
                break
        time1=QSER[tt,0]
        time2=QSER[tt+1,0]
        QQ=(QSER[tt,1:].copy()-QSER[tt+1,1:].copy())/(time1-time2)*(time_day-time2)+QSER[tt+1,1:].copy()
        QSUM[L,:]=QQ*QFACTOR+QCON*DZC
        for ii in range(KC):
            QSUME[L]=QSUME[L]+QSUM[L,ii]
    gb.set_value('QSUM',QSUM)
    gb.set_value('QSUME',QSUME)
    gb.set_value('QSUM1E',QSUM1E)

@jit
def Cal_OBC():
    '''
        设置水位边界条件
    '''
    PINF=gb.get_value('PINF')
    NPSIJ=gb.get_value('NPSIJ')
    CS=gb.get_value('CS')
    CW=gb.get_value('CW')
    CE=gb.get_value('CE')
    CN=gb.get_value('CN')
    FP=gb.get_value('FP')
    H1u=gb.get_value('H1u')
    H1v=gb.get_value('H1v')
    DT2=gb.get_value('DT2')
    DT=gb.get_value('DT')
    Hruo=gb.get_value('Hruo')
    Hrvo=gb.get_value('Hrvo')
    Dxyp=gb.get_value('Dxyp')
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    for i in range(NPSIJ):
        L=int(PINF[4,i])
        PREND=int(PINF[5,i])
        if(PREND==0):#东
            CS[L]=0
            CW[L]=0
            CE[L]=0
            CN[L]=0
            FP[L]=PINF[2,i]
            CWT=0.5*DT2*9.81*Hruo[L]*H1u[L]
            FP[L-1]=FP[L-1]+CWT*FP[L]
            FP[L]=2/DT2*Dxyp[L]*FP[L]
            CE[L-1]=0
        elif(PREND==1):#南
            LN=LNC[L]
            CS[L]=0
            CW[L]=0
            CE[L]=0
            CN[L]=0
            FP[L]=PINF[2,i]
            CNT=0.5*DT2*9.81*Hrvo[LN]*H1v[LN]
            FP[LN]=FP[LN]+CNT*FP[L]
            FP[L]=1/DT*Dxyp[L]*FP[L]
            CS[LN]=0
        elif(PREND==3):#北
            LS=LSC[L]
            CS[L]=0
            CW[L]=0
            CE[L]=0
            CN[L]=0
            FP[L]=PINF[2,i]
            CST=0.5*DT2*9.81*Hrvo[L]*H1v[L]
            FP[LS]=FP[LS]+CST*FP[L]
            FP[L]=2/DT2*Dxyp[L]*FP[L]
        elif(PREND==2):#西
            CS[L]=0
            CW[L]=0
            CE[L]=0
            CN[L]=0
            FP[L]=PINF[2,i]
            CET=0.5*DT2*9.81*Hruo[L+1]*H1u[L+1]
            FP[L+1]=FP[L+1]+CET*FP[L]
            FP[L]=2/DT2*Dxyp[L]*FP[L]
            CW[L+1]=0
    gb.set_value('CS',CS)
    gb.set_value('CW',CW)
    gb.set_value('CN',CN)
    gb.set_value('CE',CE)
    gb.set_value('FP',FP)

@jit
def Cal_External(IC,JC,LC,KC):
    '''
        外模态计算水位 U V
    '''
    DT=gb.get_value('DT')
    LSC=gb.get_value('LSC')
    LNC=gb.get_value('LNC')
    Hp=gb.get_value('Hp')
    Dxu=gb.get_value('Dxu')
    Dyv=gb.get_value('Dyv')
    FUHDYE=gb.get_value('FUHDYE')
    FUHDYE[0]=0
    FUHDYE[-1]=0
    FVHDXE=gb.get_value('FVHDXE')
    FVHDXE[0]=0
    FVHDXE[-1]=0
    UHDYE=gb.get_value('UHDYE')
    VHDXE=gb.get_value('VHDXE')
    Hruo=gb.get_value('Hruo')
    Hrvo=gb.get_value('Hrvo')
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    DT2=gb.get_value('DT2')
    Sub=gb.get_value('Sub')
    Subo=gb.get_value('Subo')
    Svb=gb.get_value('Svb')
    Svbo=gb.get_value('Svbo')
    RITB1=gb.get_value('RITB1')
    TSX=gb.get_value('TSX')
    TSY=gb.get_value('TSY')
    TBX=gb.get_value('TBX')
    TBY=gb.get_value('TBY')
    Dxyu=gb.get_value('Dxyu')
    Dxyv=gb.get_value('Dxyv')
    FXE=gb.get_value('FXE')
    FYE=gb.get_value('FYE')
    SNLT=gb.get_value('SNLT')
    P=gb.get_value('P')
    Sub1=Sub.copy()
    Svb1=Svb.copy()
    H2p=Hp.copy()#将Hp赋予H2p
    L=1
    for i in range(LC-2):
        LS=LSC[L]
        FUHDYE[L]=UHDYE[L]-DT2*Sub[L]*Hruo[L]*Hu[L]*(P[L]-P[L-1])+Sub[L]*DT/Dxu[L]*(Dxyu[L]*(TSX[L]-RITB1*TBX[L])-SNLT*FXE[L])
        FVHDXE[L]=VHDXE[L]-DT2*Svb[L]*Hrvo[L]*Hv[L]*(P[L]-P[LS])+Svb[L]*DT/Dyv[L]*(Dxyv[L]*(TSX[L]-RITB1*TBY[L])-SNLT*FYE[L])
        L+=1
    RCX=np.arange(LC,dtype=np.float)
    RCY=np.arange(LC,dtype=np.float)
    RCX[:]=1
    RCY[:]=1
    RCX[0]=0
    RCY[0]=0
    RCX[-1]=0
    RCY[-1]=0
    gb.set_value('RCX',RCX)
    gb.set_value('RCY',RCY)
    Sub=Subo.copy()
    Svb=Svbo.copy()
    QSUME=gb.get_value('QSUME')
    ##########################
    #####QSUME QSUMTMP
    P=gb.get_value('P')
    H1u=Hu.copy()
    H1v=Hv.copy()
    UHDY1E=UHDYE.copy()
    VHDX1E=VHDXE.copy()
    gb.set_value('UHDY1E',UHDY1E)
    gb.set_value('VHDX1E',VHDX1E)
    P1=P.copy()
    H1p=Hp.copy()
    Dxyp=gb.get_value('Dxyp')
    ####################
    FP1=np.arange(LC,dtype=np.float)
    FP1[:]=0
    FP=np.arange(LC,dtype=np.float)
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        FP1[L]=Dxyp[L]*P[L]/DT-0.5*9.81*(UHDYE[L+1]-UHDYE[L]+VHDXE[LN]-VHDXE[L])
        L+=1

    L=1
    C=0.5*9.81
    for i in range(LC-2):
        LN=LNC[L]
        FP[L]=FP1[L]-C*(FUHDYE[L+1]-FUHDYE[L]+FVHDXE[LN]-FVHDXE[L]-2.0*QSUME[L])
        L+=1
    C=-0.5*9.81*DT2
    L=1
    CS=np.arange(LC,dtype=np.float)
    CW=np.arange(LC,dtype=np.float)
    CE=np.arange(LC,dtype=np.float)
    CN=np.arange(LC,dtype=np.float)
    CC=np.arange(LC,dtype=np.float)
    for i in range(LC-2):
        LN=LNC[L]
        CS[L]=C*(Svb[L])*(Hrvo[L])*(Hv[L])
        CW[L]=C*Sub[L]*Hruo[L]*Hu[L]
        CE[L]=C*Sub[L+1]*Hu[L+1]
        CN[L]=C*Svb[LN]*Hrvo[LN]*Hv[LN]
        CC[L]=Dxyp[L]/DT-CS[L]-CW[L]-CE[L]-CN[L]
        L+=1
    ###########################################
    gb.set_value('CS',CS)
    gb.set_value('CW',CW)
    gb.set_value('CN',CN)
    gb.set_value('CE',CE)
    gb.set_value('FP',FP)
    Cal_OBC()
    CS=gb.get_value('CS')
    CW=gb.get_value('CW')
    CN=gb.get_value('CN')
    CE=gb.get_value('CE')
    FP=gb.get_value('FP')
    ###################################
    FPTMP=np.arange(LC)
    CCMMM=1000000000000000000
    FPTMP=FP.copy()
    CCMMM=min(CCMMM,np.min(CC[1:-1]))
    CCMMMI=1/CCMMM
    ############################
    CCS=CS*CCMMMI
    CCW=CW*CCMMMI
    CCE=CE*CCMMMI
    CCN=CN*CCMMMI
    CCC=CC*CCMMMI
    FPTMP=FPTMP*CCMMMI
    CCCI=1.0/CCC
    CCCI[0]=0
    CCCI[-1]=0
    #print(CCS)
    #print(CCW)
    ##print(CCE)
   # print(CCN)
   # print(CCC)
   # print(FPTMP)
   # #666到此一游
    gb.set_value('CCS',CCS)
    gb.set_value('CCW',CCW)
    gb.set_value('CCE',CCE)
    gb.set_value('CCN',CCN)
    gb.set_value('CCC',CCC)
    gb.set_value('FPTMP',FPTMP)
    gb.set_value('CCCI',CCCI)
    #########################################
    ####梯度下降法求解
    Cal_congrad(LC)
    P=gb.get_value('P')
   # print(P)
    UHE=gb.get_value('UHE')
    VHE=gb.get_value('VHE')
    L=1
    for i in range(LC-2):
        LS=LSC[L]
        UHDYE[L]=Sub[L]*(FUHDYE[L]-DT2*Hruo[L]*Hu[L]*(P[L]-P[L-1]))
        VHDXE[L]=Svb[L]*(FVHDXE[L]-DT2*Hrvo[L]*Hv[L]*(P[L]-P[LS]))
        UHE[L]=UHDYE[L]/Dyv[L]
        VHE[L]=VHDXE[L]/Dxu[L]
        L+=1
    L=1
    for i in range(LC-2):
        LN=LNC[L]
        Hp[L]=H1p[L]+DT2/Dxyp[L]*(2*QSUME[L]-(UHDYE[L+1]+UHDY1E[L+1]-UHDYE[L]-UHDY1E[L]+VHDXE[LN]+VHDX1E[LN]-VHDXE[L]-VHDX1E[L]))
        L+=1
    Belv=gb.get_value('Belv')
    ###############
    ####边界
    NPSIJ=gb.get_value('NPSIJ')
    PINF=gb.get_value('PINF')
    for i in range(NPSIJ):
        L=int(PINF[4,i])
        Hp[L]=P[L]/9.81-Belv[L]
    P=9.81*(Hp+Belv)

    L=1
    for i in range(LC-2):
        LS=LSC[L]
        Hu[L]=0.5*(Dxyp[L]*Hp[L]+Dxyp[L-1]*Hp[L-1])/Dxyu[L]
        Hv[L]=0.5*(Dxyp[L]*Hp[L]+Dxyp[LS]*Hp[LS])/Dxyv[L]
        L+=1
    H1p=H2p.copy()
    gb.set_value('Hu',Hu)
    gb.set_value('Hv',Hv)
    gb.set_value('H1p',H1p)
    gb.set_value('H2p',H2p)
    gb.set_value('P',P)
    gb.set_value('UHE',UHE)
    gb.set_value('VHE',VHE)
    gb.set_value('UHDYE',UHDYE)
    gb.set_value('VHDXE',VHDXE)

@jit
def Cal_congrad(LC):
    '''
        梯度下降法求解
    '''
    P=gb.get_value('P')
    CCN=gb.get_value('CCN')
    CCW=gb.get_value('CCW')
    CCE=gb.get_value('CCE')
    CCS=gb.get_value('CCS')
    CCC=gb.get_value('CCC')
    CCCI=gb.get_value('CCCI')
    FPTMP=gb.get_value('FPTMP')
    LNC=gb.get_value('LNC')
    LSC=gb.get_value('LSC')
    PN=np.arange(LC,dtype=np.float)
    PN[:]=0
    PS=np.arange(LC,dtype=np.float)
    PS[:]=0
    RCG=np.arange(LC,dtype=np.float)
    RCG[:]=0
    ##############################
    L=1
    for i in range(LC-2):
        PN[L]=P[LNC[L]]
        PS[L]=P[LSC[L]]
        L+=1

    L=1
    for i in range(LC-2):
        RCG[L]=FPTMP[L]-CCC[L]*P[L]-CCN[L]*PN[L]-CCS[L]*PS[L]-CCW[L]*P[L-1]-CCE[L]*P[L+1]
        L+=1
    PCG=RCG*CCCI
    RPCG=np.sum(PCG*RCG)
    if(RPCG==0.0):
        return
    ITER=0
    APCG=np.arange(LC,dtype=np.float)
    APCG[:]=0
    while True:
        L=1
        for i in range(LC-2):
            PN[L]=PCG[LNC[L]]
            PS[L]=PCG[LSC[L]]
            L+=1
        L=1
        for i in range(LC-2):
            APCG[L]=CCC[L]*PCG[L]+CCS[L]*PS[L]+CCN[L]*PN[L]+CCW[L]*PCG[L-1]+CCE[L]*PCG[L+1]
            L+=1
        PAPCG=np.sum(APCG*PCG)
        ALPHA=RPCG/PAPCG
        P=P+ALPHA*PCG
        L=1
        RCG=RCG-ALPHA*APCG
        TMPCG=CCCI*RCG
        RPCGN=0.0
        RPCGN=np.sum(RCG*TMPCG)
        RSQ=np.sum(RCG*RCG)
        if(RSQ<1E-10):
            break
        BETA=RPCGN/RPCG
        RPCG=RPCGN
        PCG=TMPCG+BETA*PCG
    gb.set_value('P',P)

@jit
def Cal_CDZC(KC):
    DZC=gb.get_value('DZC')
    CDZD=gb.get_value('CDZD')
    CDZR=gb.get_value('CDZR')
    CDZR[0]=DZC[0]-1
    CDZD[0]=DZC[0]
    for k in range(1,KC-1):
        CDZR[k]=DZC[k]+CDZR[k-1]
        CDZD[k]=DZC[k]+CDZD[k-1]
    for k in range(KC-1):
        CDZR[k]=CDZR[k]*(0.5*(DZC[k]+DZC[k+1]))*(-DZC[1]/(DZC[0]+DZC[1]))
    gb.set_value('CDZR',CDZR)
    gb.set_value('CDZD',CDZD)

@jit
def Reset_2V():
    '''
        倒腾一下变量
    '''
    UHDY1=gb.get_value('UHDY1')
    UHDY=gb.get_value('UHDY')
    VHDX1=gb.get_value('VHDX1')
    VHDX=gb.get_value('VHDX')
    U1=gb.get_value('U1')
    V1=gb.get_value('V1')
    W1=gb.get_value('W1')
    U=gb.get_value('U')
    V=gb.get_value('V')
    W=gb.get_value('W')

    UHDY2=UHDY1.copy()
    UHDY1=UHDY.copy()
    VHDX2=VHDX1.copy()
    VHDX1=VHDX.copy()
    U2=U1.copy()
    U1=U.copy()
    V2=V1.copy()
    V1=V.copy()
    W2=W1.copy()
    W1=W.copy()
    gb.set_value('UHDY2',UHDY2)
    gb.set_value('UHDY1',UHDY1)
    gb.set_value('UHDY',UHDY)
    gb.set_value('VHDX',VHDX)
    gb.set_value('VHDX1',VHDX1)
    gb.set_value('VHDX2',VHDX2)
    gb.set_value('U',U)
    gb.set_value('U1',U1)
    gb.set_value('U2',U2)
    gb.set_value('V',V)
    gb.set_value('V1',V1)
    gb.set_value('V2',V2)
    gb.set_value('W',W)
    gb.set_value('W1',W1)
    gb.set_value('W2',W2)

@jit
def Cal_UVW(IC,JC,KC,LC):
    DZC=gb.get_value('DZC')
    DT=gb.get_value('DT')
    AVO=gb.get_value('AVO')
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    Du=gb.get_value('Du')
    Dv=gb.get_value('Dv')
    RCX=gb.get_value('RCX')
    RCY=gb.get_value('RCY')
    U1=gb.get_value('U1')
    V1U=gb.get_value('V1U')
    U=gb.get_value('U')
    VU=gb.get_value('VU')
    UV=gb.get_value('UV')
    V=gb.get_value('V')
    V1=gb.get_value('V1')
    U1V=gb.get_value('U1V')
    STBX=gb.get_value('STBX')
    STBY=gb.get_value('STBY')
    ###########################
    L=1
    for i in range(LC-2):
        Q1=math.sqrt(U1[L,0]*U1[L,0]+V1U[L]*V1U[L])
        Q2=math.sqrt(U[L,0]*U[L,0]+VU[L]*VU[L])
        RCX[L]=STBX[L]*math.sqrt(Q1*Q2)
        Q1=math.sqrt(U1V[L]*U1V[L]+V1[L,0]*V1[L,0])
        Q2=math.sqrt(UV[L]*UV[L]+V[L,0]*V[L,0])
        RCY[L]=STBY[L]*math.sqrt(Q1*Q2)
        L+=1

    ######################
    CU1=gb.get_value('CU1')
    CU2=gb.get_value('CU2')
    UHE=gb.get_value('UHE')
    VHE=gb.get_value('VHE')
    UUU=gb.get_value('UUU')
    VVV=gb.get_value('VVV')
    CDZM=DZC[0]*DZC[1]*0.5/DT
    CDZU=-DZC[0]/(DZC[0]+DZC[1])
    CDZL=-DZC[1]/(DZC[0]+DZC[1])
    L=1
    for i in range(LC-2):
        CMU=1.0+CDZM*Hu[L]/AVO
        CMV=1.0+CDZM*Hv[L]/AVO
        EU=1.0/CMU
        EV=1.0/CMV
        CU1[L,0]=CDZU*EU
        CU2[L,0]=CDZU*EV
        Du[L,0]=(Du[L,0]-CDZL*RCX[L]*UHE[L]/Hu[L])*EU
        Dv[L,0]=(Dv[L,0]-CDZL*RCY[L]*VHE[L]/Hv[L])*EV
        UUU[L,0]=EU
        UUU[L,1]=EV
        L+=1

    L=1
    for i in range(LC-2):
        for k in range(1,KC-1):
            CDZM=DZC[k]*DZC[k+1]*0.5/DT
            CDZU=-DZC[k]/(DZC[k]+DZC[k+1])
            CDZL=-DZC[k+1]/(DZC[k]+DZC[k+1])
            CMU=1.0+CDZM*Hu[L]/AVO
            CMV=1.0+CDZM*Hv[L]/AVO
            EU=1.0/CMU
            EV=1.0/CMV
            CU1[L,k]=CDZU*EU
            CU2[L,k]=CDZU*EV
            Du[L,k]=(Du[L,k]-CDZL*RCX[L]*UHE[L]/Hu[L])*EU
            Dv[L,k]=(Dv[L,k]-CDZL*RCY[L]*VHE[L]/Hv[L])*EV
            UUU[L,k]=-CDZL*UUU[L,k-1]*EU
            VVV[L,k]=-CDZL*VVV[L,k-1]*EV
        L+=1
    L=1
    for i in range(LC-2):
        for k in range(KC-3,-1,-1):
            Du[L,k]=Du[L,k]-CU1[L,k]*Du[L,k+1]
            Dv[L,k]=Dv[L,k]-CU2[L,k]*Dv[L,k+1]
            UUU[L,k]=UUU[L,k]-CU1[L,k]*UUU[L,k+1]
            VVV[L,k]=VVV[L,k]-CU2[L,k]*VVV[L,k+1]
        L+=1

    AAU=gb.get_value('AAU')
    AAV=gb.get_value('AAV')
    BBU=gb.get_value('BBU')
    BBV=gb.get_value('BBV')
    AAU[:]=0
    AAV[:]=0
    BBU[1:-1]=1
    BBV[1:-1]=1
    CDZR=gb.get_value('CDZR')
    for k in range(KC-1):
        RCDZR=CDZR[k]
        L=1
        for i in range(LC-2):
            CRU=RCDZR*RCX[L]/AVO
            CRV=RCDZR*RCY[L]/AVO
            AAU[L]=AAU[L]+CRU*Du[L,k]
            AAV[L]=AAV[L]+CRV*Dv[L,k]
            BBU[L]=BBU[L]+CRU*UUU[L,k]
            BBV[L]=BBV[L]+CRV*VVV[L,k]
            L+=1
    L=1
    for i in range(LC-2):
        AAU[L]=AAU[L]/BBU[L]
        AAV[L]=AAV[L]/BBV[L]
        L+=1

    for k in range(KC-1):
        RDZG=0.5*(DZC[k]+DZC[k+1])
        L=1
        for i in range(LC-2):
            Du[L,k]=RDZG*Hu[L]/AVO*(Du[L,k]-AAU[L]*UUU[L,k])
            Dv[L,k]=RDZG*Hv[L]/AVO*(Dv[L,k]-AAV[L]*VVV[L,k])
            L+=1
    CDZD=gb.get_value('CDZD')
    for k in range(KC-1):
        RCDZD=CDZD[k]
        L=1
        for i in range(LC-2):
            UHE[L]=UHE[L]+RCDZD*Du[L,k]
            VHE[L]=VHE[L]+RCDZD*Dv[L,k]
            L+=1
    Sub=gb.get_value('Sub')
    Svb=gb.get_value('Svb')
    UHDY=gb.get_value('UHDY')
    VHDX=gb.get_value('VHDX')
    UHDY[:,KC-1]=UHE*Sub
    VHDX[:,KC-1]=VHE*Svb
    for k in range(KC-2,-1,-1):
        L=1
        for i in range(LC-2):
            UHDY[L,k]=UHDY[L,k+1]-Du[L,k]*Sub[L]
            VHDX[L,k]=VHDX[L,k+1]-Dv[L,k]*Svb[L]
            L+=1

    Dyu=gb.get_value('Dyu')
    Dxv=gb.get_value('Dxv')
    for k in range(KC):
        L=1
        for i in range(LC-2):
            U[L,k]=UHDY[L,k]/Hu[L]
            V[L,k]=VHDX[L,k]/Hv[L]
            UHDY[L,k]=UHDY[L,k]*Dyu[L]
            VHDX[L,k]=VHDX[L,k]*Dxv[L]
            L+=1
    #############################
    TVAR3E=np.arange(LC,dtype=np.float)
    TVAR3N=np.arange(LC,dtype=np.float)
    TVAR3E[:]=0
    TVAR3N[:]=0
    for k in range(KC):
        L=1
        for i in range(LC-2):
            TVAR3E[L]=TVAR3E[L]+UHDY[L,k]*DZC[k]
            TVAR3N[L]=TVAR3N[L]+VHDX[L,k]*DZC[k]
            L+=1
    UERMX=-1.0E+12
    UERMN=1.0E+12
    VERMX=-1.0E+12
    VERMN=1.0E+12
    UHDYE=gb.get_value('UHDYE')
    VHDXE=gb.get_value('VHDXE')
    L=1
    for i in range(LC-2):
        TVAR3E[L]=TVAR3E[L]-UHDYE[L]
        TVAR3N[L]=TVAR3N[L]-VHDXE[L]
        L+=1

    for k in range(KC):
        L=1
        for i in range(LC-2):
            UHDY[L,k]=UHDY[L,k]-TVAR3E[L]/DZC[k]
            VHDX[L,k]=VHDX[L,k]-TVAR3N[L]/DZC[k]
            L+=1
    #####################################
    #重置速度
    UHE[:]=0
    VHE[:]=0
    for k in range(KC):
        L=1
        for i in range(LC-2):
            UHE[L]=UHE[L]+UHDY[L,k]*DZC[k]
            VHE[L]=VHE[L]+VHDX[L,k]*DZC[k]
            U[L,k]=UHDY[L,k]/Hu[L]
            V[L,k]=VHDX[L,k]/Hv[L]
            L+=1
    for k in range(KC):
        L=1
        for i in range(LC-2):
            U[L,k]=U[L,k]/Dyu[L]
            V[L,k]=V[L,k]/Dxv[L]
            L+=1
    L=1
    for i in range(LC-2):
        UHE[L]=UHE[L]/Dyu[L]
        VHE[L]=VHE[L]/Dxv[L]
        L+=1
    #################################
    ####计算W
    W=gb.get_value('W')
    Dxyp=gb.get_value('Dxyp')
    UHDY1E=gb.get_value('UHDY1E')
    UHDY1=gb.get_value('UHDY1')
    VHDX1=gb.get_value('VHDX1')
    VHDX1E=gb.get_value('VHDX1E')
    QSUM=gb.get_value('QSUM')
    QSUME=gb.get_value('QSUME')
    LNC=gb.get_value('LNC')

    for k in range(KC-1):
        L=1
        for i in range(LC-2):
            LN=LNC[L]
            LE=L+1
            if(k==0):
                WW=0
            else:
                WW=W[L,k-1]
            W[L,k]=WW-0.5*DZC[k]/Dxyp[L]*(UHDY[LE,k]-UHDY[L,k]-UHDYE[LE]+UHDYE[L]+UHDY1[LE,k]-UHDY1[L,k]-UHDY1E[LE]+UHDY1E[L]+VHDX[LN,k]-VHDX[L,k]-VHDXE[LN]+VHDXE[L]+VHDX1[LN,k]-VHDX1[L,k]-VHDX1E[LN]+VHDX1E[L])+(QSUM[L,k]-DZC[k]*QSUME[L])/Dxyp[L]
            L+=1

    #########################
    PINF=gb.get_value('PINF')
    NPSIJ=gb.get_value('NPSIJ')
    for i in range(NPSIJ):
        L=int(PINF[4,i])
        for k in range(KC-1):
            W[L,k]=0
    ##############################
    UHDY2=gb.get_value('UHDY2')
    VHDX2=gb.get_value('VHDX2')
    U2=gb.get_value('U2')
    V2=gb.get_value('V2')
    W2=gb.get_value('W2')
    U1=gb.get_value('U1')
    V1=gb.get_value('V1')
    W1=gb.get_value('W1')
    for k in range(KC):
        L=1
        for i in range(LC-2):
            UHDY2[L,k]=0.5*(UHDY[L,k]+UHDY1[L,k])
            VHDX2[L,k]=0.5*(VHDX[L,k]+VHDX1[L,k])
            U2[L,k]=0.5*(U[L,k]+U1[L,k])
            V2[L,k]=0.5*(V[L,k]+V1[L,k])
            W2[L,k]=0.5*(W[L,k]+W1[L,k])
            L+=1
    gb.set_value('U2',U2)
    gb.set_value('V2',V2)
    gb.set_value('W2',W2)
    gb.set_value('U1',U1)
    gb.set_value('V1',V1)
    gb.set_value('W1',W1)
    gb.set_value('U',U)
    gb.set_value('V',V)
    gb.set_value('W',W)
    gb.set_value('UHE',UHE)
    gb.set_value('VHE',VHE)
    gb.set_value('UHDY2',UHDY2)
    gb.set_value('VHDX2',VHDX2)
    gb.set_value('UHDY',UHDY)
    gb.set_value('VHDX',VHDX)
    gb.set_value('UHDY1',UHDY1)
    gb.set_value('VHDX1',VHDX1)
    gb.set_value('UHDYE',UHDY1E)
    gb.set_value('VHDXE',VHDXE)
    gb.set_value('UHDY1E',UHDY1E)
    gb.set_value('VHDX1E',VHDX1E)
    ############################
    L=1
    TVAR3E[:]=0
    TVAR3N[:]=0
    for k in range(KC):
        L=1
        for i in range(LC-2):
            TVAR3E[L]=TVAR3E[L]+UHDY2[L,k]*DZC[k]
            TVAR3N[L]=TVAR3N[L]+VHDX2[L,k]*DZC[k]
            L+=1
    #################################
    L=1
    Hp=gb.get_value('Hp')
    H1p=gb.get_value('H1p')
    P=gb.get_value('P')
    Belv=gb.get_value('Belv')
    SPB=gb.get_value('SPB')
    for i in range(LC-2):
        LN=LNC[L]
        HPPTMP=H1p[L]+DT/Dxyp[L]*(QSUME[L]-TVAR3E[L+1]+TVAR3E[L]-TVAR3N[LN]+TVAR3N[L])
        Hp[L]=SPB[L]*HPPTMP+(1.0-SPB[L])*(1/9.81*P[L]-Belv[L])
        L+=1
###########################################
    gb.set_value('Hp',Hp)

@jit
def Reset_1V():
    '''
        倒腾一下变量
    '''
    Hp=gb.get_value('Hp')
    Hu=gb.get_value('Hu')
    Hv=gb.get_value('Hv')
    UHDYE=gb.get_value('UHDYE')
    VHDXE=gb.get_value('VHDXE')
    U=gb.get_value('U')
    V=gb.get_value('V')
    UHDY=gb.get_value('UHDY')
    VHDX=gb.get_value('VHDX')

    H1p=Hp.copy()
    H1u=Hu.copy()
    H1v=Hv.copy()
    UHDY1E=UHDYE.copy()
    VHDX1E=VHDXE.copy()
    U1=U.copy()
    V1=V.copy()
    UHDY1=UHDY.copy()
    VHDX1=VHDX.copy()

    gb.set_value('H1p',H1p)
    gb.set_value('H1u',H1u)
    gb.set_value('H1v',H1v)
    gb.set_value('UHDY1E',UHDY1E)
    gb.set_value('VHDX1E',VHDX1E)
    gb.set_value('U1',U1)
    gb.set_value('V1',V1)
    gb.set_value('UHDY1',UHDY1)
    gb.set_value('VHDX1',VHDX1)
