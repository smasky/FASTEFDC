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

def Cal_LCT_IL_JL(IC,JC,LC):
    '''
    计算LCT,IL,JL LC网格个数 Dxp Dyp Hmp
    return:LCT IL JL LIJ LSC LNC
    '''
    Cell=gb.get_value('Cell')
    CCell=gb.get_value('CCell')
    Dxp=gb.get_value('Dxp')
    Dxp[:]=0
    Dyp=gb.get_value('Dyp')
    Dyp[:]=0
    Hmp=gb.get_value('Hmp')
    Hmp[:]=0
    Belv=gb.get_value('Belv')
    Belv[:]=0
    Zbr=gb.get_value('Zbr')
    Zbr[:]=0
    IL=gb.get_value('IL')
    JL=gb.get_value('JL')
    LCT=gb.get_value('LCT')
    LIJ=gb.get_value('LIJ')
    Dxpl=np.arange(LC,dtype=np.float)
    Dypl=np.arange(LC,dtype=np.float)
    Hmpl=np.arange(LC,dtype=np.float)
    Belvl=np.arange(LC,dtype=np.float)
    Zbrl=np.arange(LC,dtype=np.float)
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
    print(L)
    L=1
    for i in range(LC-1):
        ii=IL[i]
        jj=JL[i]
        NValue=CCell[ii+1][jj]
        SValue=CCell[ii-1][jj]
        NWValue=CCell[ii+1][jj-1]
        SEValue=CCell[ii-1][jj+1]
        SWValue=CCell[ii-1][jj-1]
        NEValue=CCell[ii+1][jj+1]
        if(NValue==9):
            LNC[L]=LC
        else:
            LNC[L]=LIJ[ii+1][jj]
        if(SValue==9):
            LSC[L]=LC
        else:
            LSC[L]=LIJ[ii-1][jj]
        if(NWValue==9):
            LNWC[L]=LC
        else:
            LNWC[L]=LIJ[ii+1][jj-1]
        if(SEValue==9):
            LSEC[L]=LC
        else:
            LSEC[L]=LIJ[ii-1][jj+1]
        if(SWValue==9):
            LSWC[L]=LC
        else:
            LSWC[L]=LIJ[ii-1][jj-1]
        if(NEValue==9):
            LNEC[L]=LC
        else:
            LNEC[L]=LIJ[ii+1][jj+1]
        L+=1
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


def Cal_Dx_Dy_HM(IC,JC,LC):
    '''
        计算Dxu Dyu Dxv Dyv Hu Hv
    '''
    Dxp=gb.get_value('Dxp')
    Dyp=gb.get_value('Dyp')
    LSC=gb.get_value('LSC')
    Hp=gb.get_value('Hp')
    Dxu=np.arange(LC)
    Dxu[:]=0
    Dxv=np.arange(LC)
    Dxv[:]=0
    Dyu=np.arange(LC)
    Dyu[:]=0
    Dyv=np.arange(LC)
    Dyv[:]=0
    Hu=np.arange(LC)
    Hu[:]=0
    Hv=np.arange(LC)
    Hv[:]=0
    for i in range(LC-1):
        LS=LSC(i+1)
        Dxu[i+1]=0.5*(Dxp[i+1]+Dxp[i])
        Dyu[i+1]=0.5*(Dyp[i+1]+Dyp[i])
        Dxv[i+1]=0.5*(Dxp[i+1]+Dxp[LS])
        Dyv[i+1]=0.5*(Dyp[i+1]+Dyp[LS])
        Hu=0.5*(Dxp[i+1]*Dyp[i+1]*Hp[i+1]+Dxp[i]*Dyp[i]*Hp[i])/(Dxu[i+1]*Dyu[i+1])
        Hv=0.5*(Dxp[i+1]*Dyp[i+1]*Hp[i+1]+Dxp[LS]*Dyp[LS]*Hp[LS])/(Dxv[i+1]*Dyv[i+1])
    Hu[0]=Hu[1]
    Hv[0]=Hv[1]
    Hu[-1]=Hu[-2]
    Hv[-1]=Hv[-2]
    Hpi=1.0/Hp
    Hui=1.0/Hu
    Hvi=1.0/Hv
    Hv1i=Hvi.copy()
    Hu1i=Hui.copy()
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
    Sub=np.arange(LC)
    Svb=np.arange(LC)
    IL=gb.get_value('IL')
    JL=gb.get_value('JL')
    LCT=gb.get_value('LCT')
    Cell=gb.get_value('Cell')
    for i in range(LC):
        ii=IL[L]
        jj=JL[L]
        LCTN=LCT[L]
        if(LCTN==5):
            if(Cell[ii-1][jj]==5):
                Sub[L]=1
            else:
                Sub[L]=0
            if(Cell[ii][jj-1]==5):
                Svb[L]=1
            else:
                Svb[L]=0
    Sub[0]=0
    Svb[0]=0
    Sub[-1]=0
    Svb[-1]=0
    Subo=Sub.copy() #Sub origin
    Svbo=Svb.copy() #Svb origin

    gb.set_value('Sub',Sub)
    gb.set_value('Svb',Svb)
    gb.set_value('Subo',Subo)
    gb.set_value('Svbo',Svbo)


@jit
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
    for i in range(LC):
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
            Dyp[L]=Dxp[L]-0.5*Dyp[L]*DDXDDDY
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
    CCell=gb.get_value('Cell')
    IL=gb.get_value('IL')
    JL=gb.get_value('JL')
    L=1
    for i in range(LC):
        LN=LNC(L)
        LS=LSC(L)
        ii=IL(L)
        jj=JL(L)
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
    Hrv=Svb*Hv*Dxv/Dyv
    Hruo=Subo*Hu*Dyu/Dxu
    Hrvo=Svbo*Hv*Dxv/Dyv
    Dxyu=Dxu*Dyu
    Dxyv=Dxv*Dyv
    Dxyp=Dxp*Dyp
    Hrxyv=np.arange(LC)
    Hrxyu=np.arange(LC)
    Hrxyv[-1]=0
    Hrxyu[-1]=0
    Hrxyv=Dxu/Dyu
    Hrxyu=Dxv/Dyv
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
    U1V=gb.get_value('U1V')
    V1U=gb.get_value('V1U')
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
    L=1
    for i in range(LC):
        LN=LNC[L]
        LS=LSC[L]
        LSE=LSEC[L]
        LNW=LNWC[L]
        LSW=LSWC[L]
        UV[L]=0.25*(Hp[LS])*(U[LSE,0]+U[LS,0])+Hp[L]*(U[L+1,0]+U[L,0])*Hvi[L]
        U1V[L]=0.25*(H1p[LS])*(U1[LSE,0]+U1[LS,0])+H1p[L]*(U1[L+1,0]+U1[L,0])*Hv1i[L]
        VU[L]=0.25*(Hp[L-1]*(V[LNW,0]+V[L-1,0])+Hp[L]*(V[LN,0]+V[L,1]))*Hui[L]
        V1U[L]=0.25*(H1p[L-1]*(V1[LNW,0]+V1[L-1,0])+H1p[L]*(V1[LN,0]+V1[L,0]))*Hu1i[L]
    gb.set_value('UV',UV)
    gb.set_value('U1V',U1V)
    gb.set_value('VU',VU)
    gb.set_value('V1U',V1U)

def Cal_init_STBXY(IC,JC,KC,LC):
    '''
        STBXY的初始化
    '''
    STBX=np.arange(LC)
    STBX[:]=1
    STBY=np.arange(LC)
    STBY[:]=1
    RITB1=1.0#
    RITB=0.0#
    CDLIMIT=0.5#三个系数
    gb.set_value('STBX',STBX)
    gb.set_value('STBY',STBY)
    gb.set_value('RITB1',RITB1)
    gb.set_value('RITB',RITB)
    gb.set_value('CDLIMIT',CALIMIT)

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
    for i in range(LC):
        LS=LSC[L]
        ZBRATU=0.5*(Dxp[L-1]*Zbr[L-1]+Dxp[L]*Zbr[L])/Dxu[L]
        ZBRATV=0.5*(Dyp[LS]*Zbr[LS]+Dyp[L]*Zbr[L])/Dyv[L]
        UMAGTMP=np.sqrt(U1[L,0]*U1[L,0]+V1U[L,0]*V1U[L,0])
        VMAGTMP=np.sqrt(U1V[L,0]*U1V[L,0]+V1[L,0]*V1[L,0])
        CDMAXU=CDLIMIT*H1u[L]/(DT*UMAGTMP)
        CDMAXV=CDLIMIT*H1v[L]/(DT*VMAGTMP)
        HURTMP=max(ZBRATU,H1u[L])
        HVRTMP=max(ZBRATV,H1v[L])
        DZHUDZBR=1.0+0.5*DZC[0]*HURTMP/ZBRATU
        DZHVDZBR=1.0+0.5*DZC[0]*HVRTMP/ZBRATV
        STBX[L]=0.16/(np.power((np.log(DZHUDZBR)),2))
        STBY[L]=0.16/(np.power((np.log(DZHVDZBR)),2))
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
    for i in range(LC):
        TBX[L]=(AVCON1/Hu[L]+STBX[L]*np.sqrt(VU[L]*VU[L]+U[L,0]*U[L,0]))*U[L,0]
        TBY[L]=(AVCON1/Hv[L]+STBY[L]*np.sqrt(UV[L]*UV[L]+V[L,0]*V[L,0]))*V[L,0]
        L+=1
    gb.set_value('TBX',TBX)
    gb.set_value('TBY',TBY)

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

        for i in range(LC):
            LS=LSC[L]
            LW=L-1
            if(Sub(L)<0.5 and Sub(LS)<0.5):
                ICORDYU[L]=0
            if(Sub[L]>0.5 and Sub(LS)>0.5):
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
        gb.get_value('ICORDXV',ICORDXV)
        gb.get_value('ICORDYU',ICORDYU)
        ######################################
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
    for i in range(LC):
        LN=LNC[L]
        LS=LSC[L]
        LW=L-1
        for k in range(KC):
            Dxu1[L,k]=Sub[L+1]*(U(L+1,k)-U(L,k))/Dxp[L]
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
        for i in range(LC):
            for k in range(KC):
                TMPVAL=AHD*Dxp[L]*Dyp[L]
                DSQR=Dxu1[L,k]*Dxu1[L,k]+Dyv1[L,k]*Dyv1[L,k]+Sxy[L,k]*Sxy[L,k]/4
                AH[L,k]=AHO+TMPVAL*np.sqrt(DSQR)
            L+=1
    elif(N<10):
        for i in range(LC):
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
    for i in range(LC):
        LS=LSC(L)
        LN=LNC(L)
        for k in range(KC):
            FMDUX[L,k]=(Dyp[L]*Hp[L]*AH[L,k]*Dxu1[L,k]-Dyp[L-1]*Hp[L-1]*AH[L-1,k]*Dxu1[L-1,k])*Sub[L]
            FMDUY[L,k]=(0.5*(Dxu[LN]+Dxu[L])*Hu[L]*(AH[LN,k]*Sxy[LN,k]-AH[L,k]*Sxy[L,k]))*Svb[LN]
            FMDVX[L,k]=0.5*(Dyv[L+1]+Dyv[L])*Hv[L]*(AH[L+1,k]*Sxy[L+1,k]-AH[L,k]*Sxy[L,k])*Sub[L+1]
            FMDVY=(Dxp[L]*Hp[L]*AH[L,k]*Dyv1[L,k]-Dxp[LS]*Hp[LS]*AH[LS,k]*Dyv1[LS,k])*Svb[L]
        L+=1
    gb.set_value('FMDUX',FMDUX)
    gb.set_value('FMDUY',FMDUY)
    gb.set_value('FMDVX',FMDVX)
    gb.set_value('FMDVY',FMDVY)
    ###################################################################
    #还有边界的FMDUX。。。。没有设置
    ##########################################


def Cal_init_Tsxy(IC,JC,KC):
    '''
    '''
    pass

def Cal_Exp2T(IC,JC,LC,KC):
    '''
        计算显示右边项
    '''
    FCAXE=gb.get_value('FCAXE')
    FCAYE=gb.get_value('FCAYE')
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
    for i in range(LC):
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
    for i in range(LC-1):
        LS=LSC(L)
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
    for i in range(LC-1):
        LN=LNC[L]
        LS=LSC[L]
        for k in range(KC):
            FX[L,k]=(FUHU[L,k]-FUHU[L-1,k]+FVHU[LN,k]-FVHU[L,k])#缺FUHJ SAAX大概边界取0
            FY[L,k]=FUHV[L+1,k]-FUHV[L,k]+FVHV[L,k]-FVHV[LS,k]#缺FVHJ
        L+=1
    ##################################
    #######水平动量扩散
    ##############################
    ###########计算FACXE
    L=1
    FMDUX=gb.get_value('FMDUX')
    FMDVX=gb.get_value('FMDVX')
    FMDUY=gb.get_value('FMDUY')
    FMDVY=gb.get_value('FMDVY')
    for i in range(LC-1):
        for k in range(KC):
            FX[L,k]=FX[L,k]-(FMDUX[L,k]+FMDUY[L,k])
            FY[L,k]=FY[L,k]-(FMDVX[L,k]+FMDVY[L,k])
        L+=1
    ########################################
    #计算FXE 外模态计算
    L=1
    DZC=gb.get_value('DZC')
    for i in range(LC-1):
        for k in range(KC):
            ###FCAXC
            FXE[L]=FXE[L]+FX[L,k]*DZC[k]
            FYE[L]=FYE[L]+FY[L,k]*DZC[k]
        L+=1
    #####################
    #内模块计算 覆盖了FX FY
    L=1
    for i in range(LC-1):
        for k in range(KC):
            FX[L,k]=FX[L,k]+(FWU[L,k]-FWU[L,k-1])/DZC[k]
            FY[L,k]=FY[L,k]+(FWV[L,k]-FWV[L,k-1])/DZC[k]
        L+=1
    ############################
    ###计算内部剪切力项
    #######################
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

def Cal_QVS(IC,JC,LC,KC):
    '''
        读取流量数据
    '''
    pass

def Cal_PSER(IC,JC,LC,KC):
    '''
        读取压力边界数据
    '''
    pass

def Cal_External(IC,JC,LC,KC):
    '''
        外模态计算水位 U V
    '''
    DT=gb.get_value('DT')
    LSC=gb.get_value('LSC')
    LNC=gb.get_value('LNC')
    H2p=gb.get_value('H2p')
    Hp=gb.get_value('Hp')
    H2p=Hp.copy()#将Hp赋予H2p
    FUHDYE=gb.get_value('FUHDYE')
    FVHDXE=gb.get_value('FVHDXE')
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
    Dxiu=gb.get_value('Dxiu')
    RITB1=gb.get_value('RITB1')
    TSX=gb.get_value('TSX')
    TSY=gb.get_value('TSY')
    TBX=gb.get_value('TBX')
    TBY=gb.get_value('TBY')
    Dxyu=gb.get_value('Dxyu')
    Dxyv=gb.get_value('Dxyv')
    Dyiv=gb.get_value('Dyiv')
    FXE=gb.get_value('FXE')
    FYE=gb.get_value('FYE')
    SNLT=gb.get_value('SNLT')
    P=gb.get_value('P')
    L=1
    for i in range(LC):
        LS=LSC[L]
        FUHDYE[L]=UHDYE[L]-DT2*Sub[L]*Hruo[L]*Hu[L]*(P[L]-P[L-1])+Sub[L]*DT*Dxiu[L]*(Dxyu[L]*(TSX[L]-RITB1*TBX[L])-SNLT*FXE[L])
        FVHDXE[L]=VHDXE[L]-DT2*Svb[L]*Hrvo[L]*Hv[L]*(P[L]-P[LS])+Svb[L]*DT*Dyiv[L]*(Dxyv[L]*(TSX[L]-RITB1*TBY[L])-SNLT*FYE[L])
        L+=1

    Subo=Sub.copy()
    Svbo=Svb.copy()
    QSUME=gb.get_value('QSUME')
    ##########################
    #####QSUME QSUMTMP
    P=gb.get_value('P')
    H1u=Hu.copy()
    H1v=Hv.copy()
    UHDY1E=UHDYE.copy()
    VHDX1E=VHDXE.copy()
    P1=P.copy()
    H1p=Hp.copy()
    Dxyp=gb.get_value('Dxyp')
    ####################
    FP1=np.arange(LC)
    FP=np.arange(LC)
    L=1
    for i in range(LC):
        LN=LNC[L]
        FP1[L]=Dxyp[L]*P[L]/DT-0.5*9.81*(UHDYE[L+1]-UHDYE[L]+VHDXE[LN]-VHDXE[L])
        L+=1

    L=1
    C=0.5*9.81
    for i in range(LC):
        LN=LNC[L]
        FP[L]=FP1[L]-C*(FUHDYE[L+1]-FUHDYE[L]+FVHDXE[LN]-FVHDXE[L]-2.0*QSUME[L])
        L+=1

    C=0.5*9.81*DT2
    L=1
    CS=np.arange(LC)
    CW=np.arange(LC)
    CE=np.arange(LC)
    CN=np.arange(LC)
    CC=np.arange(LC)
    for i in range(LC):
        LN=LNC(L)
        CS[L]=C*Svb[L]*Hrvo[L]*Hv[L]
        CW[L]=C*Sub[L]*Hruo[L]*Hu[L]
        CE[L]=C*Sub[L+1]*Hu[L+1]
        CN[L]=C*Svb[LN]*Hrvo[LN]*Hv[L+1]
        CC[L]=Dxyp[L]/DT-CS[L]-CW[L]-CE[L]-CN[L]
        L+=1
    ###########################################
    FPTMP=np.arange(LC)
    CCMMM=1000000000000000000
    FPTMP=FP.copy()
    CCMMM=max(CCMMM,np.min(CC))
    CCMMMI=1/CCMMM
    ############################
    CCS=CS*CCMMMI
    CCW=CW*CCMMMI
    CCE=CE*CCMMMI
    CCN=CN*CCMMMI
    CCC=CC*CCMMMI
    FPTMP=FPTMP*CCMMMI
    CCCI=1.0/CCC
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
    UHE=gb.get_value('UHE')
    VHE=gb.get_value('VHE')
    L=1
    for i in range(LC-1):
        LS=LSC[L]
        UHDYE[L]=Sub[L]*(FUHDYE[L]-DT2*Hruo[L]*Hu[L]*(P[L]-P[L-1]))
        VHDXE[L]=Svb[L]*(FVHDXE[L]-DT2*Hrvo[L]*Hv[L]*(P[L]-P[LS]))
        UHE[L]=UHDYE[L]*Dyiv[L]
        VHE[L]=VHDXE[L]*Dxiu[L]
        L+=1
    L=1
    for i in range(LC-1):
        LN=LNC[L]
        Hp[L]=H1p[L]+DT2/Dxyp[L]*(2*QSUME[L]-(UHDYE[L+1]+UHDY1E[L+1]-UHDYE[L]-UHDY1E[L]+VHDXE[LN]+VHDX1E[LN]-VHDXE[L]-VHDX1E[L]))
        L+=1
    Belv=gb.get_value('Belv')
    P=9.81*(Hp+Belv)

    L=1
    for i in range(LC):
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
    PN=np.arange(LC)
    PS=np.arange(LC)
    RCG=np.arange(LC)
    ##############################
    L=1
    for i in range(LC):
        PN[L]=P[LNC[L]]
        PS[L]=P[LSC[L]]
        L+=1

    L=1
    for i in range(LC):
        RCG[L]=FPTMP[L]-CCC[L]*P[L]-CCN[L]*PN[L]-CCS[L]*PS[L]-CCW*P[L-1]-CCE*P[L+1]
        L+=1
    PCG=RCG*CCCI
    RPCG=np.sum(PCG*RCG)
    if(RPCG==0.0):
        return
    ITER=0
    APCG=np.arange(LC)
    while True:
        L=1
        for i in range(LC-1):
            PN=PCG[LNC[L]]
            PS=PCG[LSC[L]]
            L+=1
        L=1
        for i in range(LC-1):
            APCG[L]=CCC[L]*PCG[L]+CCS[L]*PS[L]+CCN[L]*PN[L]+CCW*PCG[L-1]+CCE[L]*PCG[L+1]
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
        if(RSQ<0.000000001):
            break
        BETA=RPCGN/RPCG
        RPCG=RPCGN
        PCG=TMPCG+BETA*PCG
    gb.set_value('P',P)
