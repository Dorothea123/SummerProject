"""
Summer Research Project on Green Finance

Created on Mon Jun 17 15:29:53 2024

@author: dorothealangner
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf

tickers = sorted(['GDX','XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLRE','XLK','XLU'])

d = len(tickers)

#work with daily adjusted closing price 
data = yf.download(tickers,start='2021-06-01',end='2024-06-01')
print(data.head())

#plot daily adjusted closing price of selcted ETFs
def plot_prices():
    data['Adj Close'].plot(title='ETF price development').legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
    plt.show()
    
plot_prices()

#data2 = yf.download(tickers,start='2021-06-01',end='2024-06-01',interval='1wk')
#data3 = yf.download(tickers,start='2021-06-01',end='2024-06-01',interval='1mo')
#amount of entries for each year:
#252 - 250 - 251 => average 251


#get daily return data to yearly data
data_ret = data['Adj Close'].pct_change()
data_av_ret = data_ret.describe(include='all').loc['mean']
av_d_ret = data_av_ret.to_numpy()
av_ret = 251*av_d_ret
av_ret = av_ret.reshape((d,1))

#get daily covariance data
C_d = data['Adj Close'].cov().to_numpy()
cov_all = 251*C_d

'''
#get weekly return data to yearly data 
data_ret = data2['Adj Close'].pct_change()
data_av_ret = data_ret.describe(include='all').loc['mean']
av_d_ret = data_av_ret.to_numpy()
av_ret = 52*av_d_ret
av_ret = av_ret.reshape((d,1))

#get daily covariance data
C_d = data2['Adj Close'].cov().to_numpy()
#rho = data2['Adj Close'].corr().to_numpy()
#var_d = C_d.diagonal()
#D = np.diag(np.sqrt(52*var_d))
#cov_all = D @ rho @ D
cov_all = 52*C_d
'''
'''
#get monthly return data to yearly data 
data_ret = data3['Adj Close'].pct_change()
data_av_ret = data_ret.describe(include='all').loc['mean']
av_d_ret = data_av_ret.to_numpy()
av_ret = 12*av_d_ret
av_ret = av_ret.reshape((d,1))

#get monthly covariance data
C_d = data3['Adj Close'].cov().to_numpy()
#rho = data3['Adj Close'].corr().to_numpy()
#var_d = C_d.diagonal()
#D = np.diag(np.sqrt(12*var_d))
#cov_all = D @ rho @ D
cov_all = 12*C_d
'''
'''
#manual function to get yearly data (1 year interval not supported by yfinance)
#data only available from 2021-06-01 to 2024-05-31
data_y = data.loc[(data.index=='2021-06-01')|
                  (data.index=='2022-06-01')|(data.index=='2023-06-01')|
                  (data.index=='2024-05-31')]


#computations including all ETFs
#compute covariance matrix (on yearly basis)
C = data_y['Adj Close'].cov()

#compute yearly returns
data_y_ret = data_y['Adj Close'].pct_change()

#get mean and standard deviation of yearly returns
data_y_av_ret = data_y_ret.describe(include='all').loc['mean']

#compute average yearly return rates
av_ret = data_y_av_ret.to_numpy()
av_ret = av_ret.reshape((d,1))
cov_all = C.to_numpy()
'''

#do this for green portfolios, removed ETFs 
def green(noticker):
    def get_indices(lst, targets):
        return [index for index, element in enumerate(lst) if element in targets]
    indices = get_indices(tickers,noticker)
    tickers_green = [x for x in tickers if x not in noticker]
    data_green = data.drop(noticker,level=1,axis=1) #change with weekly/daily data
    cov_green = 251*data_green['Adj Close'].cov().to_numpy() #change to 52/12
    data_green_ret = data_green['Adj Close'].pct_change()
    av_green_ret = 251*data_green_ret.describe(include='all').loc['mean'].to_numpy() #change to 52/12
    dg = len(tickers_green)
    av_green_ret = av_green_ret.reshape((dg,1))
    return tickers_green,av_green_ret,cov_green,dg,indices

#check if covariance matrix is valid and invertible (i.e. positive definite)
#if not compute nearest positive definite matrix by A'=A+epsilon*Id

def is_posdef(A):
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def PD(A):
    if is_posdef(A):
        return A
    else:
        A = 1/2*(A+A.T)
        eigvals = list(np.linalg.eigvalsh(A))
        for i in range(len(eigvals)):
            if abs(eigvals[i])<=np.finfo(float).eps:
                eigvals[i]=0
        ev_min = min(list(filter(lambda num: num!=0,eigvals)))
        if ev_min < 0:
            A = A + (-ev_min+1e-15)*np.eye(len(A)) 
        else:
            A = A + 1e-15*np.eye(len(A))
        return A    

def PSD(A):
  E = np.linalg.eigvalsh(A)
  if np.all(E > -1e-15):
      return A
  else:
      B = 1/2*(A+A.T)
      w,v = np.linalg.eigh(B)
      D = np.diag(w)
      Q = v
      D_p = np.where(D<0,0,D)
      cov_near = Q @ D_p @ Q.T
      return cov_near

#compute Markowitz portfolio
def constants(avret,cov,d):
    cov = PD(PSD(cov))
    #use Pseudoinverse instead of inverse, uses SVD vs LU
    #SVD more expensive, but numerically more stable
    cov_inv = np.linalg.pinv(cov) 
    a = np.ones((1,d)) @ cov_inv @ avret
    a = a.item()
    b = avret.T @ cov_inv @ avret
    b = b.item()
    c = np.ones((1,d)) @ cov_inv @ np.ones((d,1))
    c = c.item()
    return a,b,c

def markowitz_portf(avret,cov,r,d):
    cov = PD(PSD(cov))
    cov_inv = np.linalg.pinv(cov) 
    a = constants(avret, cov,d)[0]
    b = constants(avret, cov,d)[1]
    c = constants(avret, cov,d)[2]
    w_0 = (b*np.ones((d,1))-a*avret)/(b*c-a**2)
    w_r = (c*avret-a*np.ones((d,1)))/(b*c-a**2)
    #the Markowitz portfolio:
    w_star = cov_inv@(w_0+r*w_r)   
    s_2 = c*((r-a/c)**2)/(b*c-a**2)+1/c #variance
    return (np.sqrt(s_2),w_star)

#efficient portfolio front equation
def effport(a,b,c,s):
    return np.sqrt((s**2-1/c)*(b*c-a**2)/c)+a/c

#equation as function of r
def inverse(a,b,c,r):
    return np.sqrt(c*(r-a/c)**2/(b*c-a**2)+1/c)

#plot efficient portfolio front
def plot_effport(avret,cov,d,color,label):
    a = constants(avret, cov,d)[0]
    b = constants(avret, cov,d)[1]
    c = constants(avret, cov,d)[2]
    s = np.linspace(0,50,num=1000)
    plt.plot(s,effport(a,b,c,s),color=color,linewidth=1,label=label)
    plt.title('Efficient portfolio front')
    plt.xlabel('risk sigma')
    plt.ylabel('expected return r')
    plt.legend()
    plt.show()

#plot both efficient portfolio fronts together
def plot_together(lists):
    a1 = constants(av_ret, cov_all,d)[0]
    b1 = constants(av_ret, cov_all,d)[1]
    c1 = constants(av_ret, cov_all,d)[2]
    s = np.linspace(0,50,num=1000)
    plt.plot(s,effport(a1,b1,c1,s),color='navy',linewidth=1,label='all')
    for i in range(len(lists)):
        a = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[0]
        b = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[1]
        c = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[2]
        string = ','.join(str(elm) for elm in lists[i])
        plt.plot(s,effport(a,b,c,s),linewidth=1,label='no '+string)
    plt.title('Efficient portfolio front')
    plt.xlabel('risk sigma')
    plt.ylabel('expected return r')
    plt.legend()
    plt.show()

#expected loss in return at fixed risk
def losses_s(s,lists):
    a1 = constants(av_ret, cov_all,d)[0]
    b1 = constants(av_ret, cov_all,d)[1]
    c1 = constants(av_ret, cov_all,d)[2]
    val = []
    for i in range(len(lists)):
        a = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[0]
        b = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[1]
        c = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[2]
        val.append(effport(a1,b1,c1,s)-effport(a,b,c,s))
    return val
    
#increase in risk at fixed expected return
def losses_r(r,lists): 
    a1 = constants(av_ret, cov_all,d)[0]
    b1 = constants(av_ret, cov_all,d)[1]
    c1 = constants(av_ret, cov_all,d)[2]
    losses = []
    for i in range(len(lists)):
        a = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[0]
        b = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[1]
        c = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[2]
        losses.append(inverse(a,b,c,r)-inverse(a1,b1,c1,r))
    return losses

#plot loss in r
def plot_loss_s(lists):
    s = np.linspace(0,100,num=1000)
    for i in range(len(lists)):
        string = ','.join(str(elm) for elm in lists[i])
        plt.plot(s,losses_s(s,lists)[i],linewidth=1,label='loss without '+string)
    plt.title('Losses in expected return at same risk')
    plt.xlabel('risk')
    plt.ylabel('loss in expected return')
    plt.legend()
    plt.show()
    
#plot increase in s
def plot_loss_r(lists):
    r = np.linspace(0,1,num=1000)
    for i in range(len(lists)):
        string = ','.join(str(elm) for elm in lists[i])
        plt.plot(r,losses_r(r,lists)[i],linewidth=1,label='increased risk without '+string)
    plt.title('Increase in risk at same expected return')
    plt.xlabel('expected return')
    plt.ylabel('increase in risk')
    plt.legend()
    plt.show()

def plot_investments(tickers,avret,cov,r,d):
    values = list(markowitz_portf(avret,cov,r,d)[1].flat)
    colors = ['g' if m > 0 else 'r' for m in values]
    plt.bar(tickers,values,width=.5,color=colors)
    plt.xlabel('ETFs')
    plt.ylabel('share in portfolio')
    plt.title('Markowitz portfolio for r= '+str(r))
    plt.show()

def plot_investments_together(r,lists):
    n = len(lists)+1
    values1 = list(markowitz_portf(av_ret,cov_all,r,d)[1].flat)
    x_axis = np.arange(len(tickers))
    plt.bar(x_axis - len(lists)/8,values1,color='navy',label='all',width=1/(n+1))
    for i in range(len(lists)):
        values = list(markowitz_portf(green(lists[i])[1],green(lists[i])[2],r,green(lists[i])[3])[1].flat)
        for j in range(len(green(lists[i])[4])):
            values.insert(green(lists[i])[4][j],0)
        string = ','.join(str(elm) for elm in lists[i])
        plt.bar(x_axis - len(lists)/8+(i+1)*1/(n+1),values,label='no '+string,width=1/(n+1))
    plt.xticks(x_axis,tickers)
    plt.xlabel('ETFs')
    plt.ylabel('share in portfolio')
    plt.title('Markowitz portfolios for r= '+str(r))
    plt.legend()
    plt.show()


#plot results
print('risk,portfolio')
print(markowitz_portf(av_ret, cov_all, 0.1,d))
print(markowitz_portf(green(['XLE'])[1],green(['XLE'])[2], 0.1,green(['XLE'])[3]))
print(markowitz_portf(green(['XLE','GDX'])[1],green(['XLE','GDX'])[2], 0.1,green(['XLE','GDX'])[3]))
plot_effport(av_ret,cov_all,d,'navy','all')
plot_effport(green(['XLE'])[1],green(['XLE'])[2],green(['XLE'])[3],'green','no XLE')
plot_effport(green(['XLE','GDX'])[1],green(['XLE','GDX'])[2],green(['XLE','GDX'])[3],'green','no XLE, GDX')
plot_together([['XLE'],['XLE','GDX']])
plot_loss_s([['XLE'],['XLE','GDX']])
plot_loss_r([['XLE'],['XLE','GDX']])
plot_investments(tickers,av_ret, cov_all, 0.1, d)
plot_investments(green(['XLE'])[0],green(['XLE'])[1],green(['XLE'])[2],0.1,green(['XLE'])[3])
plot_investments(green(['XLE','GDX'])[0],green(['XLE','GDX'])[1],green(['XLE','GDX'])[2],0.1,green(['XLE','GDX'])[3])
plot_investments_together(0.1,[['XLE']])
plot_investments_together(0.1, [['XLE'],['XLE','GDX']])



#create ESG dataframe
#data from 26-06-2024
data_gdx = pd.DataFrame([['NEM',0.1373,21.4],['AEM',0.0936,21.1],['GOLD',0.0848,29.5],['WPM',0.0688,7.1],
            ['FNV',0.0655,6.7],['GFI',0.0470,23.9],['2899.HK',0.0418,36.7],['AU',0.0358,24.4],
            ['KGC',0.0335,23.7],['NST.AX',0.0343,31.0],['RGLD',0.0285,8.5],['PAAS',0.0253,24.2],
            ['AGI',0.0219,35.1],['HMY',0.0198,32.9],['EDV CN',0.0178,17.3],['EVN.AX',0.0155,27.4],
            ['1818.HK',0.0161,48.3],['BVN',0.0147,41.4],['BTG',0.0119,24.2],['HL',0.0105,32.6],
            ['EGO',0.0104,20.0],['OR',0.0099,10.0],['CDE',0.0077,32.0],['PRU.AX',0.0074,29.4],
            ['EQX.TO',0.0070,30.1],['IMG.TO',0.0064,30.5],['CEY.L',0.0061,23.8],['AG.TO',0.0060,30.1],
            ['RED.AX',0.0057,42.6],['OGC.TO',0.0056,30.3],['SSL.TO',0.0055,12.0],['NGD.TO',0.0053,28.4],
            ['FVI.TO',0.0052,25.0],['RMS.AX',0.0051,33.2],['CG.TO',0.0051,30.4],['DPM.TO',0.0050,29.1],
            ['RMS.AX',0.0049,33.2],['BGL.AX',0.0047,31.6],['GMD.AX',0.0045,np.nan],['TXG.TO',0.0045,26.7],
            ['KNT.TO',0.0045,38.4],['AYA.TO',0.0044,38.5],['SIL.TO',0.0042,47.0],['WDO.TO',0.0042,32.0],
            ['MAG.TO',0.0042,28.6],['GOR.AX',0.0041,22.3],['CMM.AX',0.0041,53.2],['WAF.AX',0.0037,32.0],
            ['CXB.TO',0.0035,29.0],['SSRM.TO',0.0032,32.5],['RRL.AX',0.0030,34.0],['EXK',0.0029,23.5],
            ['WGX.AX',0.0027,37.1],['DRD',0.0026,32.4]],columns=['ticker','share','ESG'])

#data from 25-06-2024
data_xlc = pd.DataFrame([['META',0.22723,32.7],['GOOGL',0.12697,24.8],['GOOG',0.10652,24.8],['T',0.04651,22.1],
            ['VZ',0.04546,18.2],['EA',0.04540,13.3],['CMCSA',0.04502,22.6],['DIS',0.04488,15.0],
            ['TMUS',0.04431,25.0],['NFLX',0.04410,15.5],['CHTR',0.04221,23.7],['TTWO',0.03743,16.0],
            ['OMC',0.02606,14.5],['WBD',0.02393,18.1],['LYV',0.01897,21.5],['IPG',0.01631,8.7],
            ['NWSA',0.01566,10.0],['MTCH',0.01191,16.7],['FOXA',0.01186,12.2],['PARA',0.00741,14.5],
            ['FOX',0.00630,12.2],['NWS',0.00487,10.0]],columns=['ticker','share','ESG'])

data_xle = pd.DataFrame([['XOM',0.22967,41.3],['CVX',0.17947,35.3],['EOG',0.04768,34.4],['SLB',0.04738,19.2],
            ['COP',0.04506,33.1],['MPC',0.04416,30.3],['PSX',0.04333,33.0],['WMB',0.03810,21.2],
            ['VLO',0.03665,30.5],['OKE',0.03447,25.0],['OXY',0.03058,37.7],['HES',0.03020,32.0],
            ['KMI',0.02829,17.8],['FANG',0.02572,37.3],['BKR',0.02535,19.4],['HAL',0.02234,23.9],
            ['DVN',0.02182,31.6],['TRGP',0.02098,31.9],['CTRA',0.01500,32.7],['EQT',0.01241,31.7],
            ['MRO',0.01185,38.7],['APA',0.00761,42.9]],columns=['ticker','share','ESG'])

data_xlf = pd.DataFrame([['BRK.B',0.13011,27.3],['JPM',0.09947,27.3],['V',0.07530,15.0],['MA',0.06542,15.6],
            ['BAC',0.04685,24.3],['WFC',0.03487,35.9],['GS',0.02579,24.2],['SPGI',0.02512,11.5],
            ['AXP',0.02291,18.3],['MS',0.02140,24.8],['PGR',0.02133,19.8],['C',0.02054,22.1],
            ['BLK',0.01940,18.4],['SCHW',0.01929,23.4],['CB',0.01870,22.4],['MMC',0.01829,21.5],
            ['BX',0.01555,23.9],['FI',0.01523,17.7],['ICE',0.01377,18.6],['KKR',0.01250,22.0],
            ['CME',0.01222,17.1],['MCO',0.01174,14.6],['AON',0.01113,15.3],['USB',0.01092,24.9],
            ['PYPL',0.01085,16.4],['PNC',0.01057,23.7],['AJG',0.00992,20.5],['COF',0.00901,21.3],
            ['TFC',0.00867,17.3],['AIG',0.00865,24.1],['TRV',0.00836,20.5],['AFL',0.00814,17.7],
            ['BK',0.00769,19.0],['AMP',0.00761,17.9],['MET',0.00746,15.1],['ALL',0.00745,20.9],
            ['PRU',0.00745,18.6],['FIS',0.00732,14.4],['MSCI',0.00677,14.4],['ACGL',0.00667,21.1],
            ['DFS',0.00554,22.7],['HIG',0.00536,14.9],['WTW',0.00470,16.9],['TROW',0.00458,17.7],
            ['MTB',0.00430,25.3],['FITB',0.00430,16.9],['GPN',0.00427,19.5],['NDAQ',0.00399,13.2],
            ['RJF',0.00397,26.7],['STT',0.00382,23.2],['BRO',0.00377,21.1],['CPAY',0.00332,22.5],
            ['HBAN',0.00325,16.6],['CINF',0.00317,23.1],['SYF',0.00317,16.5],['CBOE',0.00311,21.5],
            ['RF',0.00305,14.9],['PFG',0.00302,11.4],['NTRS',0.00294,24.9],['EG',0.00292,np.nan],
            ['WRB',0.00286,21.3],['FDS',0.00280,16.5],['CFG',0.00278,23.0],['L',0.00242,29.1],
            ['KEY',0.00225,22.4],['JKHY',0.00213,18.7],['AIZ',0.00152,23.5],['MKTX',0.00130,13.9],
            ['GL',0.00123,np.nan],['BEN',0.00121,19.5],['IVZ',0.00118,20.9]],columns=['ticker','share','ESG'])

data_xli = pd.DataFrame([['GE',0.04676,34.5],['CAT',0.04282,29.1],['UBER',0.03988,23.2],['HON',0.03734,27.1],
            ['UNP',0.03656,20.0],['RTX',0.03611,29.6],['ETN',0.03451,18.1],['ADP',0.02723,15.1],
            ['BA',0.02702,36.6],['LMT',0.02673,28.6],['UPS',0.02621,18.8],['DE',0.02558,16.0],
            ['WM',0.02074,18.8],['TT',0.02031,15.1],['TDG',0.01976,38.2],['GD',0.01795,33.9],
            ['PH',0.01737,27.1],['ITW',0.01726,22.8],['CSX',0.01710,21.1],['EMR',0.01650,22.8],
            ['CTAS',0.01645,17.0],['NOC',0.01594,26.7],['FDX',0.01552,19.0],['MMM',0.01506,40.3],
            ['PCAR',0.01492,24.6],['CARR',0.01433,16.7],['GEV',0.01330,np.nan],['CPRT',0.01290,15.7],
            ['NSC',0.01287,23.3],['JCI',0.01218,16.1],['URI',0.01140,15.7],['LHX',0.01136,20.1],
            ['PAYX',0.01069,16.7],['PWR',0.01069,36.7],['GWW',0.01064,16.0],['RSG',0.01062,18.5],
            ['AME',0.01040,21.1],['VRSK',0.01039,16.3],['OTIS',0.01036,18.6],['CMI',0.01036,18.8],
            ['FAST',0.00985,25.0],['IR',0.00970,10.2],['XYL',0.00888,18.1],['DAL',0.00842,30.5],
            ['ODFL',0.00834,15.9],['HWM',0.00816,23.6],['ROK',0.00812,17.5],['EFX',0.00781,21.8],
            ['WAB',0.00757,22.7],['FTV',0.00682,27.9],['DOV',0.00662,24.5],['BR',0.00630,15.5],
            ['VLTO',0.00580,23.9],['AXON',0.00551,30.5],['HUBB',0.00542,18.9],['LDOS',0.00536,17.0],
            ['EXPD',0.00478,16.2],['J',0.00470,27.1],['LUV',0.00455,28.3],['BLDR',0.00449,26.7],
            ['TXT',0.00439,33.5],['UAL',0.00425,31.6],['IEX',0.00404,27.6],['MAS',0.00385,22.3],
            ['ROL',0.00374,18.6],['SNA',0.00370,27.6],['JBHT',0.00340,14.2],['SWK',0.00335,26.2],
            ['NDSN',0.00331,24.3],['PNR',0.00328,22.0],['CHRW',0.00278,17.5],['AOS',0.00273,26.8],
            ['ALLE',0.00272,19.8],['HII',0.00265,34.0],['GNRC',0.00222,22.0],['DAY',0.00207,17.3],
            ['AAL',0.00194,23.7],['PAYC',0.00184,18.4]],columns=['ticker','share','ESG'])
#GEV only in market since April 2024, no rating yet

data_xlk = pd.DataFrame([['MSFT',0.22370371,14.2],['NVDA',0.20701784,13.2],['AAPL',0.04444324,16.8],['AVGO',0.04115458,18.9],
            ['AMD',0.02638064,13.3],['ADBE',0.02404085,14.0],['CRM',0.02391538,14.4],['QCOM',0.0230207,13.4],
            ['ORCL',0.02259607,14.7],['AMAT',0.01982548,11.6],['ACN',0.01966981,8.6],['CSCO',0.01953999,12.9],
            ['TXN',0.01803387,21.9],['INTU',0.01797888,16.9],['IBM',0.01614853,13.3],['MU',0.01591629,18.6],
            ['NOW',0.01575995,15.0],['LRCX',0.01401987,12.2],['INTC',0.01332812,15.3],['ADI',0.01163403,18.1],
            ['KLAC',0.01121845,16.2],['PANW',0.01061883,13.4],['SNPS',0.00932391,13.9],['CRWD',0.00908376,17.6],
            ['ANET',0.00864819,13.7],['CDNS',0.00863467,11.2],['APH',0.00831441,19.0],['NXPI',0.00702902,19.2],
            ['MSI',0.00663099,12.2],['ROP',0.00613786,19.4],['ADSK',0.00528420,15.1],['MCHP',0.00490667,29.6],
            ['TEL',0.00466252,13.0],['SMCI',0.00432436,19.5],['MPWR',0.00406420,19.0],['FTNT',0.00379023,16.3],
            ['FICO',0.00366802,20.1],['IT',0.00355699,19.2],['CTSH',0.00348909,15.2],['HPQ',0.00318818,11.0],
            ['GLW',0.00312282,16.7],['CDW',0.00307467,7.5],['ON',0.00297272,20.8],['ANSS',0.00288438,14.5],
            ['FSLR',0.00279912,17.3],['HPE',0.00279639,11.4],['NTAP',0.00268478,14.3],['WDC',0.00255607,9.3],
            ['KEYS',0.00244564,5.2],['TER',0.00234632,15.3],['PTC',0.00215894,18.5],['TYL',0.00211128,18.4],
            ['STX',0.00206344,11.6],['GDDY',0.00202659,18.4],['TDY',0.00186036,34.3],['SWKS',0.00175479,24.1],
            ['ZBRA',0.00160297,10.0],['VRSN',0.00157029,20.9],['ENPH',0.00141150,19.9],['JBL',0.00140500,9.5],
            ['TRMB',0.00138146,9.5],['AKAM',0.00136904,13.3],['GEN',0.00135641,14.8],['JNPR',0.00117440,12.9],
            ['QRVO',0.00113516,20.2],['EPAM',0.00107405,27.4],['FFIV',0.00101206,16.3]],columns=['ticker','share','ESG'])

data_xlp = pd.DataFrame([['PG',0.14696,26.3],['COST',0.14160,26.2],['WMT',0.10748,23.9],['KO',0.09237,24.2],
            ['PEP',0.04527,20.8],['PM',0.04424,26.8],['MDLZ',0.03919,21.4],['CL',0.03487,25.0],
            ['MO',0.03433,32.2],['TGT',0.02924,17.1],['KMB',0.02007,27.5],['STZ',0.01814,26.0],
            ['GIS',0.01635,25.8],['SYY',0.01597,15.3],['KDP',0.01547,23.7],['KVUE',0.01538,17.0],
            ['MNST',0.01532,32.8],['KR',0.01436,23.2],['ADM',0.01291,31.6],['DG',0.01216,21.2],
            ['HSY',0.01189,21.7],['EL',0.01143,24.0],['KHC',0.01116,32.6],['CHD',0.01103,21.0],
            ['DLTR',0.00941,18.8],['MKC',0.00744,26.1],['CLX',0.00719,20.0],['TSN',0.00704,36.8],
            ['K',0.00654,25.7],['BG',0.00651,np.nan],['CAG',0.00598,26.9],['LW',0.00525,22.5],
            ['SJM',0.00490,27.2],['WBA',0.00480,16.0],['TAP',0.00404,25.5],['HRL',0.00376,26.9],
            ['CPB',0.00371,26.3],['BF.B',0.00333,25.8]],columns=['ticker','share','ESG'])

data_xlre = pd.DataFrame([['PLD',0.10385,10.6],['AMT',0.09189,12.6],['EQIX',0.07200,13.0],['WELL',0.06163,12.2],
             ['SPG',0.04922,12.5],['DLR',0.04853,12.2],['O',0.04706,15.5],['PSA',0.04677,11.7],
             ['CCI',0.04245,12.0],['EXR',0.03399,14.1],['CSGP',0.03082,21.1],['VICI',0.02978,13.9],
             ['AVB',0.02962,8.1],['CBRE',0.02720,6.3],['IRM',0.02651,12.4],['EQR',0.02416,11.4],
             ['SBAC',0.02114,9.7],['WY',0.02105,15.2],['INVH',0.02090,16.0],['VTR',0.02062,11.2],
             ['ARE',0.01869,13.1],['ESS',0.01808,11.6],['MAA',0.01684,11.7],['DOC',0.01379,11.3],
             ['HST',0.01302,12.9],['KIM',0.01279,10.4],['UDR',0.01263,12.9],['CPT',0.01196,14.2],
             ['REG',0.01030,11.3],['BXP',0.00911,12.0],['FRT',0.00760,12.4]],columns=['ticker','share','ESG'])
#WELL data taken from Welltower OP LLC

data_xlu = pd.DataFrame([['NEE',0.14245,24.9],['SO',0.08105,28.1],['DUK',0.07347,26.8],['CEG',0.06660,28.3],
            ['SRE',0.04541,23.2],['AEP',0.04351,22.1],['D',0.03917,28.0],['PCG',0.03573,30.4],
            ['PEG',0.03481,21.2],['EXC',0.03308,18.8],['ED',0.02937,21.1],['XEL',0.02836,26.3],
            ['VST',0.02774,29.3],['EIX',0.02627,24.0],['AWK',0.02395,18.7],['WEC',0.02350,22.8],
            ['DTE',0.02163,31.3],['ETR',0.02157,24.9],['PPL',0.01948,26.9],['ES',0.01908,18.1],
            ['FE',0.01889,28.0],['CNP',0.01847,24.8],['AEE',0.01777,26.0],['CMS',0.01678,20.3],
            ['ATO',0.01670,31.2],['NRG',0.01629,34.2],['AES',0.01276,23.5],['LNT',0.01233,17.1],
            ['NI',0.01228,20.6],['EVRG',0.01161,29.2],['PNW',0.00816,25.8]],columns=['ticker','share','ESG'])

data_xlv = pd.DataFrame([['LLY',0.13347,23.6],['UNH',0.08241,17.0],['JNJ',0.06547,21.3],['MRK',0.06222,21.1],
            ['ABBV',0.05572,26.8],['TMO',0.03926,12.7],['ABT',0.03391,22.2],['AMGN',0.03166,22.7],
            ['DHR',0.03096,10.7],['PFE',0.02931,17.7],['ISRG',0.02899,19.5],['ELV',0.02299,10.0],
            ['VRTX',0.02260,19.3],['SYK',0.02128,23.9],['BSX',0.02106,22.1],['REGN',0.02100,16.8],
            ['MDT',0.01973,22.2],['CI',0.01779,13.0],['GILD',0.01603,21.7],['BMY',0.01579,21.2],
            ['MCK',0.01452,13.4],['ZTS',0.01417,15.1],['CVS',0.01402,18.7],['BDX',0.01249,23.7],
            ['HCA',0.01219,27.8],['EW',0.01014350,22.1],['MRNA',0.00847883,19.8],['DXCM',0.00810720,21.7],
            ['HUM',0.00792353,19.0],['IDXX',0.00749501,16.3],['A',0.00729477,11.4],['COR',0.00729134,11.3],
            ['IQV',0.00709810,16.1],['CNC',0.00665289,15.7],['GEHC',0.00618282,30.0],['BIIB',0.00600539,20.7],
            ['MTD',0.00563232,12.1],['RMD',0.00497815,24.7],['CAH',0.00464215,11.2],['WST',0.00435610,18.0],
            ['ZBH',0.00410545,25.9],['STE',0.00391855,30.1],['MOH',0.00329657,23.1],['COO',0.00327676,15.6],
            ['LH',0.00323781,np.nan],['BAX',0.00319816,21.8],['WAT',0.00312392,12.8],['HOLX',0.00309028,23.8],
            ['ALGN',0.00304356,18.6],['DGX',0.00282407,20.3],['PODD',0.00264544,21.5],['RVTY',0.00238514,16.2],
            ['VTRS',0.00226874,25.3],['INCY',0.00215039,23.8],['TECH',0.00212279,25.8],['UHS',0.00207036,30.8],
            ['CRL',0.00194947,18.1],['CTLT',0.00185776,20.8],['TFX',0.00171463,23.8],['HSIC',0.00156434,14.5],
            ['DVA',0.00136668,21.8],['SOLV',0.00130223,np.nan],['BIO',0.00105635,16.8]],columns=['ticker','share','ESG'])

data_xly = pd.DataFrame([['AMZN',0.22986,29.3],['TSLA',0.14337,24.7],['HD',0.09248,12.8],['MCD',0.04499,25.8],
            ['BKNG',0.03804,17.2],['TJX',0.03501,15.5],['LOW',0.03472,11.8],['NKE',0.03206,18.7],
            ['CMG',0.02519,20.0],['SBUX',0.02508,22.3],['ABNB',0.01861,23.7],['ORLY',0.01740,11.8],
            ['MAR',0.01641,20.3],['HLT',0.01502,16.2],['GM',0.01479,28.3],['AZO',0.01420,11.0],
            ['ROST',0.01390,17.2],['F',0.01324,23.0],['DHI',0.01163,21.6],['RCL',0.01063,19.4],
            ['YUM',0.01041,20.5],['LEN',0.01020,26.0],['LULU',0.00989226,14.4],['TSCO',0.00796684,13.1],
            ['EBAY',0.00762533,15.4],['DECK',0.00699161,14.2],['GRMN',0.00688052,21.0],['NVR',0.00662521,21.0],
            ['PHM',0.0646561,21.0],['APTV',0.00557520,9.1],['GPC',0.00541276,12.8],['ULTA',0.00519766,15.5],
            ['DPZ',0.00514269,28.2],['DRI',0.00510369,27.5],['CCL',0.00502788,22.4],['BBY',0.00465368,14.0],
            ['EXPE',0.00460094,22.5],['LVS',0.00452456,20.7],['POOL',0.00332625,10.9],['KMX',0.0015531,11.1],
            ['LKQ',0.00309181,11.0],['MGM',0.00294129,24.1],['TPR',0.00268723,13.5],['BBWI',0.00255779,26.2],
            ['WYNN',0.00232888,24.9],['CZR',0.00228445,20.2],['NCLH',0.00219198,24.0],['HAS',0.00217140,7.3],
            ['BWA',0.00209197,10.0],['RL',0.00198736,14.0],['ETSY',0.00194299,15.6],['MHK',0.00157449,14.0]],
            columns=['ticker','share','ESG'])

#the higher the ESG-value the worse

#compute ESG portfolio
#does not allow for negative portfolio weights
def ESG_value(df):
    total = df.dropna()['share'].sum() 
    #ignore missing values instead of computing as if ESG value=0
    df.share *=1/total
    df['esg_rel'] = df.share*df.ESG 
    ESG = df['esg_rel'].sum()
    return ESG

def ESG_portf(dfs):
    val = []
    for i in range(len(dfs)):
        val.append(50-ESG_value(dfs[i]))
    total = sum(val)
    val = [x*1/total for x in val]
    return val 

def plot_ESG_portf(dfs):
    values = ESG_portf(dfs)
    colors = ['g' if m > 0 else 'r' for m in values]
    plt.bar(tickers,values,width=.5,color=colors)
    plt.xlabel('ETFs')
    plt.ylabel('share in portfolio')
    plt.title('ESG portfolio')
    plt.show()
    
def ESG_ret_risk(dfs):
    av = av_ret.flatten().tolist()
    val = []
    for i in range(len(dfs)):
        val.append(av[i]*ESG_portf(dfs)[i])
    matr = np.array(ESG_portf(dfs))
    s_2 = matr.reshape((1,11)) @ cov_all @ matr.reshape((11,1))
    s = np.sqrt(s_2.item())
    return sum(val),s

#plot both efficient portfolio fronts together
def plot_with_ESG(lists,dfs):
    a1 = constants(av_ret, cov_all,d)[0]
    b1 = constants(av_ret, cov_all,d)[1]
    c1 = constants(av_ret, cov_all,d)[2]
    s = np.linspace(0,50,num=1000)
    plt.plot(s,effport(a1,b1,c1,s),color='navy',linewidth=1,label='all')
    for i in range(len(lists)):
        a = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[0]
        b = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[1]
        c = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[2]
        string = ','.join(str(elm) for elm in lists[i])
        plt.plot(s,effport(a,b,c,s),linewidth=1,label='no '+string)
    plt.scatter(ESG_ret_risk(dfs)[1],ESG_ret_risk(dfs)[0],c='green',marker='x',label='ESG')
    plt.title('Efficient portfolio front & ESG-portfolio')
    plt.xlabel('risk sigma')
    plt.ylabel('expected return r')
    plt.legend()
    plt.show()

#plot portfolios together
def plot_with_ESG_investments(r,lists,dfs):
    n=len(lists)+2
    values1 = list(markowitz_portf(av_ret,cov_all,r,d)[1].flat)
    x_axis = np.arange(len(tickers))
    plt.bar(x_axis - (len(lists)+1)/8,values1,color='navy',label='all',width=1/(n+1))
    values2 = ESG_portf(dfs)
    plt.bar(x_axis -(len(lists)+1)/8+1/(n+1),values2,label='ESG',color='green',width=1/(n+1))
    for i in range(len(lists)):
        values = list(markowitz_portf(green(lists[i])[1],green(lists[i])[2],r,green(lists[i])[3])[1].flat)
        for j in range(len(green(lists[i])[4])):
            values.insert(green(lists[i])[4][j],0)
        string = ','.join(str(elm) for elm in lists[i])
        plt.bar(x_axis - (len(lists)+1)/8+(i+2)*1/(n+1),values,label='no '+string,width=1/(n+1))
    plt.xticks(x_axis,tickers)
    plt.xlabel('ETFs')
    plt.ylabel('share in portfolio')
    plt.title('Markowitz portfolios for r= '+str(r))
    plt.legend()
    plt.show()
    
#loss functions
def loss_ESG_s(lists,dfs):
    s = ESG_ret_risk(dfs)[1]
    a1 = constants(av_ret, cov_all,d)[0]
    b1 = constants(av_ret, cov_all,d)[1]
    c1 = constants(av_ret, cov_all,d)[2]
    val = []
    val.append(effport(a1,b1,c1,s)-ESG_ret_risk(dfs)[0])
    for i in range(len(lists)):
        a = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[0]
        b = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[1]
        c = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[2]
        val.append(effport(a,b,c,s)-ESG_ret_risk(dfs)[0])
    return val
    
def loss_ESG_r(lists,dfs):
    r = ESG_ret_risk(dfs)[0]
    a1 = constants(av_ret, cov_all,d)[0]
    b1 = constants(av_ret, cov_all,d)[1]
    c1 = constants(av_ret, cov_all,d)[2]
    losses = []
    losses.append(ESG_ret_risk(dfs)[1]-inverse(a1,b1,c1,r))
    for i in range(len(lists)):
        a = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[0]
        b = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[1]
        c = constants(green(lists[i])[1],green(lists[i])[2],green(lists[i])[3])[2]
        losses.append(ESG_ret_risk(dfs)[1]-inverse(a,b,c,r))
    return losses

#plot loss functions
def plot_loss_ESG_s(lists,dfs):
    s = ESG_ret_risk(dfs)[1]
    plt.scatter(s,loss_ESG_s(lists,dfs)[0],marker='o',label='all')
    for i in range(1,len(loss_ESG_s(lists,dfs))):
        string = ','.join(str(elm) for elm in lists[i-1])
        plt.scatter(s,loss_ESG_s(lists,dfs)[i],marker='o',label='no '+string)
    plt.title('Loss in expexted return of ESG portfolio vs Markowitz')
    plt.xlabel('risk')
    plt.ylabel('expected loss')
    plt.legend()
    plt.show()
    
def plot_loss_ESG_r(lists,dfs):
    r = ESG_ret_risk(dfs)[0]
    plt.scatter(r,loss_ESG_r(lists,dfs)[0],marker='o',label='all')
    for i in range(1,len(loss_ESG_r(lists,dfs))):
        string = ','.join(str(elm) for elm in lists[i-1])
        plt.scatter(r,loss_ESG_r(lists,dfs)[i],marker='o',label='no '+string)
    plt.title('Increase in risk of ESG portfolio vs Markowitz')
    plt.xlabel('expected return')
    plt.ylabel('risk increase')
    plt.legend()
    plt.show()

#plotting
dfs = [data_gdx,data_xlc,data_xle,data_xlf,data_xli,data_xlk,data_xlp,data_xlre,
       data_xlu,data_xlv,data_xly]
plot_ESG_portf(dfs)
print(ESG_ret_risk(dfs))
plot_with_ESG([['XLE'],['XLE','GDX']], dfs)
plot_with_ESG_investments(ESG_ret_risk(dfs)[0],[['XLE'],['XLE','GDX']],dfs)
plot_loss_ESG_s([['XLE'],['XLE','GDX']],dfs)
plot_loss_ESG_r([['XLE'],['XLE','GDX']],dfs)

print('GDX: '+str(ESG_value(data_gdx)))
print('XLC: '+str(ESG_value(data_xlc)))
print('XLE: '+str(ESG_value(data_xle)))
print('XLF: '+str(ESG_value(data_xlf)))
print('XLI: '+str(ESG_value(data_xli)))
print('XLK: '+str(ESG_value(data_xlk)))
print('XLP: '+str(ESG_value(data_xlp)))
print('XLRE: '+str(ESG_value(data_xlre)))
print('XLU: '+str(ESG_value(data_xlu)))
print('XLV: '+str(ESG_value(data_xlv)))
print('XLY: '+str(ESG_value(data_xly)))





