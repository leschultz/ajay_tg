#!/usr/bin/python3

from scipy.interpolate import UnivariateSpline as interpolate
from matplotlib import pyplot as pl

from scipy.stats import linregress
from os.path import join

import numpy as np
import os


def rmse(act, pred):
    act = np.array(act)
    pred = np.array(pred)

    e = (pred-act)**2
    e = e.mean()
    e = np.sqrt(e)

    return e


def opt(x, y, tol=0.1):
    '''
    '''

    x = np.array(x)
    y = np.array(y)

    left_rmse = []
    right_rmse = []

    n = len(x)
    indexes = list(range(n-1))
    ldata = []
    rdata = []
    for i in indexes:
        xl = x[:n-i]
        xr = x[i:]

        yl = y[:n-i]
        yr = y[i:]

        ml, il, _, _, _ = linregress(xl, yl)
        mr, ir, _, _, _ = linregress(xr, yr)

        yfitl = ml*xl+il
        yfitr = mr*xr+ir

        rmsel = rmse(yl, yfitl)
        rmser = rmse(yr, yfitr)

        if i > 0:
            ldata.append((xl[-1], rmsel))
            rdata.append((xr[0], rmser))

        left_rmse.append(rmsel)
        right_rmse.append(rmser)

    left_rmse = np.array(left_rmse)
    right_rmse = np.array(right_rmse)

    ldata = np.array(ldata)
    rdata = np.array(rdata[::-1])

    middle_rmse = (ldata[:, 1]+rdata[:, 1])/2
    mcut = np.argmin(middle_rmse)

    lcut = np.argmax(left_rmse <= tol*left_rmse.max())
    rcut = np.argmax(right_rmse <= tol*right_rmse.max())
    xcut = ldata[mcut, 0]

    left = x[n-lcut]
    right = x[rcut]

    return xcut, left, right, ldata, rdata, middle_rmse

datadir = 'data'
items = os.listdir(datadir)

data = {}
for item in items:
    try:
        data[item] = np.loadtxt(join(datadir, item))
    except Exception:
        pass

tg = []
rates = []
for key, value in data.items():
    x = value[:, 0][::-1]
    y = value[:, 1][::-1]

    xlim = 1000
    condition = x <= xlim
    x = x[condition]
    y = y[condition]

    s = interpolate(x=x, y=y, k=5, s=1)
    xfit = np.linspace(x[0], x[-1], 1000)
    yfit = s(xfit)

    xcut, left, right, ldata, rdata, middle_rmse = opt(xfit, yfit)

    rate = key.strip('plotBoltzmann')
    if 'times' in rate:
        split = rate.split('times')
        rate = split[0]+'e'+split[1]
        rate = float(rate)
    else:
        rate = float('1e'+rate)

    tg.append(xcut)
    rates.append(rate)

    fig, ax = pl.subplots()

    ax.plot(x, y, marker='*', linestyle='none', color='b', label='data')
    ax.plot(xfit, yfit, color='g', label='interpolation')
    ax.axvline(left, linestyle=':', color='k', label='glass transition')
    ax.axvline(right, linestyle=':', color='k')
    ax.axvline(xcut, linestyle='--', color='r', label='Tg = '+str(xcut))

    ax.set_xlabel('Temperature [K]')
    ax.set_ylabel(key+' y data')

    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig('figures/'+key)
    pl.close('all')

    fig, ax = pl.subplots()

    ax.plot(ldata[:, 0], ldata[:, 1], label='left fits')
    ax.plot(rdata[:, 0], rdata[:, 1], label='right fits')
    ax.plot(ldata[:, 0], middle_rmse, label='left and right fits')

    ax.set_title(key)
    ax.set_xlabel('End Temperature [K]')
    ax.set_ylabel('RMSE')
    ax.grid()
    ax.legend()
    
    fig.tight_layout()
    fig.savefig('figures/'+key+'_msqe')
    pl.close('all')

figtg, axtg = pl.subplots()
axtg.plot(rates, tg, linestyle='none', marker='.')

axtg.set_xscale('log')
axtg.set_xlabel('Rates [K/s]')
axtg.set_ylabel('Tg [K]')

axtg.grid()

figtg.tight_layout()
figtg.savefig('figures/Tg_rates')
