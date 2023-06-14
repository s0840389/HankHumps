
import os
import pickle

mainpath='/home/jamie/OneDrive/Documents/research/HANK/HankHumps'

os.chdir(mainpath)

import numpy as np
import matplotlib.pyplot as plt

from sequence_jacobian import simple, solved, combine, create_model  # functions
from sequence_jacobian import grids, hetblocks                       # modules

from hh_prob_1A import * #. py file with household prob

from sticky_exp import *


# household prob (from package)

def make_grid(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, pi_e, Pi = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, pi_e, Pi, a_grid


def transfers(pi_e, Div,Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax_rule, div_rule = e_grid, e_grid
    div = Div / np.sum(pi_e * div_rule) * div_rule
    tax = Tax / np.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T

def wages(w, e_grid):
    we = w * e_grid
    return we


def labor_supply(n, e_grid):
    ne = e_grid[:, np.newaxis] * n
    return ne

hh = hh.add_hetinputs([make_grid, transfers, wages])

hh_ext = hh.add_hetoutputs([labor_supply])


# firms

@simple
def firm(Y, w, Z, pi, mu, kappa):
    L = Y / Z
    Div = Y - w * L - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y
    return L, Div


# monetary and fiscal

@simple
def monetary(pi, rstar, phi):
    r = (1 + rstar(-1) + phi * pi(-1)) / (1 + pi) - 1
    return r


#@simple
#def fiscal(r, B):
#    Tax = r * B
#    return Tax


@simple
def fiscalSS(B,r,G):
    Tax=(r*B+G)    
    return Tax

@solved(unknowns={'B': 4.,'Tax': 0.3 }, targets=['B_res','tax_res'], solver="broyden_custom")
def fiscal(Tax,B,r, G,gammatax,rhotax,taxss,Bss):
    B_res=B(-1)*(1+r)+G-Tax-B
    tax_res = taxss*(Tax(-1)/taxss)**rhotax*(B(-1)/Bss)**(gammatax*(1-rhotax))-Tax
    return B_res,tax_res


# market clearing

@simple
def mkt_clearing(A, NE, C, L, Y, B, pi, mu, kappa,G):
    asset_mkt = A - B
    labor_mkt = NE - L
    goods_mkt = Y - C - mu/(mu-1)/(2*kappa) * (1+pi).apply(np.log)**2 * Y -G
    return asset_mkt, labor_mkt, goods_mkt

# philips curve
@simple
def nkpc(pi, w, Z, Y, r, mu, kappa):
    nkpc_res = kappa * (w / Z - 1 / mu) + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))\
               - (1 + pi).apply(np.log)
    return nkpc_res

# steady state

@simple
def nkpc_ss(Z, mu):
    w = Z / mu
    return w

blocks_ss = [hh_ext, firm, monetary, fiscalSS, mkt_clearing, nkpc_ss]

hank_ss = create_model(blocks_ss, name="One-Asset HANK SS")

calibration = {'eis': 0.5, 'frisch': 0.5, 'rho_e': 0.966, 'sd_e': 0.5, 'nE': 7,
               'amin': 0.0, 'amax': 150, 'nA': 250, 'Y': 1.0, 'Z': 1.0, 'pi': 0.0,
               'mu': 1.2, 'kappa': 0.1, 'rstar': 0.005, 'phi': 1.5, 'B': 5.6,
               'cbar': 0.0,'G':0.0,'rhotax':0.9,'gammatax':0.9}

unknowns_ss = {'beta': 0.986, 'vphi': 0.8}
targets_ss = {'asset_mkt': 0, 'labor_mkt': 0}

ss = hank_ss.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")

ss['Bss']=ss['B']
ss['taxss']=ss['Tax']

# dynamic model

blocks = [hh_ext, firm, monetary, fiscalSS, mkt_clearing, nkpc]
hank = create_model(blocks, name="One-Asset HANK")

T = 300

exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w', 'Y']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']

J_ha = hh_ext.jacobian(ss, inputs=['Tax','Div', 'r','w'], T=T)

G = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha})

J_ha_sticky=stick_jacob(J_ha,0.94) # Reduce forwarding lookingness of hh Jacobian and resolve model

G_sticky = hank.solve_jacobian(ss, unknowns, targets, exogenous, T=T,Js={'hh': J_ha_sticky})


# Consumption decomposition chart

drstar = -0.0025 * 0.61 ** (np.arange(T)[:, np.newaxis])


dc_lin=100*G['C']['rstar'] @ drstar/ss['C']

dc_lin_sticky=100*G_sticky['C']['rstar'] @ drstar/ss['C']

fig1,(ax1,ax2) =plt.subplots(1,2)

tt=np.arange(0,T)

yyoldminus=0*tt[0:24]
yyoldplus=0*yyoldminus

yyoldminusST=0*tt[0:24]
yyoldplusST=0*yyoldminus

bcolor=['darkblue','darkgreen','grey','gold']

iter=0
for i in ['r','Div','Tax','w']:
    
    yy=J_ha['C'][i]@G[i]['rstar']@drstar/ss['C']*100 # combine HH jacobian with GE inputs 

    yyST=J_ha_sticky['C'][i]@G_sticky[i]['rstar']@drstar/ss['C']*100 # combine HH jacobian with GE inputs 

    ax1.bar(tt[:24],yy[:24,-1].clip(min=0),bottom=yyoldplus,label=i,color=bcolor[iter])
    ax1.bar(tt[:24],yy[:24,-1].clip(max=0),bottom=yyoldminus,color=bcolor[iter])
    
    ax2.bar(tt[:24],yyST[:24,-1].clip(min=0),bottom=yyoldplusST,label=i,color=bcolor[iter])
    ax2.bar(tt[:24],yyST[:24,-1].clip(max=0),bottom=yyoldminusST,color=bcolor[iter])

    yyoldplus=yy[:24,-1].clip(min=0)+yyoldplus
    yyoldminus=yy[:24,-1].clip(max=0)+yyoldminus

    yyoldplusST=yyST[:24,-1].clip(min=0)+yyoldplusST
    yyoldminusST=yyST[:24,-1].clip(max=0)+yyoldminusST

    iter=iter+1

ax1.plot(dc_lin[:24], label='Total', linestyle='-', linewidth=2.5)
ax2.plot(dc_lin_sticky[:24], label='Total', linestyle='-', linewidth=2.5)

ax1.set_title('Full info')
ax2.set_title('Sticky')

ax2.legend()


#plt.title('Decomposition of consumption response to monetary policy')
plt.show()

fig1.savefig('Consump_decomp.png')

# Output IRFs comparison

varpick='Y'
shockpick='rstar'
shockname='MP shock'

dy=100*G[varpick][shockpick] @ drstar/ss['Y']
dy_sticky=100*G_sticky[varpick][shockpick] @ drstar/ss['Y']

# forward gudiance 
drstar_fwd=np.roll(drstar,10)
drstar_fwd[:10]=0
dy_fwd=100*G[varpick][shockpick] @ drstar_fwd/ss['Y']

dy_sticky_fwd=100*G_sticky[varpick][shockpick] @ drstar_fwd/ss['Y']


fig2,ax=plt.subplots()

plt.plot(dy[:24], label='Full info', linestyle='-', linewidth=2.5)
plt.plot(dy_fwd[:24], label='Fwd Full info', linestyle='--', linewidth=2.5)

plt.plot(dy_sticky[:24], label='Sticky', linestyle='--', linewidth=2.5)
plt.plot(dy_sticky_fwd[:24], label='Fwd Sticky', linestyle='--', linewidth=2.5)

tit=varpick+' response to 1% '+shockname

plt.title(tit)
plt.xlabel('quarters')
plt.ylabel('% deviation from ss')
plt.legend()
plt.show()

fig2.savefig('MP_IRF_1A.png')


fig4, ax=plt.subplots()

Adist=np.sum(ss.internals['hh']['D'],axis=0)
agrid=ss.internals['hh']['a_grid']

ax.bar(agrid,Adist)
plt.xlim([0,100])
plt.title('asset distribution')
plt.show()

fig4.savefig('Asset_dist_1A.png')

