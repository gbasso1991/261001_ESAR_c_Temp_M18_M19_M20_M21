#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return SAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% M17
nombre_M17='M17'
ciclos_M17 = glob("M17/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_M17 = glob("M17/**/*resultados.txt", recursive=True)
ciclos_M17.sort()
resultados_M17.sort()
conc_M17 =  23.9 #g/L 

print('Importando ciclos de', nombre_M17,'\n')
for i,p in enumerate(ciclos_M17):
    print('  ',i,p)
print('\n')
for i,res in enumerate(resultados_M17):
    print('  ',i,res)
print('-'*50)

SAR_M17, tau_M17, Hc_M17 = extraer_SAR_tau(resultados_M17)
res_M17=[]
#%% ploteo ciclos
fig00, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('1',os.path.basename(e))
        axs[0].plot(H_M17/1000,M_M17,'-',label=f'{SAR_M17[i]:.3uS}')

    elif '125dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('2',os.path.basename(e))
        axs[1].plot(H_M17/1000,M_M17,'-',label=f'{SAR_M17[i]:.3uS}')

    elif '150dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('3',os.path.basename(e))
        axs[2].plot(H_M17/1000,M_M17,'-',label=f'{SAR_M17[i]:.3uS}')


for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_M17} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M17:.1f} g/L')

#%%%
fig000, axs =plt.subplots(1,1,figsize=(9,7),constrained_layout=True,sharey=True,sharex=True)
axs.set_ylabel('M (A/m)')
lines=['-','--','-.',':']*3
ejes=[-57 , -47, -38,0, 38, 47, 57]
str_ejes=[str(e) for e in ejes] 

for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('1',os.path.basename(e))
        axs.plot(H_M17/1000,M_M17,c='C0',ls=lines[i],label=f'{SAR_M17[i]:.3uS}')

    elif '125dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('2',os.path.basename(e))
        axs.plot(H_M17/1000,M_M17,c='C1',ls=lines[i],label=f'{SAR_M17[i]:.3uS}')

    elif '150dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('3',os.path.basename(e))
        axs.plot(H_M17/1000,M_M17,c='C2',ls=lines[i],label=f'{SAR_M17[i]:.3uS}')

axs.set_xticks(ejes)
axs.set_xticklabels(str_ejes)
axs.grid()
axs.set_xlabel('H (kA/m)')
axs.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)',ncol=1)
plt.suptitle(f'Ciclos promedio {nombre_M17} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M17:.1f} g/L')


#%%
print('Resultados M17', '='*80,'\n')
for r in resultados_M17:
    res_M17.append(ResultadosESAR(os.path.dirname(r)))
rates_M17 = []

#%% Templogs
fig01, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')


for i,r in enumerate(res_M17):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M17.append(rate)
    print(i,f'WRate = {rate:.2f} °C/s')
    if '100dA' in r.directorio:
        axs[0].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')
    elif '125dA' in r.directorio:
        axs[1].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')
    elif '150dA' in r.directorio:
        axs[2].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')

axs[0].set_ylabel('T (°C)')
for a in axs:
    a.grid()
    a.set_xlabel('t (s)')
    a.legend(loc='best',frameon=True,shadow=True,title='Warming Rate (°C/s)')
plt.suptitle(f'Templogs {nombre_M17} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M17:.1f} g/L')

################################################################################################################################
#%% 2- M16 
nombre_M16='M16'
ciclos_M16 = glob("M16/**/*ciclo_promedio_H_M.txt", recursive=True)
resultados_M16 = glob("M16/**/*resultados.txt", recursive=True)
ciclos_M16.sort()
resultados_M16.sort()
conc_M16 =  16.5 #g/L 

print('Importando ciclos de', nombre_M16,'\n')
for i,p in enumerate(ciclos_M16):
    print('  ',i,p)
print('\n')
for i,res in enumerate(resultados_M16):
    print('  ',i,res)
print('-'*50)

SAR_M16, tau_M16, Hc_M16 = extraer_SAR_tau(resultados_M16)
res_M16=[]
#%% ploteo ciclos
fig10, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('20 kA/m',loc='left')
axs[1].set_title('38 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print(os.path.split(e)[-1])
        axs[0].plot(H_M16/1000,M_M16,'-',label=f'{SAR_M16[i]:.3uS}')

for i,e in enumerate(ciclos_M16):
    if '125dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print(os.path.split(e)[-1])
        axs[1].plot(H_M16/1000,M_M16,'-',label=f'{SAR_M16[i]:.3uS}')

for i,e in enumerate(ciclos_M16):
    if '150dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print(os.path.split(e)[-1])
        axs[2].plot(H_M16/1000,M_M16,'-',label=f'{SAR_M16[i]:.3uS}')

for a in axs:
    a.grid()
    a.set_xlabel('H (kA/m)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Ciclos promedio {nombre_M16} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M16:.1f} g/L')
#%%
fig100, axs =plt.subplots(1,1,figsize=(9,7),constrained_layout=True,sharey=True,sharex=True)
axs.set_ylabel('M (A/m)')
ls=['-','--','-.',':']*3
ejes=[-57 , -47, -38,0, 38, 47, 57]
str_ejes=[str(e) for e in ejes] 
for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print(e)
        axs.plot(H_M16/1000,M_M16,c='C0',ls=ls[i],label=f'{SAR_M16[i]:.3uS}')

    elif '125dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print(e)
        axs.plot(H_M16/1000,M_M16,c='C1',ls=ls[i],label=f'{SAR_M16[i]:.3uS}')

    elif '150dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print(e)
        axs.plot(H_M16/1000,M_M16,c='C2',ls=ls[i],label=f'{SAR_M16[i]:.3uS}')
axs.set_xticks(ejes)
axs.set_xticklabels(str_ejes)

axs.grid()
axs.set_xlabel('H (kA/m)')
axs.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)',ncol=1)
plt.suptitle(f'Ciclos promedio {nombre_M16} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M16:.1f} g/L')

#%%
print('Resultados M16', '='*80,'\n')
for r in resultados_M16:
    res_M16.append(ResultadosESAR(os.path.dirname(r)))
rates_M16 = []

#%% Templogs
fig11, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=False)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,r in enumerate(res_M16):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    rates_M16.append(rate)
    print(f'WRate = {rate:.2f} °C/s')
    if '100dA' in r.directorio:
        axs[0].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')
    elif '125dA' in r.directorio:
        axs[1].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')
    elif '150dA' in r.directorio:
        axs[2].plot(r.time,r.temperatura,'.-',label=f'{rate:.1f} °C/s')

axs[0].set_ylabel('T (°C)')
for a in axs:
    a.grid()
    a.set_xlabel('t (s)')
    a.legend(loc='upper left',frameon=True,shadow=True,title='Warming Rate (°C/s)')
plt.suptitle(f'Templogs {nombre_M16} \n300 kHz & [38, 47, 57] kA/m\nC = {conc_M16:.1f} g/L')
#%% Normalizo ciclos por concentracion y ploteo comparativo

fig2, axs =plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True,sharex=True)
axs[0].set_ylabel('M (A/m)')
axs[0].set_title('38 kA/m',loc='left')
axs[1].set_title('47 kA/m',loc='left')
axs[2].set_title('57 kA/m',loc='left')

for i,e in enumerate(ciclos_M17):
    if '100dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('1.1',os.path.split(e)[-1])
        axs[0].plot(H_M17/1000,M_M17/conc_M17,'-',c='C0',label=f'M17\n{conc_M17} g/L' if i==0 else "")
    elif '125dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('2.1',os.path.split(e)[-1])
        axs[1].plot(H_M17/1000,M_M17/conc_M17,'-',c='C0',label=f'M17\n{conc_M17} g/L' if i==4 else "")
    elif '150dA' in e:
        _,_,_, H_M17,M_M17,_ = lector_ciclos(ciclos_M17[i])
        print('3.1',os.path.split(e)[-1])
        axs[2].plot(H_M17/1000,M_M17/conc_M17,'-',c='C0',label=f'M17\n{conc_M17} g/L' if i==7 else "")
        
for i,e in enumerate(ciclos_M16):
    if '100dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('1.2',os.path.split(e)[-1])
        axs[0].plot(H_M16/1000,M_M16/conc_M16,'-',c='C1',label=f'M16\n{conc_M16} g/L' if i==0 else "")
    elif '125dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('2.2',os.path.split(e)[-1])
        axs[1].plot(H_M16/1000,M_M16/conc_M16,'-',c='C1',label=f'M16\n{conc_M16} g/L' if i==4 else "")
    elif '150dA' in e:
        _,_,_, H_M16,M_M16,_ = lector_ciclos(ciclos_M16[i])
        print('3.2',os.path.split(e)[-1])
        axs[2].plot(H_M16/1000,M_M16/conc_M16,'-',c='C1',label=f'M16\n{conc_M16} g/L' if i==7 else "")

axs[0].set_ylim(-80,80)
axs[0].set_ylabel('M/[NPM] (Am²/kg)')
for a in axs:
    a.set_xlabel('H (kA/m)')
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,ncol=2)
plt.suptitle(f'Ciclos promedio nomalizados por concentracion\n300 kHz & [38, 47, 57] kA/m\n')

#%% ploteo comparativo de errorbars de ESAR
categorias = ['M17', 'M16']
x = np.arange(len(categorias))

fig3, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(SAR_M17[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_M16[:3]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(SAR_M17[3:7]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_M16[3:7]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(SAR_M17[7:]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(SAR_M16[7:]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
    

ax.set_ylabel('ESAR (W/g)')
plt.suptitle(f'ESAR\n300 kHz & [38, 47, 57] kA/m\n')

plt.show()
#%% ploteo comparativo de tau

categorias = ['260630\nNF@cit', '260630\nNF@M16']
x = np.arange(len(categorias))

fig4, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(tau_M17[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_M16[:3]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(tau_M17[3:7]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_M16[3:7]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(tau_M17[7:]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(tau_M16[7:]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
ax.set_ylabel('tau (ns)')
plt.suptitle(f'tau\n300 kHz & [38, 47, 57] kA/m\n')
plt.show()

#%% Idem Hc
fig5, (ax,ax2,ax3) = plt.subplots(1,3,figsize=(12,4),constrained_layout=True,sharey=True)

sep = 0.25

for i,s in enumerate(Hc_M17[:3]):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_M16[:3]):
    ax.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(Hc_M17[3:7]):
    ax2.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_M16[3:7]):
    ax2.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

for i,s in enumerate(Hc_M17[7:]):
    ax3.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

for j,s in enumerate(Hc_M16[7:]):
    ax3.bar(j*sep + 3*sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')
for a in [ax,ax2,ax3]:
    a.grid(axis='y', alpha=0.3)
    a.set_xticks(x)
    a.set_xticklabels(categorias)
ax.set_ylabel('Hc (kA/m)')
plt.suptitle(f'Hc\n300 kHz & [38, 47, 57] kA/m\n')
plt.show()
#%% Salvo todas las figuras
fig00.savefig('00_ciclos_promedio_M17.png',dpi=300)
fig10.savefig('00_ciclos_promedio_M16.png',dpi=300)
fig000.savefig('01_ciclos_promedio_all_M17.png',dpi=300)
fig100.savefig('01_ciclos_promedio_all_M16.png',dpi=300)
fig01.savefig('02_templogs_M17.png',dpi=300)
fig11.savefig('03_templogs_M16.png',dpi=300)
fig2.savefig('04_ciclos_promedio_comparativa.png',dpi=300)
fig3.savefig('05_ESAR_comparativa.png',dpi=300)
fig4.savefig('06_tau_comparativa.png',dpi=300)
fig5.savefig('07_Hc_comparativa.png',dpi=300)

# %%
#%% Printeo resultados
print(f'Muestra = {nombre_M17}')
print(f'Concentracion = {conc_M17:.1f} g/L')
print(f'ESAR = {SAR_M17} W/g')
print(f'tau = {tau_M17} ns')
print(f'Hc = {Hc_M17} kA/m') 
print(f'WR = {rates_M17} °C/s')

#%%
def promedio_por_grupos(lista, tamanos):
    """
    Calcula el promedio de grupos de distinto tamaño.
    
    tamanos: lista con la cantidad de elementos de cada grupo.
    """
    promedios = []
    inicio = 0

    for n in tamanos:
        grupo = lista[inicio:inicio+n]
        promedios.append(sum(grupo) / n)
        inicio += n

    return promedios


def promedio_por_grupos_bis(lista, tamanos):
    """
    Calcula promedio y desviación estándar muestral para grupos
    de distinto tamaño, devolviendo ufloats.
    """
    prom = []
    inicio = 0

    for n in tamanos:
        grupo = np.array(lista[inicio:inicio+n])

        media = np.mean(grupo)
        desvio = np.std(grupo, ddof=1)

        prom.append(ufloat(media, desvio))

        inicio += n

    return prom


# Cantidad de mediciones correspondientes a cada campo
tamanos = [3, 4, 4]

SAR_prom = promedio_por_grupos(SAR_M17, tamanos)
tau_prom = promedio_por_grupos(tau_M17, tamanos)
Hc_prom  = promedio_por_grupos(Hc_M17, tamanos)
WR_prom  = promedio_por_grupos_bis(rates_M17, tamanos)

campos = ['H1', 'H2', 'H3']

print(f'Muestra = {nombre_M17}')
print(f'Concentracion = {conc_M17:.1f} g/L\n')

for campo, sar, tau, hc, wr in zip(campos, SAR_prom, tau_prom, Hc_prom, WR_prom):

    print(f'{campo}:')
    print(f'  ESAR = {sar:.3uS} W/g')
    print(f'  tau  = {tau:.1uS} ns')
    print(f'  Hc   = {hc:.1uS} kA/m')
    print(f'  WR   = {wr:.1uS} °C/s')


#%% Idem para M16
print(f'Muestra = {nombre_M16}')
print(f'Concentracion = {conc_M16:.1f} g/L')
SAR_prom = promedio_por_grupos(SAR_M16, tamanos)
tau_prom = promedio_por_grupos(tau_M16, tamanos)
Hc_prom  = promedio_por_grupos(Hc_M16, tamanos)
WR_prom  = promedio_por_grupos_bis(rates_M16, tamanos)

for campo, sar, tau, hc, wr in zip(campos, SAR_prom, tau_prom, Hc_prom, WR_prom):    
    print(f'{campo}:')
    print(f'  ESAR = {sar:.3uS} W/g')
    print(f'  tau  = {tau:.3uS} ns')
    print(f'  Hc   = {hc:.3uS} kA/m')
    print(f'  WR   = {wr:.1uS} °C/s')
#%%
#%% Guardar resumen de resultados

# Conversión Idc -> H0
H0_dict = {
    '050dA': 20,
    '075dA': 28,
    '100dA': 38,
    '125dA': 47,
    '150dA': 57,
    '152dA': 58,}

muestras = [{'nombre': nombre_M17,
        'conc': conc_M17,
        'resultados': resultados_M17,
        'SAR': SAR_M17,
        'tau': tau_M17,
        'Hc': Hc_M17,
        'WR': rates_M17,
        'archivo': 'Resumen_resultados_M17_260630.txt'
    },
    {
        'nombre': nombre_M16,
        'conc': conc_M16,
        'resultados': resultados_M16,
        'SAR': SAR_M16,
        'tau': tau_M16,
        'Hc': Hc_M16,
        'WR': rates_M16,
        'archivo': 'Resumen_resultados_M16_260630.txt'
    }
]

for muestra in muestras:

    with open(muestra['archivo'], 'w', encoding='utf-8') as f:

        f.write(f'Muestra = {muestra["nombre"]}\n')
        f.write(f'Concentracion = {muestra["conc"]:.1f} g/L\n\n')

        f.write(f'{"Medición":<9}{"H0 (kA/m)":<11}{"ESAR (W/g)":<18}{"tau (ns)":<18}{"Hc (kA/m)":<18}{"WR (°C/s)":<18}\n')
        f.write('-'*90 + '\n')

        for i, (res, sar, tau, hc,wr) in enumerate(
                zip(muestra['resultados'],
                    muestra['SAR'],
                    muestra['tau'],
                    muestra['Hc'],
                    muestra['WR']),
                start=1):

            # Buscar el H0 correspondiente
            H0 = np.nan
            for key, value in H0_dict.items():
                if key in res:
                    H0 = value
                    break

            sar_str = f'{sar:.3uS}'
            tau_str = f'{tau:.3uS}'
            hc_str  = f'{hc:.3uS}'
            wr_str  = f'{wr:.2f}'

            f.write(f'{i:<9}{H0:<11.0f}{sar_str:<18}{tau_str:<18}{hc_str:<18}{wr_str:<18}\n')

    print(f'Se guardó: {muestra["archivo"]}')
# %%
