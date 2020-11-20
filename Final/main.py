#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import cv2 # OpenCV
import matplotlib.pyplot as plt # Matplotlib
import numpy as np # Numpy
import scipy
from scipy import ndimage
from pylab import *
import scipy.signal as signal

#1 carregar uma imagem em escala de cinza
img = cv2.imread('gatineo.jpeg')

def colorToMono(img): 
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mono = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) #preto e branco
    fig, ax = plt.subplots(1, 2, figsize=(32, 32))

    fig.subplots_adjust(hspace=0, wspace=0)

    ax[1].xaxis.set_major_locator(plt.NullLocator())
    ax[1].yaxis.set_major_locator(plt.NullLocator())
    ax[1].imshow(cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB) ) 

    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[0].yaxis.set_major_locator(plt.NullLocator())
    ax[0].imshow(img)  
    return mono
  
mono =colorToMono(img)


#2 Função para receber um filtro e aplicar o 2D separável

def filtro_separavel(kernel,img):
    try:
        print(kernel.shape[1])
        img_linha = cv2.filter2D(img, -1, kernel, borderType=0) #filtrando 1d
        kernel.transpose() #transpondo o kernel
        img_final = cv2.filter2D(img_linha, -1, kernel, borderType=0) #aplicando o mesmo filtro, transposto
    except:   
        img_linha = cv2.filter2D(img, -1, kernel, borderType=0) #filtrando 1d
        kernel = np.array([kernel]) #necessario para vetores 1D
        kernel.transpose() #transpondo o kernel
        img_final = cv2.filter2D(img_linha, -1, kernel,borderType=0) #aplicando o mesmo filtro, transposto  
        
    return(img_linha,img_final)


def trunca(result):
    l = len(result)
    x = arange(0,l)
    stem(x, result)
    ylabel('h[n]')
    xlabel(r'n (amostras)')
    title(r'Filtro derivador com 16 pontos em um intervalo simétrico e deslocado (Causal)')

#Calculando a inversa (h[n]), filtro derivativo

def foo(x):
    return -1j*((2j*np.pi*x*np.exp(2j*np.pi*x) - np.exp(2j*np.pi*x) + 1) / (2*np.pi*x**2))
    #return int((1/n))

result = []
n = []
for i in range(-500,500):
    result.append(foo(i))
    n.append(i)
    
#Plotando    
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='w', axisbelow=True)
ax.plot(n, result, 'k', alpha=0.95, lw=4, linestyle='-', label=' Filtro Derivativo Ideal h[n]')

ax.legend(loc='best')
ax.set_title("Filtro derivativo h[n] = 1/n, com fase nula.")
ax.set_xlabel('t')
ax.set_ylabel(' Filtro Derivativo Ideal h[n]')
ax.set_ylim(-2,2)

ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='0.45', lw=0.25, ls='-')
legend = ax.legend(prop={'size': 12})
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(True)

plt.show()

#Truncando em 16 pontos em um intervalo simétrico e deslocando para ficar causal:
#a = signal.firwin(16, cutoff = 0.3, window = "boxcar")

fir =np.zeros(16)
for i in range(-8,9):
    if i==0:
        fir[8] = 1
    else:
        if i==8:
            fir[15]=(1/i)
        else:
            fir[i+8]= (1/i)
trunca(fir)


def plot_comparativo(filtro,img,mono):
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    linha, final =filtro_separavel(filtro, img) #derivativo proposto na imagem colorida
    linha2, final2 =filtro_separavel(filtro, mono) #derivativo proposto na imagem preto e branco

    fig, ax = plt.subplots(1, 4, figsize=(32, 32))

    fig.subplots_adjust(hspace=0, wspace=0)

    ax[3].xaxis.set_major_locator(plt.NullLocator())
    ax[3].yaxis.set_major_locator(plt.NullLocator())
    ax[3].imshow(cv2.cvtColor(final2,cv2.COLOR_GRAY2RGB)) 

    ax[2].xaxis.set_major_locator(plt.NullLocator())
    ax[2].yaxis.set_major_locator(plt.NullLocator())
    ax[2].imshow(final)    

    ax[1].xaxis.set_major_locator(plt.NullLocator())
    ax[1].yaxis.set_major_locator(plt.NullLocator())
    ax[1].imshow(linha)

    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[0].yaxis.set_major_locator(plt.NullLocator())
    ax[0].imshow(img)


#testando para filtro de média, filtro gaussiano, sharpening e derivativo proposto por nós (h[n])
#filtro =  np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])  #sharpening 
#filtro = np.ones((1, 20), dtype="float32")/20   #media
gauss = cv2.getGaussianKernel(ksize=9, sigma=0)  #gaussiano
sobel =  np.array([[-1,0,1],[-2, 0, 2],[-1,0,1]])# kernel derivador ideal exemplo



#Aplicando o filtro derivativo 2d separavel
plot_comparativo(fir,img,mono)

#4 Aplicação do filtro passa todas

#Filtro passa todas para os 6 polos
polos = np.linspace(-0.75, 0.75, 7)

fig1, ax11 = plt.subplots(figsize=(10, 7))
ax11.grid()
plt.title(r'Resposta em frequência, para vários polos $\lambda$')
plt.ylim(-1, 1)
plt.ylabel('amplitude (dB)', color='C0')
plt.xlabel(r'Frequencia normalizada (x$\pi$rad/amostra)')

for i, wf in enumerate(polos):
    w, h = scipy.signal.freqz([-wf, 1.0], [1.0, -wf])
    ax11.plot(w/max(w), 20 * np.log10(abs(h)), 'C{0}'.format(i), label=wf) #plotando a amplitude
ax21 = ax11.twinx()
plt.ylabel('Fase (rad)', color='g')

for i, wf in enumerate(polos):
    w, h = scipy.signal.freqz([-wf, 1.0], [1.0, -wf]) 
    angles = np.unwrap(np.angle(h))                     #plotando a fase
    ax21.plot(w/max(w), angles, 'C{0}'.format(i), label=wf)

ax11.legend(loc='upper right')

##############################################

fig3, ax3 = plt.subplots(figsize=(10, 7))
plt.title(r'Resposta ao impulso para diversos polos, varied $\lambda$') 
l = len(h)
x_axis = np.arange(l)
impulse = np.zeros(l); impulse[0] = 1.0
plt.xlabel('n (amostras)')
plt.ylabel('Amplitude da resposta ao impulso', color='b')

#Plotando as inversas
for i, wf in enumerate(polos):
    w, h = scipy.signal.freqz([-wf, 1.0], [1.0, -wf])
    inversa = np.fft.ifft(h)
    plt.plot(x_axis, np.real(inversa), label='h[n], polo={0}'.format(wf), color='C{0}'.format(i), alpha=0.9)

plt.legend(loc='upper right')
ax4 = ax3.twinx()
plt.show()

#Aplicando os filtros para cada polo, na ordem decrescente
for i, wf in enumerate(polos):
    w, h = scipy.signal.freqz([-wf, 1.0], [1.0, -wf])   
    if wf==0.0:
        continue     
    inversa = np.fft.ifft(h)*0.25  
    plot_comparativo(np.abs(inversa),img,mono)

#5 Detector de bordas

#aplicando um gaussiano antes:
orig1, orig2 = filtro_separavel(gauss,mono)
img1, img2 = filtro_separavel(flip(fir)*-1,mono) #aplicando filtro derivador na linha e na coluna

mag_grad = np.sqrt(np.square(img1) + np.square(img2)) #obtendo o modulo dos vetores de derivada
mag_grad *= 255 / mag_grad.max()

fig, ax = plt.subplots(1, 4, figsize=(32, 32))

fig.subplots_adjust(hspace=0, wspace=0)

ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[0].yaxis.set_major_locator(plt.NullLocator())
ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 

ax[1].xaxis.set_major_locator(plt.NullLocator())
ax[1].yaxis.set_major_locator(plt.NullLocator())
ax[1].imshow(img1,cmap="gray")    

ax[2].xaxis.set_major_locator(plt.NullLocator())
ax[2].yaxis.set_major_locator(plt.NullLocator())
ax[2].imshow(img2, cmap="gray")

ax[3].xaxis.set_major_locator(plt.NullLocator())
ax[3].yaxis.set_major_locator(plt.NullLocator())
ax[3].imshow(mag_grad.astype(int),cmap = "gray")

