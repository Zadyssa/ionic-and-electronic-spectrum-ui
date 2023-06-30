# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:50:37 2020

Raymond Diab
"""
#Import all necessary modules
import PIL.Image as Img
import numpy as np
import os
from matplotlib import cm
from tkinter.messagebox import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import *
import math
from tkinter import *
from tkinter.filedialog import *
from tkinter.messagebox import *


image_i=None #variable globale pour stocker le path de l'image ionic à utiliser
width_i = None  # Variable globale pour stocker la largeur de l'image ionic
height_i = None  # Variable globale pour stocker la hauteur de l'image ionic

image_e=None #variable globale pour stocker le path de l'image eletronic à utiliser
width_e = None  # Variable globale pour stocker la largeur de l'image electronic
height_e = None  # Variable globale pour stocker la hauteur de l'image electronic


###permet de choisir l'image à étudier directement depuis l'interface##

def select_image_ionic():
    global image_i, width_i, height_i
    filename = askopenfilename(title="Sélectionnez une image type ionic", filetypes=(("Fichiers images", "*.png;*.jpg;*.jpeg;*.tif"), ("Tous les fichiers", "*.*")))
    print("filename ", filename)
    if filename:
        image_i = Img.open(filename)
        width_i, height_i = image_i.size


def select_image_electronic():
    global image_e, width_e, height_e
    filename = askopenfilename(title="Sélectionnez une image type electronic", filetypes=(("Fichiers images", "*.png;*.jpg;*.jpeg"), ("Tous les fichiers", "*.*")))
    print("filename ", filename)
    if filename:
        image_e = Img.open(filename)
        width_e, height_e = image_e.size


fenetre = Tk()
fenetre.title('Test')
fenetre.geometry("2000x2000")


#electron parameters
Te1 = DoubleVar()
ne1 = DoubleVar()
u_e1 = DoubleVar()

#Parameters for Tse
spectral_width_Tse1 = DoubleVar() # [nm] Used to account for the instrument's precision
l_min_Tse1 = DoubleVar() #[nm]
l_max_Tse1 = DoubleVar() #[nm]
nlambda_Tse1 = DoubleVar() # number for points

#Parmeters for Tsi
spectral_width_Tsi1 = DoubleVar() # [nm] Used to account for the instrument's precision
l_min_Tsi1 = DoubleVar() #[nm]
l_max_Tsi1 = DoubleVar() #[nm]
nlambda_Tsi1 = DoubleVar() # number of points

#Ti ion with or without H
Ti1=DoubleVar() #[KeV]
value = StringVar() #pour décider de si c'est avec ou sans hydrogène
u_i1 = DoubleVar()

#normalisation 
norm = 1.0
dim_i=dim_e = 15

#wavelength range
cut1=DoubleVar()
cut2=DoubleVar()


##Fonction pour rentrer les variables qui changent d'un tir ou d'une expérience à l'autre##

def parameters():
    global dim1_e,dim2_e,dim1_i,dim2_i
    root = Toplevel(fenetre)
    root.title("Paramètres")


    #cadre
    cadre1=Frame(root,bg="white",bd=4,relief=GROOVE)
    cadre1.grid(padx=10,pady=10)
    cadre2=Frame(root,bg="white",bd=4,relief=GROOVE)
    cadre2.grid(padx=10,pady=10)
    cadre3=Frame(root,bg="white",bd=4,relief=GROOVE)
    cadre3.grid(padx=10,pady=10)
    cadre4=Frame(root,bg="white",bd=4,relief=GROOVE)
    cadre4.grid(padx=10,pady=10)
    cadre5=Frame(root,bg="white",bd=4,relief=GROOVE)
    cadre5.grid(padx=10,pady=10)



    # texte devant la case Electron
    a_label = Label(cadre1, text='Te [KeV]  ', font=('calibre', 10, 'bold'))
    b_label = Label(cadre1, text='ne en cm-3', font=('calibre', 10, 'bold'))
    m_label=Label(cadre1,text="u_e",font=('calibre', 10, 'bold'))
    
    #texte devant case Tse
    c_label=Label(cadre2,text="l_min_Tse [nm]",font=('calibre', 10, 'bold'))
    d_label=Label(cadre2,text="l_max_Tse [nm]",font=('calibre', 10, 'bold'))
    e_label=Label(cadre2,text="spectral_width_Tse [nm]",font=('calibre', 10, 'bold'))
    f_label=Label(cadre2,text="nlambda_Tse [nombre de points]",font=('calibre', 10, 'bold'))

    #texte devant case Tsi
    h_label=Label(cadre3,text="l_min_Tsi [nm]",font=('calibre', 10, 'bold'))
    i_label=Label(cadre3,text="l_max_Tsi [nm]",font=('calibre', 10, 'bold'))
    j_label=Label(cadre3,text="spectral_width_Tsi [nm]",font=('calibre', 10, 'bold'))
    k_label=Label(cadre3,text="nlambda_Tsi [nombre de points]",font=('calibre', 10, 'bold'))

    #texte devant case Ti avec ou sans H
    l_label=Label(cadre4,text="Ti [KeV]",font=('calibre', 10, 'bold'))
    n_label=Label(cadre4,text="u_i",font=('calibre', 10, 'bold'))

    #lambda_0 et theta et cut
    o_label=Label(cadre5,text="lambda_0 [nm]",font=('calibre',10,'bold'))
    p_label=Label(cadre5,text="theta [deg]",font=('calibre',10,'bold'))
    q_label=Label(cadre5,text="wavelength down [nm]",font=('calibre', 10, 'bold'))
    r_label=Label(cadre5,text="wavelength up",font=('calibre', 10, 'bold'))


    # rentrer les données
    #Electron
    a = Entry(cadre1, textvariable=Te1)
    b = Entry(cadre1, textvariable=ne1)
    m=Entry(cadre1,textvariable=u_e1)
    #Tse
    c=Entry(cadre2,textvariable=l_min_Tse1)
    d=Entry(cadre2,textvariable=l_max_Tse1)
    e=Entry(cadre2,textvariable=spectral_width_Tse1)
    f=Entry(cadre2,textvariable=nlambda_Tse1)
    #Tsi
    h=Entry(cadre3,textvariable=l_min_Tsi1)
    i=Entry(cadre3,textvariable=l_max_Tsi1)
    j=Entry(cadre3,textvariable=spectral_width_Tsi1)
    k=Entry(cadre3,textvariable=nlambda_Tsi1)
    #Ti
    l=Entry(cadre4,textvariable=Ti1)
    n=Entry(cadre4,textvariable=u_i1)

    #lambda_0 et theta et cut
    o=Entry(cadre5,textvariable=lambda_01)
    p=Entry(cadre5,textvariable=theta1)
    q=Entry(cadre5,textvariable=cut1)
    r=Entry(cadre5,textvariable=cut2)

    #focus
    a.focus_set()
    b.focus_set()
    c.focus_set()
    d.focus_set()
    e.focus_set()
    f.focus_set()
    h.focus_set()
    i.focus_set()
    j.focus_set()
    k.focus_set()
    l.focus_set()
    m.focus_set()
    n.focus_set()
    o.focus_set()
    p.focus_set()
    q.focus_set()
    r.focus_set()

    #bouton final
    sub_btn = Button(root, text='Submit', command=lambda: root.destroy())
    sub_btn.grid(row=2, column=1)

    # bouton cadre 1 (electron)
    a_label.grid(row=0, column=0)
    b_label.grid(row=1, column=0)
    m_label.grid(row=2,column=0)
    a.grid(row=0, column=1)
    b.grid(row=1, column=1)
    m.grid(row=2,column=1)

    #bouton cadre 2 (Tse)
    c_label.grid(row=0,column=0)
    d_label.grid(row=1,column=0)
    e_label.grid(row=2,column=0)
    f_label.grid(row=3,column=0)
    c.grid(row=0, column=1)
    d.grid(row=1, column=1)
    e.grid(row=2, column=1)
    f.grid(row=3, column=1)

    #bouton cadre 3 (Tsi)
    h_label.grid(row=0,column=0)
    i_label.grid(row=1,column=0)
    j_label.grid(row=2,column=0)
    k_label.grid(row=3,column=0)
    h.grid(row=0, column=1)
    i.grid(row=1, column=1)
    j.grid(row=2, column=1)
    k.grid(row=3, column=1)

    #bouton cadre 4 (Ti)
    l_label.grid(row=0,column=0)
    n_label.grid(row=1,column=0)
    l.grid(row=0, column=1)
    n.grid(row=1,column=1)
    bouton1 = Radiobutton(cadre4, text="Avec H", variable=value, value="1")
    bouton2 = Radiobutton(cadre4, text="Sans H", variable=value, value="2")
    bouton1.grid(row=2,column=0)
    bouton2.grid(row=2,column=1)

    #bouton cadre  5 (lambda_0 et theta)
    o_label.grid(row=0,column=0)
    p_label.grid(row=1,column=0)
    q_label.grid(row=2,column=0)
    r_label.grid(row=3,column=0)
    o.grid(row=0,column=1)
    p.grid(row=1,column=1)
    q.grid(row=2,column=1)
    r.grid(row=3,column=1)


    

    root.wait_window(root)

    dim1_i=dim1_e=lambda_01.get()-2.0
    dim2_i=dim2_e=lambda_01.get()+2.0

    #variables retournées 

    return Te1.get(), ne1.get(),spectral_width_Tse1.get(),spectral_width_Tsi1.get(),l_min_Tse1.get(),l_min_Tsi1.get(),l_max_Tse1.get(),l_max_Tsi1.get(),nlambda_Tsi1.get(),nlambda_Tse1.get(),value.get(),u_e1.get(),u_i1.get(),lambda_01.get(),theta1.get(),cut1.get(),cut2.get()





def quit_program():
    # Fermer toutes les fenêtres
    fenetre.destroy()
    for window in fenetre.winfo_children():
        if isinstance(window, Toplevel):
            window.destroy()


##Fonctions pour "translater" la courbe expérimentale par rapport à la courbe théorique, liées à des boutons##

def incrementationmax():
    a=u_i1.get()*10
    b=u_e1.get()*10
    u_i1.set(a)
    u_e1.set(b)

def incrementationmin():
    #pour u_i
    decimal_parti, power_parti = math.modf(u_i1.get())
    decimal_parti+=0.1
    a=decimal_parti+power_parti
    #pour u_e
    decimal_parte, power_parte = math.modf(u_e1.get())
    decimal_parte+=0.1
    b=decimal_parte+power_parte

    #mettre à jour les valeurs
    u_i1.set(a)
    u_e1.set(b)

def decrementationmax():
    a=u_i1.get()*0.1
    b=u_e1.get()*0.1
    u_i1.set(a)
    u_e1.set(b)

def decrementationmin():
    #pour u_i
    decimal_parti, power_parti = math.modf(u_i1.get())
    decimal_parti-=0.1
    a=decimal_parti+power_parti
    #pour u_e
    decimal_parte, power_parte = math.modf(u_e1.get())
    decimal_parte-=0.1
    b=decimal_parte+power_parte

    u_i1.set(a)
    u_e1.set(b)



menubar = Menu(fenetre)
menu4 = Menu(menubar, tearoff=0)
menu4.add_command(label="Variables", command=parameters)
menubar.add_cascade(label="Paramètres", menu=menu4)
menu1=Menu(menubar,tearoff=0)
menu1.add_command(label="Sélectionner image ionic",command=select_image_ionic)
menu1.add_command(label="Sélectionner image electronic",command=select_image_electronic)
menu1.add_command(label="Quitter", command=quit_program)
menubar.add_cascade(label="Fichier", menu=menu1)
fenetre.config(menu=menubar)




#Constants
c =2.99792e10 # [cm/s]
e = 1.602e-19 #electron charge [C]
kB = 1.38064e-23 #Boltzmann constant [J/K]
me = 9.11e-31 #[kg]
mp = 1.67e-27 #[kg]
eps0 = 8.854e-12 #[F/m]


lambda_01 = DoubleVar() # wavelength of probing laser [nm] #526.5
theta1 = DoubleVar() #angle between incoming and scattered radiation [deg] #90



#Parmeters for Tsi
ZCs = np.load("charge_state_C.npy")
ZFs = np.load("charge_state_F.npy")

#Extrapolates the value of Z from a list of values
#which can be generated in charges.py
def extrapolate_Z(Z,T):
    i= np.argmin(np.abs(Z[0]-T*1e3))
    if Z[0,i]-T*1e3 >0: i -= 1
    a = (Z[1,i+1]-Z[1,i])/(Z[0,i+1]-Z[0,i])
    return a*(T*1e3 - Z[0,i])+Z[1,i]
  

########If you need a second spectrum (ex. broad and narrow for cold
######## and hot regions, you can uncomment the lines below)
########### Elecron parameters ####################
#Te2 = 0.12 # [KeV]
#ne2 = 2.77e18 # [cm-3]
#u_e2 = 1e7# Drift velocity [cm/s]

########## Ion parameters without H ###############
#N_ions2 = 2 #Number of ions
#names2 = ['C','F']
#Ti2 = [0.05,0.05] #[KeV]
#Zi2 = [extrapolate_Z(ZCs, Ti[0]), extrapolate_Z(ZFs,Ti[1])]
#print("Zi2 = ", Zi2)
#Ai2 = [12,19]
#prop2 = [0.34,0.66] #Proportion of ions
#u_i2 = [0e7,0e7] # Drift velocity [cm/s]

######## Ion parameters with H #############
# N_ions2 = 3 #Number of ions
# names2 = ['C','F','H']
# Ti2 =[0.06,0.06,0.06] #[KeV]
# Zi2 = [extrapolate_Z(ZCs, Ti2[0]), extrapolate_Z(ZFs,Ti2[1]),1]
# print("Zi2 = ", Zi2)
# Ai2 = [12,19,1]
# nH = 1.39e18
# x = (ne2 - nH*Zi2[2])/(nH*(0.66*Zi2[1]+0.34*Zi2[0]-Zi2[2])+ne2)
# prop2 = [0.34*x,0.66*x,1-x] #Proportion of ions
# u_i2 = [-0.1e7 for _ in range(N_ions2)] # Drift velocity [cm/s]


####### Experimental data ##########
LambdaScale = 2.6933 #[px/nm]
SpatialScale = 5.5981 #[microns/px] #5.5981

#variables pour changer la position des traits rouges
red1_i=red1_e=440 #initialisation 
red2_i=red2_e=460#initialisation 

def modif_rouge_haut1():
    global red1_i,red1_e
    if current_mode=="ionic":
        if red1_i<=cut2.get()-10:
            red1_i=red1_i+10
    else :
        if red1_e<cut2.get()-10:
            red1_i=red1_e+10

def modif_rouge_haut_vite1():
    global red1_i,red1_e
    if current_mode=="ionic":
        if red1_i<=cut2.get()-100:
            red1_i+=100
    else :
        if red1_e<=cut2.get()-100:
            red1_e+=100

def modif_rouge_bas1():
    global red1_i,red1_e
    if current_mode=="ionic":
        if red1_i>=cut1.get()+10:
            red1_i-=10
    else :
        if red1_e>=cut1.get()+10:
            red1_e-=10


def modif_rouge_bas_vite1():
    global red1_i,red1_e
    if current_mode=="ionic":
        if red1_i>=cut1.get()+100:
            red1_i-=100
    else :
        if red1_e>=cut1.get()+100:
            red1_e-=100


def modif_rouge_haut2():
    global red2_i,red2_e
    if current_mode=="ionic":
        if red2_i<=cut2.get()-10:
            red2_i=red2_i+10
    else :
        if red2_e<=cut2.get()-10:
            red2_e=red2_e+10


def modif_rouge_haut_vite2():
    global red2_i,red2_e
    if current_mode=="ionic":
        if red2_i<=cut2.get()-100:
            red2_i+=100
    else :
        if red2_e<=cut2.get()-100:
            red2_e+=100

def modif_rouge_bas2():
    global red2_i,red2_e
    if current_mode=="ionic":
        if red2_i>=cut1.get()+10:
            red2_i-=10
    else :
        if red2_e>=cut1.get()+10:
            red2_e-=10

def modif_rouge_bas_vite2():
    global red2_i,red2_e
    if current_mode=="ionic":
        if red2_i>=cut1.get()+100:
            red2_i-=100
    else :
        if red2_e>=cut1.get()+100:
            red2_e-=100



def normalisation_incr():
    global norm
    norm+=0.1    

def normalisation_decr():
    global norm
    if norm<0.1:
        norm =0.1
    else :
        norm-=0.1    
    
def modif_dim_moins():
    global dim_i,dim_e
    if current_mode=="ionic":
        if dim_i>0:
            dim_i-=1
    else :
        if dim_e>0:
            dim_e-=1
        

def modif_dim_plus():
    global dim_i,dim_e
    if current_mode=="ionic":
        dim_i+=1
    else : dim_e+=1


#You can use these to save your nomalized and treated experimental spectrum into a .npy file that you can use later.
#np.save("2d_tir35_2.npy",array)
#np.save("cut_tir35_2.npy",cut)

#You can use this to load a previously saved spectrum.
#array = np.load("2d_tir35_2.npy")
#cut = np.load("cut_tir35_2.npy")
##############################################################################################################
def zprime(x):
    #Calculate the derivative of the plasma dispertion function
    #Uses the table produced in ZpCalc.py using composite numerical integration with gaussian quadrature rule.
    z_prime = np.load("z_prime.npy")
    y = np.zeros(len(x),dtype = 'complex')
    for i in range(len(x)):
        if np.abs(x[i])<15:
           index = np.argmin(np.abs(z_prime[0]-x[i]))
           y[i] =  z_prime[1,index] #Use table in z_prime.npy to approximate the value of z_prime
           #We can do this as the function is continuous!
        else:
            y[i] = -1/(2*x[i]**2) - 3/(4*x[i]**4) # asymptotic expansion! See Fried & Conte p.3
    return y

def gaussian(x,sigma, norm):
    return np.exp(-x**2 /(2*(sigma**2)))/norm

def Maxwell(v_phase,v_thermal,v_drift):
    return  np.exp(-((v_phase - v_drift)/v_thermal)**2.) / (np.sqrt(np.pi) * v_thermal)

def chie(w,k,Ld_e,v_thermal,v_drift):
    z = (w - k * v_drift) /  (k * v_thermal)
    return zprime(z) / (k**2 * Ld_e**2.)
    
def chii(w,k,Ld_i,v_thermal,v_drift):
    z = (w - k*v_drift) /  (k * v_thermal)
    return zprime(z) / (k**2 * Ld_i**2)

#################################################################################################################
#Program for computing the spectrum 
def spectrum(mode): 

    theta=theta1.get()
    lambda_0=lambda_01.get()


    #Electron
    Te=Te1.get()
    ne=ne1.get()
    #Ti
    Tif=Ti1.get()
    #Tse ou Tsi : 
    if mode=="Tse":
        spectral_width = spectral_width_Tse1.get() # [nm] Used to account for the instrument's precision
        l_min = l_min_Tse1.get() #[nm]
        l_max = l_max_Tse1.get() #[nm]
        nlambda = int(nlambda_Tse1.get()) # number for points
    if mode=="Tsi":
        spectral_width = spectral_width_Tsi1.get() # [nm] Used to account for the instrument's precision
        l_min = l_min_Tsi1.get() #[nm]
        l_max = l_max_Tsi1.get() #[nm]
        nlambda = int(nlambda_Tsi1.get()) # number for p


    #variables
    #Avec H
    if value.get()=="1":
        N_ions = 3 #Number of ions
        Ti =[Tif,Tif,Tif] #est-ce que Ti est une liste de valeurs Te ? Est-ce que les trois valeurs peuvent être différentes ? Si oui, ajouter Ti2 et Ti3 à rentrer dans Ti

        Zi = [extrapolate_Z(ZCs, Ti[0]), extrapolate_Z(ZFs,Ti[1]),1]
        Ai = [12,19,1]
        nH = 0.75e18
        x = (ne - nH*Zi[2])/(nH*(0.66*Zi[1]+0.34*Zi[0]-Zi[2])+ne)#ions parameters with H
        prop = [0.34*x,0.66*x,1-x] #Proportion of ions 
    #Sans H
    if value.get()=="2":
        N_ions = 2 #Number of ions
        Ti =[Tif,Tif] #est-ce que Ti est une liste de valeurs Te ? Est-ce que les trois valeurs peuvent être différentes ? Si oui, ajouter Ti2 et Ti3 à rentrer dans Ti #0.19 dans le truc précédent
        Zi = [extrapolate_Z(ZCs, Ti[0]), extrapolate_Z(ZFs,Ti[1])]
        Ai = [12,19]
        prop = [0.34,0.66] #Proportion of ions

   
    u_e=u_e1.get()
    u_i = [u_i1.get() for _ in range(N_ions)] # Drift velocity [cm/s]
    spectral_width = spectral_width*1e-7
    sigma = spectral_width/(2*np.sqrt(2*np.log(2))) #spectral width = FWHM
    lambda_0 = lambda_0*1e-7 #conversion to [cm]
    theta = theta*np.pi/180 #conversion to [rad]


    
    
    #ions
    Zavr = sum(prop[i]*Zi[i] for i in range(N_ions))
    ni = [prop[i]*ne/Zavr for i in range(N_ions)] #intervention dudit ne
    vth_i = [4.376e7 * np.sqrt(Ti[i]/Ai[i]) for i in range(N_ions)]
    Ld_i = [np.sqrt(5.5364e8 * Ti[i] /(Zi[i]**2 * ni[i])) for i in range(N_ions)]
    #wp_i = [vth_i[i]/(np.sqrt(2)*Ld_i[i]) for i in range(N_ions)] 
    

    #avant que l'utilisateur rentre les variables, elles sont initialisées à 0.0. Évite de faire crash le programme avant qu'il commence 
    if ne!=0 : 
        Ld_e = np.sqrt(5.5364e8 * Te / ne) # Debye length [cm]
    else :
        Ld_e = np.sqrt(5.5364e8 * Te /1) # Debye length [cm]
    vth_e = 1.8755e9* np.sqrt(Te) # Thermal velocity [cm/s]
    wp_e = vth_e/(np.sqrt(2)*Ld_e) # Plasma frequency [rad/s]
    
    dlambda = (l_max - l_min)/(nlambda - 1) # separation between wavelengths
    lambda_s = np.array([l_min + i*dlambda for i in range(nlambda)]) # wavelengths of scattered light
    lambda_s = lambda_s*1e-7 # conversion to [cm]
    w_0 = 2*np.pi*c/lambda_0
    k_s = 2*np.pi/lambda_s # wavenumbers of scattered light [cm-1]
    k_0 = np.sqrt(w_0**2-wp_e**2)/c # wavenumber of probing beam [cm-1]
    w_s = c*k_s
    k = np.sqrt(k_s**2 + k_0**2 - 2*k_s*k_0*np.cos(theta)) # wavenumber of plasma waves
    w = w_0 - w_s
    #w_EPW = np.sqrt( wp_e**2 + 3* (vth_e * 2 * k_0 * np.sin(theta/2))**2 / 2)
    
    # Calculation of the spectral form function S(k,w)
    Xi = np.zeros((N_ions, nlambda),dtype = 'complex')
    Xe = chie(w, k, Ld_e, vth_e, u_e)
    eps = 1 + Xe
    
    for i in range(N_ions):
        Xi[i] = chii(w, k, Ld_i[i], vth_i[i], u_i[i])
        eps = eps + Xi[i]
    
    vphase = w / k # phase velocity
    f_e = Maxwell(vphase, vth_e, u_e)
    
    # Electronic contribution
    S = 2*np.pi*f_e*np.abs(1 - Xe/eps)**2 / k
    
    # Ionic contribution
    for i in range(N_ions):
        f_i = Maxwell(vphase, vth_i[i], u_i[i] )
        S = S + 2* np.pi * (f_i * abs(Xe/eps)**2 * Zi[i]**2 * ni[i]/ne) / k
     
    # Use only for Tse. 
    # Removes central Tsi part
    if mode == "Tse":
        imin = np.argmin(np.abs(lambda_s*1e7-lambda_0*1e7 +2))
        imax = np.argmin(np.abs(lambda_s*1e7-lambda_0*1e7 -2))
        S[imin:imax] = 0 #remove Tsi
        
    S = S/np.amax(S)  # Normalize spectrum
    
    # Account for instrument's precision  
    numx = lambda_s - lambda_s[len(lambda_s)//2] #center the gaussian in the array to prevent shifting
    gauss = gaussian(numx,sigma,norm = 1)
    spectre = np.convolve(S,gauss,mode = 'same')
    spectre = spectre/np.amax(spectre)
    
    return lambda_s,S,spectre

########################################################################################
########## plot the 2D experimental spectrum #############

### Use if you want two spectra
# narrow = spectre*0.45 #relative intensity
# broad = spectre2*0.55 #relative intensity
# full = broad+narrow
# plt.plot(lambda_s*1e7,broad,label = "broad")
# plt.plot(lambda_s*1e7,narrow,label = "narrow")
# plt.plot(lambda_s*1e7,full,label = "narrow+broad")
# Ratio = np.amax(narrow)/np.amax(broad)
# print("Ratio = ", Ratio)
# plt.legend(fontsize = 16)
# plt.title("TS ion",fontsize = 20)
# plt.xlabel("$\lambda$ (nm)", fontsize = 16)
# plt.ylabel("Intensity (a.u.)", fontsize = 16)

############# Save data in text file #############
#np.savetxt("lambda.txt",lambda_s*1e7)
#np.savetxt("unconvolved.txt", S)
#np.savetxt("spectre.txt",spectre)


##modification de la wavelength range pour centrer l'image sur le lambda_0##

#variable dimension wavelength range
dim1_i=dim1_e=lambda_01.get()-2.0
dim2_i=dim2_e=lambda_01.get()+2.0

#première variable 
def modif_wavelength_plus1():
    global dim1_i,dim1_e
    if current_mode =="ionic": #permet de tester quelle image (electronic ou ionic) sera modifiée
        dim1_i+=0.1
    else : dim1_e+=0.1

def modif_wavelength_plusg1():
    global dim1_i,dim1_e
    if current_mode=="ionic":
        dim1_i+=1.0
    else : dim1_e+=1.0

def modif_wavelength_moins1():
    global dim1_i,dim1_e
    if current_mode=="ionic":
        dim1_i-=0.1
    else :
        dim1_e-=0.1

def modif_wavelength_moinsg1():
    global dim1_i,dim1_e
    if current_mode=="ionic":
        dim1_i-=1.0
    else :
        dim1_e-=1.0

#deuxième variable 
def modif_wavelength_plus2():
    global dim2_i,dim2_e
    if current_mode=="ionic":
        dim2_i+=0.1
    else :
        dim2_e+=0.1

def modif_wavelength_plusg2():
    global dim2_i,dim2_e
    if current_mode=="ionic":
        dim2_i+=1.0
    else :
        dim2_e+=1.0


def modif_wavelength_moins2():
    global dim2_i,dim2_e
    if current_mode=="ionic":
        dim2_i-=0.1
    else :
        dim2_e-=0.1

def modif_wavelength_moinsg2():
    global dim2_i,dim2_e
    if current_mode=="ionic":
        dim2_i-=1.0
    else :
        dim2_e-=1.0


##plot la courbe electronic##

def plot_electronic(): 

    global cut,red1_e,red2_e,image_e
    window=Toplevel(fenetre)
    window.geometry("800x800+1000+50")
    window.title("Eletronic")

    array = np.array(image_e)

    cut = sum(array[i] for i in range(red1_e,red2_e))/20 #temporal average
    cut = cut[int(cut1.get()):int(cut2.get())] #Take the region corresponding to the wavelengths that you want

    # #smoothening by moving average - if needed
    # for i in range(len(cut)):
    #     m= max(0,i-4)
    #     M = min(len(cut)-1,i+4)
    #     cut[i] = sum(cut[j] for j in range(m,M+1))/(M+1-m)
        
    zero_level = min(cut[:50])
    cut = cut-zero_level  #Remove zero level
    for i in range(len(cut)):
        if cut[i]<0: cut[i] = 0 #because the intensity of central hidden region might be lower than the background

    cut = cut/965 #normalization of experimental spectrum
    lambda_s,S,spectre = spectrum("Tse")
    x_cut_e = np.linspace(l_min_Tse1.get(),l_max_Tse1.get(),len(cut/norm))
    fig3=Figure(figsize = (8,8),dpi=70)
    plot3=fig3.add_subplot(111)
    plot3.set_xlabel("$\lambda$ (nm)", fontsize = 16)
    plot3.set_ylabel("Intensity (a.u.)", fontsize = 16)
    plot3.set_title("TS electron",fontsize = 20)
    plot3.plot(x_cut_e,cut/norm,label = "experimental")
    plot3.plot(lambda_s*1e7,spectre, label = "theoretical")
    plot3.legend(fontsize=16)
     
    canvas=FigureCanvasTkAgg(fig3,master=window)
    canvas.draw()
    canvas.get_tk_widget().place(x=100,y=100)


##plot la courbe ionic##

def plot_ionic():

    global cut,image_i,red1_i,red2_i
    window=Toplevel(fenetre)
    window.geometry("800x800+150+50")
    window.title("Ionic")
    array = np.array(image_i)

    cut = sum(array[i] for i in range(red1_i,red2_i))/20 #temporal average
    cut = cut[int(cut1.get()):int(cut2.get())] #Take the region corresponding to the wavelengths that you want

    # #smoothening by moving average - if needed
    # for i in range(len(cut)):
    #     m= max(0,i-4)
    #     M = min(len(cut)-1,i+4)
    #     cut[i] = sum(cut[j] for j in range(m,M+1))/(M+1-m)
        
    zero_level = min(cut[:50])
    cut = cut-zero_level  #Remove zero level
    for i in range(len(cut)):
        if cut[i]<0: cut[i] = 0 #because the intensity of central hidden region might be lower than the background

    cut = cut/965 #normalization of experimental spectrum
    lambda_s,S,spectre = spectrum("Tsi")
    x_cut_i = np.linspace(l_min_Tsi1.get(),l_max_Tsi1.get(),len(cut/norm))
    fig2=Figure(figsize=(8,8),dpi=70)
    plot2=fig2.add_subplot(111) #111 dans la fenêtre (8x8)x70 créée dans l'interface. Pas la position dans l'interface par rapport à un subplot normal par exemple
    plot2.set_title("TS ion",fontsize = 20)
    plot2.plot(x_cut_i,cut/norm,label = "experimental")
    plot2.plot(lambda_s*1e7,spectre, label = "theoretical")
    plot2.set_xlabel("$\lambda$ (nm)", fontsize = 16)
    plot2.set_ylabel("Intensity (a.u.)", fontsize = 16)
    plot2.legend(fontsize=16)

    canvas=FigureCanvasTkAgg(fig2,master=window)
    canvas.draw()
    canvas.get_tk_widget().place(x=100,y=100)

##savoir quoi plot lorsque l'on change les paramètres des images (et non des spectres)##

current_mode = "ionic"

def update_mode(mode):
    global current_mode
    current_mode = mode


plot_button2=Button(master=fenetre,command=lambda:[plot_electronic(),plot_ionic()],height=2,width=10,text="Spectrums")
plot_button2.place(x=100,y=700)


plot_button3=Button(master=fenetre,command=lambda:update_mode("ionic"),height=2,width=8,text="IONIC")
plot_button4=Button(master=fenetre,command=lambda:update_mode("electronic"),height=2,width=8,text="ELECTRONIC")
plot_button3.place(x=1300,y=850)
plot_button4.place(x=1390,y=850)

#Boutons liés à ces fonctions placés dans la fenêtre principale sous le tracé de courbe

cadre=Frame(fenetre,bg="green",bd=4,relief=GROOVE)
cadre.place(x=300,y=850)

titre=Label(cadre,text="courbes")
titre.grid(row=0,column=0)

#boutons translation

btn1=Button(cadre,command=lambda:[incrementationmin(),plot_ionic(),plot_electronic()],text="droite")
btn2=Button(cadre,command=lambda:[incrementationmax(),plot_ionic(),plot_electronic()],text="DROITE")
btn3=Button(cadre,command=lambda:[decrementationmin(),plot_ionic(),plot_electronic()],text='gauche')
btn4=Button(cadre,command=lambda:[decrementationmax(),plot_ionic(),plot_electronic()],text='GAUCHE')
btn6=Button(cadre,command=lambda:[normalisation_incr(),plot_ionic(),plot_electronic()],text='BAS')
btn5=Button(cadre,command=lambda:[normalisation_decr(),plot_ionic(),plot_electronic()],text='HAUT') 
btn1.grid(row=3,column=4)
btn2.grid(row=3,column=5)
btn3.grid(row=3,column=2)#
btn4.grid(row=3,column=0)
btn5.grid(row=0,column=3)
btn6.grid(row=5,column=3)

#bouton translation rouge red1
cadre1=Frame(fenetre,bg="red",bd=4,relief=GROOVE)
cadre1.place(x=1830,y=700)

titre1=Label(cadre1,text="inf")
titre1.grid(row=0,column=0)

btn7=Button(cadre1,command=lambda:[modif_rouge_haut1(),plot_image()],text="haut",height=2,width=5)
btn8=Button(cadre1,command=lambda:[modif_rouge_haut_vite1(),plot_image()],text="HAUT",height=2,width=5)
btn9=Button(cadre1,command=lambda:[modif_rouge_bas1(),plot_image()],text="bas",height=2,width=5)
btn10=Button(cadre1,command=lambda:[modif_rouge_bas_vite1(),plot_image()],text="BAS",height=2,width=5)
btn7.grid(row=2,column=0)
btn8.grid(row=1,column=0)
btn9.grid(row=3,column=0)
btn10.grid(row=4,column=0)

#frame translation rouge red2
cadre2=Frame(fenetre,bg="red",bd=4,relief=GROOVE)
cadre2.place(x=1830,y=200)


titre2=Label(cadre2,text="sup")
titre2.grid(row=0,column=0)

btn11=Button(cadre2,command=lambda:[modif_rouge_haut2(),plot_image()],text="haut",height=2,width=5)
btn12=Button(cadre2,command=lambda:[modif_rouge_haut_vite2(),plot_image()],text="HAUT",height=2,width=5)
btn13=Button(cadre2,command=lambda:[modif_rouge_bas2(),plot_image()],text="bas",height=2,width=5)
btn14=Button(cadre2,command=lambda:[modif_rouge_bas_vite2(),plot_image()],text="BAS",height=2,width=5)

btn11.grid(row=2,column=0)
btn12.grid(row=1,column=0)
btn13.grid(row=3,column=0)
btn14.grid(row=4,column=0)

#bouton changer nb de points abscisse

btn15=Button(master=fenetre,command=lambda:[modif_dim_plus(),plot_image()],height=2,width=4,text="PLUS")
btn16=Button(master=fenetre,command=lambda:[modif_dim_moins(),plot_image()],height=2,width=4,text="MOINS")

btn15.place(x=1810,y=530)
btn16.place(x=1860,y=530)


##changer la wavelenght##
cadre3=Frame(fenetre,bg="blue",bd=4,relief=GROOVE)
cadre3.place(x=800,y=850)

titre3=Label(cadre3,text="wavelength range")
titre3.grid(row=0,column=0)

stitre2 =Label(cadre3,text="inf")
stitre2.grid(row=1,column=0)

stitre1=Label(cadre3,text="sup")
stitre1.grid(row=2,column=0)

#boutons
btn17=Button(cadre3,command=lambda:[modif_wavelength_moins1(),plot_image()],text="moins")
btn18=Button(cadre3,command=lambda:[modif_wavelength_moinsg1(),plot_image()],text="MOINS")
btn19=Button(cadre3,command=lambda:[modif_wavelength_plus1(),plot_image()],text="plus")
btn20=Button(cadre3,command=lambda:[modif_wavelength_plusg1(),plot_image()],text="PLUS")

btn21=Button(cadre3,command=lambda:[modif_wavelength_moins2(),plot_image()],text="moins")
btn22=Button(cadre3,command=lambda:[modif_wavelength_moinsg2(),plot_image()],text="MOINS")
btn23=Button(cadre3,command=lambda:[modif_wavelength_plus2(),plot_image()],text="plus")
btn24=Button(cadre3,command=lambda:[modif_wavelength_plusg2(),plot_image()],text="PLUS")

btn17.grid(row=1,column=2)
btn18.grid(row=1,column=1)
btn19.grid(row=1,column=3)
btn20.grid(row=1,column=4)

btn21.grid(row=2,column=2)
btn22.grid(row=2,column=1)
btn23.grid(row=2,column=3)
btn24.grid(row=2,column=4)


#plot l'image de la photo expérimentale
def plot_image():
    global image_i,width_i, height_i
    global image_e,width_e,height_e
    global red1_i,red2_i,red1_e,red2_e

    global cut,dim1_i,dim1_e,dim2_i,dim2_e,dim_i,dim_e
    if image_i is not None and image_e is not None :

        #ionic
        array_i = np.array(image_i)
        fig1=Figure(figsize = (8,8),dpi=100)
        plot1=fig1.add_subplot(111)
        #red line for position determination on 2D plot
        plot1.plot(np.zeros(width_i)+red1_i, color = 'red')
        plot1.plot(np.zeros(width_i)+red2_i, color = 'red')
        cut = sum(array_i[i] for i in range(red1_i,red2_i))/20 #temporal average
        cut = cut[322:840] #Take the region corresponding to the wavelengths that you want


        im=plot1.imshow(array_i,origin="lower", interpolation="None",cmap= cm.jet) #changé 0 pour lower car erreur apparemment 
        x = np.linspace(0,width_i,dim_i) ## 1344 pixels
        xs = np.linspace(dim1_i,dim2_i,dim_i) ### wavelength range 524.25 - 529.45
        y = np.linspace(0,height_i,dim_i) ## 1024 pixels
        ys = y*SpatialScale
        xtab = np.array([round(xs[i],1) for i in range(len(xs))])
        ytab = np.array([round(ys[i],1) for i in range(len(ys))])
        plot1.set_xticks(x)
        plot1.set_yticks(y)
        plot1.set_xticklabels(xtab)
        plot1.set_yticklabels(ytab)
        plot1.set_xlabel(r"$\rm wavelength [nm]$",fontsize = 16)
        plot1.set_ylabel(r"$\rm y [\mu m]$",fontsize = 16)
        plot1.set_title("TS ion",fontsize = 17)

        fig1.colorbar(im)

        canvas=FigureCanvasTkAgg(fig1,master=fenetre)
        canvas.draw()
        canvas.get_tk_widget().place(x=300,y=0)


        #eletronic 

        array_e = np.array(image_e)
        fig2=Figure(figsize = (8,8),dpi=100)
        plot2=fig2.add_subplot(111)
        #red line for position determination on 2D plot
        plot2.plot(np.zeros(width_e)+red1_e, color = 'red')
        plot2.plot(np.zeros(width_e)+red2_e, color = 'red')
        cut = sum(array_e[i] for i in range(red1_e,red2_e))/20 #temporal average
        cut=list(cut)
        cut = cut[int(cut1.get()):int(cut2.get())] #Take the region corresponding to the wavelengths that you want


        im_e=plot2.imshow(array_e,origin="lower", interpolation="None",cmap= cm.jet) #changé 0 pour lower car erreur apparemment 
        x = np.linspace(0,width_e,dim_e) ## 1344 pixels
        xs = np.linspace(dim1_e,dim2_e,dim_e) ### wavelength range 524.25 - 529.45
        y = np.linspace(0,height_e,dim_e) ## 1024 pixels
        ys = y*SpatialScale
        xtab = np.array([round(xs[i],1) for i in range(len(xs))])
        ytab = np.array([round(ys[i],1) for i in range(len(ys))])
        plot2.set_xticks(x)
        plot2.set_yticks(y)
        plot2.set_xticklabels(xtab)
        plot2.set_yticklabels(ytab)
        plot2.set_xlabel(r"$\rm wavelength [nm]$",fontsize = 16)
        plot2.set_ylabel(r"$\rm y [\mu m]$",fontsize = 16)
        plot2.set_title("TS eletronic",fontsize = 17)

        fig2.colorbar(im_e)

        canvas2=FigureCanvasTkAgg(fig2,master=fenetre)
        canvas2.draw()
        canvas2.get_tk_widget().place(x=1000,y=0)



    else :
        showinfo('Error')

plot_button1=Button(master=fenetre,command=plot_image,height=2,width=10,text="image")
plot_button1.place(x=1810,y=450)


fenetre.mainloop()

