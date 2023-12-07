# MULTI-TUBULAR SHELL&TUBE CO2 METHANATION REACTOR 
# AUTHOR = Lukas Hiel
# 1) ------- Import packages: ------- 
import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt 

# 2) ------- Define model parameters: ------- 

# i. Feed conditions:  
T_0 = 280+273 # Inlet temperature (K) -> Fill in with correct value
P_0 =  20# Inlet pressure (bar) -> Fill in with correct value
GHSV =  3000# Gas-hourly space velocity (h-1)
GHSV_s = GHSV/3600
Ta_0 = 0 # Inlet temperature of the cooling liquid (K) -> Fill in with correct value
m_c = 0 # Inlet flow of the cooling liquid (kg/s) -> Fill in with correct value

y_CO2_0 = 0.2 # Molar fraction of CO2 (mol%) -> Fill in with correct value
y_H2_0 = 0.8 # Molar fraction of H2 (mol%) -> Fill in with correct value
y_H2O_0 = 0 # Molar fraction of H2O (mol%) -> Fill in with correct value
y_CO_0 = 0.0 # Molar fraction of CO (mol%) -> Fill in with correct value
y_CH4_0 = 0.0 # Molar fraction of CH4 (mol%) -> Fill in with correct value
y_I_0 = 1.0 # Molar fraction of inert (mol%) -> Fill in with correct value

# ii. Components parameters:
Mr_CO2 = 44.01 # Molar mass of CO2 (g/mol)
Mr_H2 = 2.016 # Molar mass of H2 (g/mol)
Mr_H2O = 18.01528 # HMolar mass of H2O (g/mol)
Mr_CO = 28.01 # Molar mass of CO (g/mol)
Mr_CH4 = 16.04 # Molar mass of CH4 (g/mol)
Mr_I = 28.0134 # Molar mass of pure N2 (g/mol)

# The heat capacity equals cp_i = cp_i_ref + (T(°C) - 200°C)*dcp_i
cp_CO2_ref = 43.9678 # Heat capacity of pure CO2 at 200°C (J/mol.K)
cp_H2_ref = 29.2994 # Heat capacity of pure H2 at 200°C (J/mol.K)
cp_H2O_ref = 34.9457 # Heat capacity of pure H2O at 200°C (J/mol.K)
cp_CO_ref = 29.646 # Heat capacity of pure CO at 200°C (J/mol.K)
cp_CH4_ref = 44.9349 # Heat capacity of pure CH4 at 200°C (J/mol.K)
cp_I_ref = 29.4721 # Heat capacity of pure N2 at 200°C (J/mol.K)

dcp_CO2 = 0.0195 # Sensitiviy of heat capacity of pure CO2 (J/mol.K**2)
dcp_H2 = 0.0015 # Sensitiviy of heat capacity of pure H2 (J/mol.K**2)
dcp_H2O = 0.012 # Sensitiviy of heat capacity of pure H2O (J/mol.K**2)
dcp_CO = 0.007 # Sensitiviy of heat capacity of pure CO (J/mol.K**2)
dcp_CH4 = 0.0537 # Sensitiviy of heat capacity of pure CH4 (J/mol.K**2)
dcp_I = 0.0064 # Sensitiviy of heat capacity of pure N2 (J/mol.K**2)

# Specific heat capacity of water cp_c = cp_c_ref + cp_c_1*(T(°C) - 50°C) + cp_c_2*(T(°C) - 50°C)**2
cp_c_ref = 4.22728E+03 # Specific heat capacity of cooling water at 55bar and 50°C (J/kg.K)
dcp_c_1 = 0.7026 # Polynomial constant 1 to calculate the specific heat capacity of water at 55 bar and T(°C)
dcp_c_2 = 0.0667 # Polynomial constant 1 to calculate the specific heat capacity of water at 55 bar and T(°C)

# The viscosity equals mu_i = mu_i_ref + (T(°C) - 200°C)*dmu_i
mu_CO2_ref = 2.26E-05 # Viscosity of pure CO2 at 200°C (Pa.s)
mu_H2_ref = 1.22E-05 # Viscosity of pure H2 at 200°C (Pa.s)
mu_H2O_ref = 1.64E-05 # Viscosity of pure H2O at 200°C (Pa.s)
mu_CO_ref = 2.48E-05 # Viscosity of pure CO at 200°C (Pa.s)
mu_CH4_ref = 1.63E-05 # Viscosity of pure CH4 at 200°C (Pa.s)
mu_I_ref = 2.49E-05 # Viscosity of pure inert at 200°C (Pa.s)

dmu_CO2 = 3E-08 # Sensitivity of viscosity of pure CO2 (Pa.s/K)
dmu_H2 = 2E-08 # Sensitivity of viscosity of pure CO2 (Pa.s/K)
dmu_H2O = 4E-08 # Sensitivity of viscosity of pure CO2 (Pa.s/K)
dmu_CO = 3E-08 # Sensitivity of viscosity of pure CO2 (Pa.s/K)
dmu_CH4 = 2E-08 # Sensitivity of viscosity of pure CO2 (Pa.s/K)
dmu_I = 3E-08 # Sensitivity of viscosity of pure CO2 (Pa.s/K)

# iii.  Reactor & bed parameters: 
L = 2 # Reactor tube length (m) -> Fill in with correct value
d_tube = 0.1  # Reactor tube inner diameter (m) -> Fill in with correct value
d_cat = 0.001 # Diameter of spherical catalyst particles (m)

eps = 0.3 # Effective void fraction of catalyst bed (-)-> Fill in with correct value
rho_cat =  3950 # Density of Ni/Al2O3 catalyst (kg/m³)-> Fill in with correct value
rho_bed =  rho_cat*(1-eps)*4/3*np.pi*d_cat**3 #-> Fill in with correct formula
U = 0 # Global heat transfer coefficient (W/m².K) -> Fill in with correct value
dH_vap_ref = 1.603E+03 # Enthalpy of vaporization of cooling water (kJ/kg) at 55 bar

# iv. Kinetic model parameters: 
T_ref = 298 # Reference temperature (K)
dH_WGS_ref = 41E+03 # Heat of reaction at 298K, 1 atm (J/mol)
dH_COmeth_ref = -206E+03 # Heat of reaction at 298K, 1 atm (J/mol)
dH_CO2meth_ref = -165E+03 # Heat of reaction at 298K, 1 atm (J/mol)

dG_WGS_ref = 29E+03 # Heat of reaction at 298K, 1 atm (J/mol)
dG_COmeth_ref = -142E+03 # Heat of reaction at 298K, 1 atm (J/mol)
dG_CO2meth_ref = -114E+03 # Heat of reaction at 298K, 1 atm (J/mol)

k_1_0 = (2.8/3.6)*1E+09 # Reaction rate coefficient CO methanation (mol/kg_cat.s) -> Fill in the correct value
k_2_0 = (3.4/3.6)*1E+06 # Reaction rate coefficient rWGS (mol/kg_cat.s/bar)

E_1 = 1E+03 # Activation energy coefficient CO methanation (J/mol)-> Fill in the correct value
E_2 = 1E+03 # Activation energy coefficient rWGS (J/mol)-> Fill in the correct value

K_C_0 = 1E-05 # Pre-exponential factor of adsorption constant for C (bar^-0.5)-> Fill in the correct value
K_H_0 = 1E-03 # Pre-exponential factor of adsorption constant for H (bar^-0.5)-> Fill in the correct value
K_CO_0 = 1E-01 # Pre-exponential factor of adsorption constant for CO (bar^-1)-> Fill in the correct value
K_H2_0 = 1E-01 # Pre-exponential factor of adsorption constant for H2 (bar^-1)-> Fill in the correct value
K_H2O_0 = 7E+04 # Pre-exponential factor of adsorption constant for H2O (-)-> Fill in the correct value
K_CH4_0 = 3E-02 # Pre-exponential factor of adsorption constant for CH4 (bar^-1)-> Fill in the correct value

dH_C = -42E+03 # Adsorption enthalpy for C (J/mol)-> Fill in the correct value
dH_H = -16E+03 # Adsorption enthalpy for H (J/mol)-> Fill in the correct value
dH_CO = -700# Adsorption enthalpy for CO (J/mol)-> Fill in the correct value
dH_H2 = -900 # Adsorption enthalpy for H2 (J/mol)-> Fill in the correct value
dH_H2O = 80 # Adsorption enthalpy for H2O (J/mol)-> Fill in the correct value
dH_CH4 = -20 # Adsorption enthalpy for CH4 (J/mol)-> Fill in the correct value

# v.  Calculate inlet flow:
R = 8.314 # Ideal gas constant (J/mol.K)
V = L*np.pi*d_tube**2/4
F_0 = GHSV_s*(V)*((P_0)/(R*T_0)) # initial flowrate into  reactor  (mol/s) -> Fill in with correct formula (use GHSV)

# 3) ------- Define functions: ------- 
def rates(p,T):
    # Create a function for rates(p,T) -> Fill in with correct formulas
    k_1 = k_1_0 * np.exp(-E_1/(R*T))
    k_2 = k_2_0*np.exp(-E_2/(R*T))
    K_C = K_C_0 * np.exp(-dH_C/(R*T))
    K_H = K_H_0 * np.exp(-dH_H/(R*T))
    K_CO = K_CO_0 * np.exp(-dH_CO/(R*T))
    K_H2 = K_H2_0*np.exp((-dH_H2)/(R*T))
    K_H2O = K_H2O_0*np.exp((-dH_H2O)/(R*T))
    K_CH4 = K_CH4_0*np.exp((-dH_CH4)/(R*T))
    K_METH = np.exp(26830/T-30.114)
    K_WGS = np.exp(4400/T-4.036)
    if p[3] == 0:
        r_meth = 0
    else:
        r_meth = (-k_1*K_C*K_H**2*p[3]**0.5*p[1])/((1+K_C*p[3]**0.5+K_H*p[1]**0.5)**3)  #-> Fill in with correct formula
    r_WGS = ((k_2/p[1])*(p[3]*p[2]-(p[1]*p[0])/K_WGS))/(1+K_CO*p[3]+K_H2*p[1]+K_CH4*p[4]+(K_H2O*p[2])/p[1]) #-> Fill in with correct formula
    r=[r_WGS, r_meth]
    return r

def Mr_mix(y):
    # Create a function for Mr_mix(y)-> Fill in with correct formula
    Mr_g = y[0]*Mr_CO2+y[1]*Mr_H2+y[2]*Mr_H2O+y[3]*Mr_CO+y[4]*Mr_CH4+y[5]*Mr_I
    return Mr_g

def cp_mix(y,T):
    # Create a function for cp_mix(y,T)-> Fill in with correct formulas
    
    cp_CO2= cp_CO2_ref + (T - 473)*dcp_CO2
    cp_H2= cp_H2_ref + (T - 473)*dcp_H2
    cp_H2O= cp_H2O_ref + (T - 473) * dcp_H2O
    cp_CO= cp_CO_ref + (T - 473) * dcp_CO
    cp_CH4= cp_CH4_ref + (T - 473) * dcp_CH4
    cp_I= cp_I_ref + (T - 473)*dcp_I
    
    cp_g = y[0]*cp_CO2+y[1]*cp_H2+y[2]*cp_H2O+y[3]*cp_CO+y[4]*cp_CH4+y[5]*cp_I #-> Fill in with correct formula
    return cp_g

def mu_mix(y,T):
    # Create a function for mu_mix(p,T)-> Fill in with correct formulas
    
    mu_CO2= mu_CO2_ref + (T - 473)*dmu_CO2
    mu_H2= mu_H2_ref + (T - 473)*dmu_H2
    mu_H2O= mu_H2O_ref + (T - 473)*dmu_H2O
    mu_CO= mu_CO_ref + (T - 473)*dmu_CO
    mu_CH4= mu_CH4_ref + (T - 473)*dmu_CH4
    mu_I= mu_I_ref + (T - 473)*dmu_I
    
    mu_g = y[0]*mu_CO2+y[1]*mu_H2+y[2]*mu_H2O+y[3]*mu_CO+y[4]*mu_CH4+y[5]*mu_I #-> Fill in with correct formula
    return mu_g

def rho_mix(Mr_g, P, T):
    # Create a function for rho_mix(Mr_g,p,T)-> Fill in with correct formula
    rho_g = (P*Mr_g)/(R*T)
    return rho_g

# 4) ------- Spatial discretization: ------- 
n_z = 100 # Number of segments in reactor 
dz = L/n_z  # Segment size (m) -> Fill in with correct formula
S = d_tube**2/4*np.pi# Cross-sectional area of 1 reactor tube (m²) -> Fill in with correct formula
Peri = d_tube*np.pi # Perimeter of 1 reactor tube (m) -> Fill in with correct formula
dV =  S*dz# Unit volume (m³) -> Fill in with correct formula
dA = S # Unit area (m²) -> Fill in with correct formula
a = dA/dV # Specific area / volume (m²/m³) -> Fill in with correct formula

# Lists & dictionaries:
Y_list = np.zeros((n_z + 1, 9)) # Results of balances
F_list = np.zeros((n_z + 1, 6)) # Molar flowrates in reactor tube (mol/s)
y_list = np.zeros((n_z + 1, 6)) # Molar fraction in reactor tube (mol%)
p_list = np.zeros((n_z + 1, 6)) # Partial pressures in reactor tube (bar)
T_list = np.zeros(n_z+1) # Axial temperature profile of gas in reactor tube (K) 
Ta_list = np.zeros(n_z+1) # Axial temperature profile of cooling liquid around reactor tube (K) 
P_list = np.zeros(n_z+1) # Axial pressure profile in reactor tube (bar) 

# 5) ------- Initialization: ------- 
y_list[0] = [y_CO2_0, y_H2_0, y_H2O_0, y_CO_0, y_CH4_0, y_I_0]
p_list[0] = y_list[0]*P_0
F_list[0] = F_0*y_list[0]
T_list[0] = T_0
Ta_list[0] = Ta_0
P_list[0] = P_0
Y_list[0] = np.concatenate([F_list[0], [T_list[0], Ta_list[0], P_list[0]]])

Mr_0 = Mr_mix(y_list[0])
mu_0 = mu_mix(y_list[0], T_list[0])
cp_0 = cp_mix(y_list[0], T_list[0])
rho_0 = rho_mix(Mr_0, P_list[0], T_list[0]) 
u_0 = GHSV_s*V/S # Gas velocity (m/s) -> Fill in with correct formula

# 6) ------- Define ODE system: 1-dimensional, steady-state, non-isothermal FBR: ------- 
tol1 = 1E-12
tol2 = 1E-12
def ode_system(z,Y):
    # Dependent variables:
    F = Y[:6] # Molar flowrates (mol/s)
    
    F = [1E-10 if i < 0 else i for i in F]
    
    T = Y[6] # Temperature of fluid (K)
    if T < 273.15:
        T = 273.15
    elif T > 900:
        T = 900

    Ta = Y[6+1] # Temperature of cooling fluid (K)
    
    P = Y[6+2] # Pressure of fluid (bar)
        
    F_tot = sum(F)
    y = F/F_tot
    p = y*P
    
    u = F_tot/(P*dA/(R*T)) # Gas velocity (m/s) -> Fill in with the correct formula 
    
    Mr_g = Mr_mix(y)
    mu_g = mu_mix(y, T)
    cp_g = cp_mix(y, T)
    rho_g = rho_mix(Mr_g, P, T)
    
    Cp_c = cp_c_ref + dcp_c_1*(T-323)+ dcp_c_2*(T-323)**2 #calculate heat capacity of cooling liquid
    # Mole balances: ---- DON'T adjust
    dF_dz = np.zeros(6)
    dF_dz_min = -np.array(F)/dz
    M_rates = np.array([[1, 0], [1, 3], [-1, -1], [-1, 1], [0, -1], [0, 0]])*S*rho_bed
    r = np.array(rates(p,T))
    dF_dz = np.dot(M_rates,r)
    
    #tolerance settings --- DON'T adjust
    while any(x + tol1 < y for x,y in zip(dF_dz, dF_dz_min)):
                
        if np.abs(r[0]) < 1E-14:
            r[0] = 0
        if np.abs(r[1]) < 1E-14:
            r[1] = 0

        if dF_dz[0] + tol1 <= dF_dz_min[0]:
            r[0] = dF_dz_min[0]/(S*rho_bed)
            dF_dz = np.dot(M_rates,r)
            
        if dF_dz[1] + tol1 <= dF_dz_min[1]:
            if r[0] >= 0 and r[1] < 0:
                r[1] = ((dF_dz_min[1]/(S*rho_bed)) - r[0])/3
            elif r[0] < 0 and r[1] >= 0:
                r[0] = (dF_dz_min[1]/(S*rho_bed)) - 3*r[1]
            elif r[0] <= 0 and r[1] <= 0:
                r[0] = (dF_dz_min[1]/(S*rho_bed))*r[0]/(r[0] + 3*r[1] + 1E-14)
                r[1] = 3*(dF_dz_min[1]/(S*rho_bed))*r[1]/(r[0] + 3*r[1] + 1E-14)
            dF_dz = np.dot(M_rates,r)
            
        if dF_dz[2] + tol1 <= dF_dz_min[2]:
            if r[0] > 0 and r[1] <= 0:
                r[0] = -((dF_dz_min[2]/(S*rho_bed)) + r[1])
            elif r[0] <= 0 and r[1] > 0:
                r[1] = -((dF_dz_min[2]/(S*rho_bed)) + r[0])
            elif r[0] >= 0 and r[1] >= 0:
                r[0] = -(dF_dz_min[2]/(S*rho_bed))*r[0]/(-r[0] - r[1] + 1E-14)
                r[1] = -(dF_dz_min[2]/(S*rho_bed))*r[1]/(-r[0] - r[1] + 1E-14)
            dF_dz = np.dot(M_rates,r)
        
        if dF_dz[3] + tol1 <= dF_dz_min[3]:
            if r[0] > 0 and r[1] >= 0:
                r[0] = r[1] - (dF_dz_min[3]/(S*rho_bed))
            elif r[0] <= 0 and r[1] < 0:
                r[1] = (dF_dz_min[3]/(S*rho_bed)) + r[0]
            elif r[0] >= 0 and r[1] <= 0:
                r[0] = (dF_dz_min[3]/(S*rho_bed))*r[0]/(r[1] - r[0] + 1E-14)
                r[1] = (dF_dz_min[3]/(S*rho_bed))*r[1]/(r[1] - r[0] + 1E-14)
            dF_dz = np.dot(M_rates,r)
                        

        if dF_dz[4] + tol1 <= dF_dz_min[4]:
            r[1] = -dF_dz_min[4]/(S*rho_bed)
            dF_dz = np.dot(M_rates,r)
            
    # Energy balance:
    dH_COmeth = dH_COmeth_ref + cp_g*(T - T_ref)
    dH_WGS = dH_WGS_ref + cp_g*(T - T_ref)
            
    dT_dz = 0  # Fill in the correct formula
    #if dT_dz > 5E+03:
    #    dT_dz = 5E+03
    #elif dT_dz < -5E+03:
    #    dT_dz = -5E+03
    
    dTa_dz = 0 # Fill in the correct formula
    #if dTa_dz > 5E+03/((m_c)*Cp_c):
    #    dTa_dz = 5E+03/((m_c)*Cp_c)
    #elif dTa_dz < -5E+03/((m_c)*Cp_c):
    #    dTa_dz = -5E+03/((m_c)*Cp_c)
    
    
    # Pressure-drop:
    dP_dz = -(150*mu_g*((1-eps)**2)*u/((eps**3)*(d_cat**2)) + 1.75*(1-eps)*rho_g*(u**2)/((eps**3)*d_cat))*1E-05 # (bar)
    
    #combine all differential equations    
    dY_dz = np.concatenate([dF_dz, [dT_dz, dTa_dz, dP_dz]])
   
    return dY_dz

# 7) ------- Solve ODE system: ------- 
for z in range(n_z):
    #print(z)
    sol = solve_ivp(ode_system, (0, dz), Y_list[z], method="Radau")
    #print(sol)
    Y_list[z+1] = sol.y[:,-1]
    F_list[z+1] = Y_list[z+1][:6]
    T_list[z+1] = Y_list[z+1][6]
    Ta_list[z+1] = Y_list[z+1][6+1]
    P_list[z+1] = Y_list[z+1][6+2]
    y_list[z+1] = F_list[z+1]/sum(F_list[z+1])
    p_list[z+1] = y_list[z+1]*P_list[z+1]
   
#%% 8) ------- Plotting: ------- 
plt.figure(0)
plt.plot(np.linspace(0,L,n_z+1), T_list - 273.15, label="x")
plt.plot(np.linspace(0,L,n_z+1), Ta_list - 273.15, label="y")
plt.xlabel('Reactor length (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature profile:')
plt.legend()

plt.figure(1)
plt.plot(np.linspace(0,L,n_z+1), F_list[:,0], label="u")
plt.plot(np.linspace(0,L,n_z+1), F_list[:,3], label="v")
plt.plot(np.linspace(0,L,n_z+1), F_list[:,4], label="CH4")
plt.plot(np.linspace(0,L,n_z+1), F_list[:,0]+F_list[:,3]+F_list[:,4], label="CARBON")
plt.xlabel('Reactor length (m)')
plt.ylabel('mol/s')
plt.title('Molar flowrates:')
plt.legend()





