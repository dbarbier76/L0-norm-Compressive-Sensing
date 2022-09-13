import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import math 
from multiprocessing.pool import ThreadPool as Pool

pi=np.arccos(-1)





##########   Free energies in/out and Bethe   ###########
#########################################################

# For ASP
def ϕ_in(B,A1,A0,s,λ):
        
    var1=(np.sqrt(2*λ)*((A1-s*A0))+np.sqrt(A1)*B)/np.sqrt(2*A0*((A1-s*A0)))
    var2=(np.sqrt(2*λ)*((A1-s*A0))-np.sqrt(A1)*B)/np.sqrt(2*A0*((A1-s*A0)))
    var3=(B-np.sqrt(2*λ*A1))/np.sqrt(2*A0)
    var4=(B+np.sqrt(2*λ*A1))/np.sqrt(2*A0)     
        
        
    ϕ2= np.exp(-s*λ)*np.sqrt(A1/((A1-s*A0)))*(0.5*special.erfc(var1)+0.5*special.erfc(var2))  
                
    ϕ3= np.exp(-s*B*B/(2*((A1-s*A0))))*(0.5*special.erfc(var3)-0.5*special.erfc(var4))   
                
    ϕ_end=B*B/(2*((A1-s*A0))) +(1/s)*np.log(ϕ2+ϕ3)
    return ϕ_end

def d_Bϕ_in(B,A1,A0,s,λ):
    var1=(np.sqrt(2*λ)*abs(A1-s*A0)+np.sqrt(abs(A1))*B)/np.sqrt(2*abs(A0*(A1-s*A0)))
    var2=(np.sqrt(2*λ)*abs(A1-s*A0)-np.sqrt(abs(A1))*B)/np.sqrt(2*abs(A0*(A1-s*A0)))
    var3=(B-np.sqrt(2*λ*abs(A1)))/np.sqrt(2*abs(A0))
    var4=(B+np.sqrt(2*λ*abs(A1)))/np.sqrt(2*abs(A0))
            
    d_Bϕ1= np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*(s*B/abs(A1-s*A0))*(0.5*special.erfc(var1)+0.5*special.erfc(var2))  \
        +np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*np.sqrt(abs(A1/(A0*(A1-s*A0))))*(-np.exp(-var1**2)+np.exp(-var2**2))/np.sqrt(2*pi)  \
        +np.exp(-s*B*B/(2*abs(A1-s*A0)))*(-np.exp(-var3**2)+np.exp(-var4**2))/np.sqrt(2*pi*abs(A0))  
            
    d_Bϕ2= np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*(0.5*special.erfc(var1)+0.5*special.erfc(var2))  \
        +np.exp(-s*B*B/(2*abs(A1-s*A0)))*(0.5*special.erfc(var3)-0.5*special.erfc(var4))   
            
    d_Bϕ=d_Bϕ1/(s*d_Bϕ2)
    return d_Bϕ

def d2_Bϕ_in(B,A1,A0,s,λ):

    a1=( 
       - (0.3989422804014327*np.exp(-((B - np.sqrt(2*abs(A1)*λ))**2/(2*abs(A0)))))*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)/np.sqrt(abs(A0))
       + (0.3989422804014327*np.exp(-((B + np.sqrt(2*abs(A1)*λ))**2/(2*abs(A0)))))*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)/np.sqrt(abs(A0)) 
       + np.sqrt(abs(A1/(A1 -s*A0)))*((
          0.3989422804014327*np.sqrt(abs(A1))*np.exp(-((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))**2/abs(2*A0*(A1 - s*A0)))))/np.sqrt(abs(A0*(A1 - s*A0))) 
       - (0.3989422804014327*np.sqrt(abs(A1))*np.exp(-(( np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))**2/abs(2*A0*(A1 - s*A0)))))/np.sqrt(abs(A0*(A1 - s*A0))) ) 
       + (s*B*np.sqrt(abs(A1/(A1 -s*A0)))*(0.5*special.erfc((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/( np.sqrt(2*abs(A0*(A1 - s*A0))))) 
                                                                         + 0.5*special.erfc((np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2*abs(A0*(A1 - s*A0)))))))/abs(
       A1 - s*A0)
       )**2

    a2=(
       np.sqrt(abs(A1/(A1 -s*A0)))*(0.5*special.erfc((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2*abs(A0*(A1 - s*A0))))) +  0.5*special.erfc((np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2*abs(A0*(A1 - s*A0)))))) 
       + 0.5*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)*special.erfc((B - np.sqrt(2*abs(A1)*λ))/(np.sqrt(2*abs(A0)))) -  0.5*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)*special.erfc((B + np.sqrt(2*abs(A1)*λ))/(np.sqrt(2*abs(A0)))))**2


    a=a1/a2



    b1=    2*B*s*np.sqrt(abs(A1/(A1 - s*A0)))*(
       0.3989422804014327*np.sqrt(abs(A1))*np.exp(-((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))**2/abs(2*A0*(A1 - s*A0))))/np.sqrt(abs(A0*(A1 - s*A0))) 
       -0.3989422804014327*np.sqrt(abs(A1))*np.exp(-(( np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))**2/abs(2*A0*(A1 - s*A0))))/np.sqrt(abs(A0*(A1 - s*A0))))/abs(A1 - s*A0) 

    b2=    np.sqrt(abs(A1/(A1 - s*A0)))*((0.3989422804014327*abs(A1)*np.exp(-((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))**2/abs(2*A0*(A1 - s*A0))))  *  (-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0)))   /abs(A0*(A1 - s*A0)*np.sqrt(abs(A0*(A1 - s*A0)))) 
                                                                     + (0.3989422804014327*abs(A1)*np.exp(-((np.sqrt(abs(A1))*B +  np.sqrt(2*λ)*abs(A1 - s*A0))**2/abs(2*A0*(A1 - s*A0))))  *  ( np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0)))   /abs(A0*(A1 - s*A0)*np.sqrt(abs(A0*(A1 - s*A0)))))
         
    b3=    0.3989422804014327*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)*np.exp(-((B - np.sqrt(2*abs(A1)*λ))**2/abs(2*A0)))*(B - np.sqrt(2*abs(A1)*λ))/(abs(A0)**(3/2)) -0.3989422804014327*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)*np.exp(-((B + np.sqrt(2*abs(A1)*λ))**2/abs(2*A0)))*(B + np.sqrt(2*abs(A1)*λ))/(abs(A0)**(3/2))  + (s**2)*(B**2)*np.sqrt(abs(A1/(A1 - s*A0)))*(0.5*special.erfc((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2)*np.sqrt(abs(A0*(A1 - s*A0))))) +0.5*special.erfc(( np.sqrt(abs(A1))*B +  np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2)*np.sqrt(abs(A0*(A1 - s*A0)))))  )/abs(A1 - s*A0)**2    
    
    b4= s*np.sqrt(abs(A1/(A1 - s*A0 )))*(0.5*special.erfc((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2*abs(A0*(A1 - s*A0))))) 
                                                                +0.5*special.erfc(( np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0))/(np.sqrt(2*abs(A0*(A1 - s*A0)))))
                                                                )/abs(A1 -s*A0)

    b5=np.sqrt(abs(A1/(A1 - s*A0)))* (0.5*special.erfc((-np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0) )/( np.sqrt(2*abs(A0*(A1 - s*A0))))) + 
                                                                0.5*special.erfc(( np.sqrt(abs(A1))*B + np.sqrt(2*λ)*abs(A1 - s*A0) )/( np.sqrt(2*abs(A0*(A1 - s*A0))))) )  + 0.5*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)*special.erfc((B - np.sqrt(2*abs(A1)*λ))/(np.sqrt(2*abs(A0)))) -  0.5*np.exp(-s*B**2/(2*abs(A1 - s*A0)) + s*λ)*special.erfc((B + np.sqrt(2*abs(A1)*λ))/(np.sqrt(2*abs(A0))))
                 
                 
    b=(b1+b2+b3+b4)/b5
    return  (-a+b)/s

def d_A1ϕ_in(B,A1,A0,s,λ):
    var1=(np.sqrt(2*λ)*abs(A1-s*A0)+np.sqrt(abs(A1))*B)/np.sqrt(2*abs(A0*(A1-s*A0)))
    var2=(np.sqrt(2*λ)*abs(A1-s*A0)-np.sqrt(abs(A1))*B)/np.sqrt(2*abs(A0*(A1-s*A0)))
    var3=(B-np.sqrt(2*λ*abs(A1)))/np.sqrt(2*abs(A0))
    var4=(B+np.sqrt(2*λ*abs(A1)))/np.sqrt(2*abs(A0))
            
    d_A1ϕ1= (-s*B*B/(2*abs(A1-s*A0)**2)+1/(2*A1)-1/(2*abs(A1-s*A0)))*np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*(0.5*special.erfc(var1)+0.5*special.erfc(var2))   \
            +np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*(-np.exp(-var1**2)/np.sqrt(2*pi))*( (np.sqrt(2*λ)+B/(2*np.sqrt(abs(A1))))/np.sqrt(abs(A0*(A1-s*A0))) -  (np.sqrt(2*λ)*abs(A1-s*A0)+np.sqrt(abs(A1))*B)/(2*np.sqrt(abs(A0*(A1-s*A0)))*abs(A1-s*A0))) \
            +np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*(-np.exp(-var2**2)/np.sqrt(2*pi))*( (np.sqrt(2*λ)-B/(2*np.sqrt(abs(A1))))/np.sqrt(abs(A0*(A1-s*A0))) -  (np.sqrt(2*λ)*abs(A1-s*A0)-np.sqrt(abs(A1))*B)/(2*np.sqrt(abs(A0*(A1-s*A0)))*abs(A1-s*A0))) \
            +np.exp(-s*B*B/(2*abs(A1-s*A0)))*(np.exp(-var3**2)+np.exp(-var4**2))*np.sqrt(λ/abs(4*A1*A0*pi))

    d_A1ϕ2= np.exp(-s*λ)*np.sqrt(abs(A1/(A1-s*A0)))*(0.5*special.erfc(var1)+0.5*special.erfc(var2)) \
            +np.exp(-s*B*B/(2*abs(A1-s*A0)))*(0.5*special.erfc(var3)-0.5*special.erfc(var4)) 
            
    d_A1ϕ=d_A1ϕ1/(s*d_A1ϕ2)
    return d_A1ϕ


def ϕ_bethe(s):
    ϕ1=  sum(ϕ_in (B[k,dt-1],A1[1,dt-1],A0[1,dt-1],s,λ)  for k in range(N))/N
    ϕ2= (α/(2*s))*np.log((1+V1[1,dt-1])/(1+V1[1,dt-1]+s*V0[1,dt-1]))
    ϕ3=-s*0.5*(A0[1,dt-1])*(sum(X[k,dt-1]**2 for k in range(N))/N+V0[1,dt-1])
    return ϕ1+ϕ2+ϕ3
    
def ds_ϕ_bethe(s):
    ds=0.0001*s
    Σ=-(ϕ_bethe(s+ds)-ϕ_bethe(s))/ds
    if  s>0:
        return abs(Σ/10**5)**2
    if  s<0:
        return 100000    

def dλ_ϕ_bethe(λ):
    dλ=0.0000001
    ρ=-(ϕ_bethe(λ+dλ)-ϕ_bethe(λ))/dλ
    return ρ


# For ASPo
def d_Bϕ_in_ASPo(B,A,ξ,λ):
    eps=ξ*λ
    var3=(B-np.sqrt(2*λ*A))/eps
    var4=(B+np.sqrt(2*λ*A))/eps                 
    ϕ_end=B/A
    ϕ_end=ϕ_end*((1-0.5*special.erfc(var3))+0.5*special.erfc(var4))
    return ϕ_end 

def d2_Bϕ_in_ASPo(B,A,ξ,λ):
    eps=ξ*λ
    var3=(B-np.sqrt(2*λ*A))/eps
    var4=(B+np.sqrt(2*λ*A))/eps               
    ϕ_end=(1/A)*((1-0.5*special.erfc(var3))+0.5*special.erfc(var4))
    ϕ_end+=(B/A)*(np.exp(-var3**2)/np.sqrt(pi)-np.exp(-var4**2)/np.sqrt(pi))/eps
    return ϕ_end 

#######   Main parameters of the run   ########
###############################################
update_λ=0

# System size
N=2000
# Compression rate
α=0.9
# Parameters of the Gauss-Bernouilli distribution of the signal
ρo=0.6
σ=1


# Max/min value for λ
λmax=1*10**(-1)
λmin=1*10**(-6)
λ=λmax
# Discretization for the switch off of  λ
Nλ=400
# Number of time steps given for each λ step
Nt=150
N_tot=Nλ*Nt



# Damping rate in the iteration
η=1
# If denoising=1 then we have ASP. If denoising=2 then we have ASPo. 
denoising=1
if denoising==2:
    ξ=0.7





for index in range(1):
    
    if denoising==1:
        f = open("ASP(N="+str(N)+",α="+str(α)+",ρo="+str(ρo)+",σ="+str(σ)+").txt", "w")
        f.write("dt	λ	y	m	q	V0	V1	ρ")
        f.write("\n")
        
    if denoising==2:    
        f = open("ASPo(N="+str(N)+",α="+str(α)+",ρo="+str(ρo)+",σ="+str(σ)+").txt", "w")
        f.write("dt	λ	m	q	V")
        f.write("\n")
    
    
    
    
    ###############################################    
    ##########   xo, F and y generator   ##########
    ###############################################
    
    
    M=int(α*N)
    xo=np.zeros(N)
    F=np.zeros((M,N))
    F2=np.zeros((M,N))
    Ft=np.zeros((N,M))
    y=np.zeros(M)
    ρo_true=0
    
    for k in range(N):
        a=np.random.rand(1,1)
        if a>ρo:
            xo[k]=0
            ρo_true+=1/N
        else:
            xo[k]=np.random.normal(0, 1, 1)


    for i in range(M):
        print("matrix creation:",i,M)
        for j in range(N):
            F[i,j]=np.random.normal(0, 1, 1)/np.sqrt(N)
            F2[i,j]=F[i,j]**2
            Ft[j,i]=F[i,j]   
    y=np.dot(F,xo) 


    ################## Algorithm ###################
    ################################################

  
    X=np.zeros((N,N_tot))
    X_new=np.zeros((N,N_tot))
    ρ=np.zeros(N_tot)
    Δ0=np.zeros((N,N_tot))
    Δ1=np.zeros((N,N_tot))
    V0=np.zeros((M,N_tot))
    V0_new=np.zeros((M,N_tot))
    V1=np.zeros((M,N_tot))
    V1_new=np.zeros((M,N_tot))

    B=np.zeros((N,N_tot))
    B_new=np.zeros((N,N_tot))
    Γ0=np.zeros((M,N_tot))
    Γ1=np.zeros((M,N_tot))
    A0=np.zeros((N,N_tot))
    A0_new=np.zeros((N,N_tot))
    A1=np.zeros((N,N_tot))
    A1_new=np.zeros((N,N_tot))
    w=np.zeros((M,N_tot))
    w_new=np.zeros((M,N_tot))
    g=np.zeros((M,N_tot))
    g_new=np.zeros((M,N_tot))


    identity_N=np.ones(N)
    identity_M=np.ones(M)

    
    λ_list=np.zeros(N_tot)
    logλ_list=np.zeros(N_tot)
    s_list=np.zeros(N_tot)
    logs_list=np.zeros(N_tot)



    # Initialization for ASP/ASPo
    g[:,0]=0
    X[:,0]=xo[:]
    V0[:,0]=0.001
    V1[:,0]=0.0000001
    s=30
    Ds=0.001
    
    # Extra parameter to switch off λ and tuning s
    t_mem=0
    dΣ_mem=0
    
    for dt in range(1,N_tot):        
        
        ############# update λ and s ############
        #########################################       
        
        # We reduce λ after Nt steps with a fixed λ
        if  dt>t_mem+Nt-1:
            λ=λ*np.exp((1/(Nλ))*(-np.log(λmax)+np.log(λmin)))
            t_mem=dt
            Ds=0.001
            
            
        print("α,ρo",α,ρo)       
        print("time,λ,s",dt,λ,s)           

        # We tune s to explore the zero-complexity states
        if  dt>10+t_mem:
            
            ds=s*(10**(-6))
            dΣ=(ds_ϕ_bethe(s+ds)-ds_ϕ_bethe(s))/ds
            print("dΣ,Ds",dΣ,Ds)
                
            if dΣ>0:
                s+=-Ds*s
            if dΣ<0:
                s+=+Ds*s
            if dΣ*dΣ_mem<0: # if we are around the dΣ=0 we decrease Ds
                Ds=Ds/2
                
            dΣ_mem=dΣ
    
            
        λ_list[dt]=λ
        logλ_list[dt]=np.log(λ)/np.log(10)
        s_list[dt]=s
        logs_list[dt]=np.log(s)/np.log(10)
        
        

        ##### update interactions (B,A0,A1) #####
        #########################################
        if denoising==1:    
            w_new[:,dt]=np.dot(F,X[:,dt-1]) - g[:,dt-1]*(s*V0[:,dt-1]+V1[:,dt-1])
            w[:,dt]=(1-η)*w[:,dt-1]+η*w_new[:,dt]        
            
            g_new[:,dt]=-(w[:,dt]-y[:])/(1+V1[:,dt-1]+s*V0[:,dt-1]) 
            g[:,dt]=(1-η)*g[:,dt-1]+η*g_new[:,dt]   

            A0_new[:,dt]=(α/s)*(1/(1+V1[1,dt-1])-1/(1+V1[1,dt-1]+s*V0[1,dt-1]))*identity_N[:]
            A1_new[:,dt]=α/(1+V1[1,dt-1])*identity_N[:]
        
            A0[:,dt]=(1-η)*A0[:,dt-1]+η*A0_new[:,dt]
            A1[:,dt]=(1-η)*A1[:,dt-1]+η*A1_new[:,dt]
    

            B_new[:,dt]=X[:,dt-1]*(A1[:,dt]-s*A0[:,dt]) + np.dot(Ft,g[:,dt])
            B[:,dt]=(1-η)*B[:,dt-1]+η*B_new[:,dt] 

        if denoising==2:    
            w_new[:,dt]=np.dot(F,X[:,dt-1]) - g[:,dt-1]*(V1[:,dt-1])
            w[:,dt]=(1-η)*w[:,dt-1]+η*w_new[:,dt]        
            
            g_new[:,dt]=-(w[:,dt]-y[:])/(1+V1[:,dt-1]) 
            g[:,dt]=(1-η)*g[:,dt-1]+η*g_new[:,dt]   

            A0_new[:,dt]=A0_new[:,dt-1]
            A1_new[:,dt]=α/(1+V1[1,dt-1])*identity_N[:]
        
            A0[:,dt]=(1-η)*A0[:,dt-1]+η*A0_new[:,dt]
            A1[:,dt]=(1-η)*A1[:,dt-1]+η*A1_new[:,dt]
    

            B_new[:,dt]=X[:,dt-1]*(A1[:,dt]) + np.dot(Ft,g[:,dt])
            B[:,dt]=(1-η)*B[:,dt-1]+η*B_new[:,dt]   
            
            
        ###### update estimators (x,V0,V1) ######
        #########################################
        if denoising==1:    
            X_new[:,dt]=d_Bϕ_in(B[:,dt],A1[:,dt],A0[:,dt],s,λ)
        
            Δ0[:,dt]= -2*d_A1ϕ_in(B[:,dt],A1[:,dt],A0[:,dt],s,λ) -  X_new[:,dt]**2
        
            Δ1[:,dt]= d2_Bϕ_in(B[:,dt],A1[:,dt],A0[:,dt],s,λ)
            Δ1[:,dt]+=-s*Δ0[:,dt]

   
            V0_new_int=sum(Δ0[k,dt] for k in range(N))/N
            V0_new[:,dt]=V0_new_int*identity_M[:]
    
            V1_new_int=sum(Δ1[k,dt] for k in range(N))/N
            V1_new[:,dt]=V1_new_int*identity_M[:]
    
            X[:,dt]=(1-η)*X[:,dt-1]+η*X_new[:,dt] 
            V0[:,dt]=(1-η)*V0[:,dt-1]+η*V0_new[:,dt] 
            V1[:,dt]=(1-η)*V1[:,dt-1]+η*V1_new[:,dt] 
            
            ρ[dt]=  dλ_ϕ_bethe(λ)

        if denoising==2: 
            X_new[:,dt]=d_Bϕ_in_ASPo(B[:,dt],A1[:,dt],ξ,λ)
        
            Δ1[:,dt]= d2_Bϕ_in_ASPo(B[:,dt],A1[:,dt],ξ,λ)

            V0_new[:,dt]=V0_new[:,dt-1]
    
            V1_new_int=sum(Δ1[k,dt] for k in range(N))/N
            V1_new[:,dt]=V1_new_int*identity_M[:]
    
            X[:,dt]=(1-η)*X[:,dt-1]+η*X_new[:,dt] 
            V0[:,dt]=(1-η)*V0[:,dt-1]+η*V0_new[:,dt] 
            V1[:,dt]=(1-η)*V1[:,dt-1]+η*V1_new[:,dt] 


        print("m,q",sum(X[k,dt]*xo[k]/N for k in range(N)),sum(X[k,dt]**2/N for k in range(N)))
        print("V0,V1",V0[1,dt],V1[1,dt])
        print("") 
        print("") 
        
        
        ########### update observables ##########
        #########################################


        if denoising==1: 
            f.write(str(dt))
            f.write("	")
            f.write(str(λ))
            f.write("	")    
            f.write(str(s))
            f.write("	")
            f.write(str(sum(X[k,dt]*xo[k]/N for k in range(N))))
            f.write("	")
            f.write(str(sum(X[k,dt]**2/N for k in range(N))))
            f.write("	")
            f.write(str(V0[1,dt]))
            f.write("	")
            f.write(str(V1[1,dt]))
            f.write("	")
            f.write(str(ρ[dt]))
            f.write("\n")
    
    
        if denoising==2: 
            f.write(str(dt))
            f.write("	")
            f.write(str(λ))
            f.write("	")
            f.write(str(sum(X[k,dt]*xo[k]/N for k in range(N))))
            f.write("	")
            f.write(str(sum(X[k,dt]**2/N for k in range(N))))
            f.write("	")
            f.write(str(V1[1,dt]))
            f.write("\n")
  
        
        
    f.close()            
            

     

    
