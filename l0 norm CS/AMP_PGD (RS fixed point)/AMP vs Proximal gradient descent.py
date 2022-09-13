import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import math 
from multiprocessing.pool import ThreadPool as Pool

pi=np.arccos(-1)
precision=np.double



####################################################
##########   Some functions of interest  ###########
####################################################



# denoiser for AMP (scalar/vector form)    
def prox_hard(B,λ,A):
    if B>np.sqrt(2*λ*A):
        return (B/A)
    if B<-np.sqrt(2*λ*A):
        return (B/A)
    if abs(B)<np.sqrt(2*λ*A):
        return 0
    
def prox_hard_vec(B,λ,A):          
    return (B/A)*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)   

# denoiser for PGD (scalar/vector form)   
def prox_hard_vec_PGD(u,λ):          
    return (u)*np.heaviside(abs(u)-np.sqrt(2*λ),1)   

# function to compute the non-zero entries of a vector
def dens_func(X):
    eps=5*10**(-3)            
    return np.heaviside(abs(X)-eps,1)  





# RS free energy
def ϕRS(α,ρo,σ,m,q,V,λ): 
        m̃=α/(1+V)
        A=α/(1+V)
        q̃=abs(α*((ρo*σ*σ-2*m+q)/(1+V)**2))

        
        
        def GS(ρo,σ,m̃,q̃,A,λ):


            a=q̃+(m̃*σ)**2
            def hard_thres(B,A,λ):
                out=B**2/(2*A)-λ    
                return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
                      
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*hard_thres(np.sqrt(a)*zo     ,A,λ)
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*hard_thres(np.sqrt(q̃)*zo     ,A,λ)
            
            GS_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            GS_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]  
            
            return GS_int

        pot=-m̃*m+(A*q-q̃*V)/2
        
        pot+= GS(ρo,σ,m̃,q̃,A,λ)
        pot+=-(α/2)*(ρo*σ*σ-2*m+q)/(1+V)
        return pot

# RS free energy derivatives
def d_ϕRS(x): 
        m=x[0]
        q=x[1]
        V=x[2]
        A=α/(1+V)
        m̃=A
        q̃=abs(α*((ρo*σ*σ-2*m+q)/(1+V)**2))

        
        
        def GS(ρo,σ,m̃,q̃,A,λ):


            a=q̃+(m̃*σ)**2
            def hard_thres(B,A,λ):
                out=B**2/(2*A)-λ    
                return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
                      
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*hard_thres(np.sqrt(a)*zo     ,A,λ)
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*hard_thres(np.sqrt(q̃)*zo     ,A,λ)
            
            GS_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            GS_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]  
            
            return GS_int

        dq̃=10**(-10)
        dq_pot=-V/2
        dq_pot+=( GS(ρo,σ,m̃,q̃+dq̃,A,λ)-GS(ρo,σ,m̃,q̃,A,λ) )/dq̃
        
        dm̃=10**(-10)
        dm_pot=-m
        dm_pot+=( GS(ρo,σ,m̃+dm̃,q̃,A,λ)-GS(ρo,σ,m̃,q̃,A,λ) )/dm̃
        
        dA=10**(-10)
        dA_pot=q/2
        dA_pot+=( GS(ρo,σ,m̃,q̃,A+dA,λ)-GS(ρo,σ,m̃,q̃,A,λ) )/dA
        
        return [dm_pot,2*dA_pot,2*dq_pot]

# RS state evolution equations 
def ϕRS_SE(x):
       m=x[0]
       q=x[1]
       V=x[2]
       E=abs(ρo*σ*σ-2*m+q)
       A=α/(1+V)
       m̃=α/(1+V)
       q̃=α*E/(1+V)**2
       
       
       def V_t(ρo,σ,E,V,λ):
           
            def d2_BϕRS__(B,A,λ):
                eps=10**(-5)
                out1=(B/A)*(np.tanh(5000000*(abs(B)-np.sqrt(2*λ*A)))+1)/2    
                out2=((B+eps)/A)*(np.tanh(5000000*(abs(B+eps)-np.sqrt(2*λ*A)))+1)/2    
                return  (out2-out1)/eps
                
            def d2_BϕRS_(B,A,λ):
                out=1/A    
                return out*(np.tanh(50000*(abs(B)-np.sqrt(2*λ*A)))+1)/2
            
            
            def d2_BϕRS(B,A,λ):
                out=1/A    
                return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
            
            a=q̃+(m̃*σ)**2
            
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS(np.sqrt(a)*zo,A,λ))
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS(np.sqrt(q̃)*zo,A,λ))
            
            ϕint1_=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS__(np.sqrt(a)*zo,A,λ))
            ϕint2_=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS__(np.sqrt(q̃)*zo,A,λ))
            
            V_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            V_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]  
            
            V_int_=     ρo*integrate.quad(ϕint1_, -np.inf, np.inf)[0]
            V_int_+=(1-ρo)*integrate.quad(ϕint2_, -np.inf, np.inf)[0]  
            

           
            return V_int_
        
       def E_t(ρo,σ,E,V,λ):
            a=q̃+(m̃*σ)**2
            
            def d_BϕRS(B,A,λ):
                out=B/A    
                return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
            
            def part_integral(x):
                a1=(np.sqrt(2*λ*A)-m̃*x)/np.sqrt(2*q̃) 
                a2=(np.sqrt(2*λ*A)+m̃*x)/np.sqrt(2*q̃)
                out1=(1/2)*(  special.erfc(a1)  +special.erfc(a2)  )
                out2=(1/(np.sqrt(2*pi)))*(np.exp(-a1**2)-np.exp(-a2**2))
                return  (m̃*x*x/A)*out1+(np.sqrt(q̃)*x/A)*out2
        
           # ϕint=  lambda    zo,xo: (np.exp(  -(zo**2)/2)/(np.sqrt(2*pi)))*(np.exp(  -(xo**2)/2)/(np.sqrt(2*pi)))*xo*d_BϕRS(np.sqrt(q̃)*zo+xo*m̃,A,λ)
            ϕint2= lambda       xo: (np.exp(  -(xo**2)/2)/(np.sqrt(2*pi)))*part_integral(xo)
            
           # m_int=   ρo*integrate.nquad(ϕint, [[-np.inf, np.inf],[-np.inf, np.inf]])[0]
            m_int=  ρo*integrate.quad(ϕint2,-np.inf, np.inf)[0]


           
           
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d_BϕRS(np.sqrt(a)*zo,A,λ)**2
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d_BϕRS(np.sqrt(q̃)*zo,A,λ)**2
            q_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            q_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0] 
           
           
            E_int=ρo
            E_int+=-2*m_int
            E_int+=+q_int
           
            return [m_int,q_int]
       
       V_int=V_t(ρo,σ,E,V,λ)
       m_int,q_int=E_t(ρo,σ,E,V,λ)
       E_int=ρo*σ*σ
       E_int+=-2*m_int
       E_int+=+q_int
       return [m_int,q_int,V_int]
        
def d_ϕRS_SE(x):
       m=x[0]
       q=x[1]
       V=x[2]
       E=ρo*σ*σ-2*m+q
       A=α/(1+V)
       m̃=α/(1+V)
       q̃=α*E/(1+V)**2
       
       def V_t(ρo,σ,E,V,λ):
           
            def d2_BϕRS__(B,A,λ):
                eps=10**(-10)
                out1=(B/A)*(np.tanh(50000000*(abs(B)-np.sqrt(2*λ*A)))+1)/2    
                out2=((B+eps)/A)*(np.tanh(50000000*(abs(B+eps)-np.sqrt(2*λ*A)))+1)/2    
                return  (out2-out1)/eps
                
            def d2_BϕRS_(B,A,λ):
                eps=10**(-10)
                
                def GS(ρo,σ,m̃,q̃,A,λ):


                    a=q̃+(m̃*σ)**2
                    def hard_thres(B,A,λ):
                        out=B**2/(2*A)-λ    
                        return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
                              
                    ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*hard_thres(np.sqrt(a)*zo     ,A,λ)
                    ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*hard_thres(np.sqrt(q̃)*zo     ,A,λ)
                    
                    GS_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
                    GS_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]  
                    
                    return GS_int
                
                out=2*( GS(ρo,σ,m̃,q̃+eps,A,λ)-GS(ρo,σ,m̃,q̃,A,λ) )/eps
                return out
            
            
            def d2_BϕRS(B,A,λ):
                out=1/A    
                return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
            
            a=q̃+(m̃*σ)**2
            
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS(np.sqrt(a)*zo,A,λ))
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS(np.sqrt(q̃)*zo,A,λ))
            
            ϕint1_=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS__(np.sqrt(a)*zo,A,λ))
            ϕint2_=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_BϕRS__(np.sqrt(q̃)*zo,A,λ))
            
            V_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            V_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]  
            
            V_int_=     ρo*integrate.quad(ϕint1_, -np.inf, np.inf)[0]
            V_int_+=(1-ρo)*integrate.quad(ϕint2_, -np.inf, np.inf)[0]  
            

           
            return V_int_
        
       def E_t(ρo,σ,E,V,λ):
            a=q̃+(m̃*σ)**2
            
            def d_BϕRS(B,A,λ):
                out=B/A    
                return out*np.heaviside(abs(B)-np.sqrt(2*λ*A),1)
            
            def part_integral(x):
                a1=(np.sqrt(2*λ*A)-m̃*x)/np.sqrt(2*q̃) 
                a2=(np.sqrt(2*λ*A)+m̃*x)/np.sqrt(2*q̃)
                out1=(1/2)*(  special.erfc(a1)  +special.erfc(a2)  )
                out2=(1/(np.sqrt(2*pi)))*(np.exp(-a1**2)-np.exp(-a2**2))
                return  (m̃*x*x/A)*out1+(np.sqrt(q̃)*x/A)*out2
        
           # ϕint=  lambda    zo,xo: (np.exp(  -(zo**2)/2)/(np.sqrt(2*pi)))*(np.exp(  -(xo**2)/2)/(np.sqrt(2*pi)))*xo*d_BϕRS(np.sqrt(q̃)*zo+xo*m̃,A,λ)
            ϕint2= lambda       xo: (np.exp(  -(xo**2)/2)/(np.sqrt(2*pi)))*part_integral(xo)
            
           # m_int=   ρo*integrate.nquad(ϕint, [[-np.inf, np.inf],[-np.inf, np.inf]])[0]
            m_int=  ρo*integrate.quad(ϕint2,-np.inf, np.inf)[0]


           
           
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d_BϕRS(np.sqrt(a)*zo,A,λ)**2
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d_BϕRS(np.sqrt(q̃)*zo,A,λ)**2
            q_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            q_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0] 
           
           
            E_int=ρo
            E_int+=-2*m_int
            E_int+=+q_int
           
            return [m_int,q_int]
       
       V_int=V_t(ρo,σ,E,V,λ)
       m_int,q_int=E_t(ρo,σ,E,V,λ)
       E_int=ρo*σ*σ
       E_int+=-2*m_int
       E_int+=+q_int
       return [m-m_int,q-q_int,V-V_int]
        









# System size
N=10000
# Parameters of the Gauss-Bernouilli distribution for the signal
ρo=0.6
σ=1
# Max/min value for λ
λmax=1*10**(-6)
λmin=1*10**(-2)
λ=λmax 
# Number of point taken for λ
Njump=20

# Number of iteration for AMP/PGD for each value of λ
N_dt=2000
Nt=Njump*N_dt

# Values for α
α_list=[0.7,0.9]
color_list=['blue','brown']


# We define variables (loss function, magnetization, self-overlap, density, ||xo-x||_2 and variance V) for the RS fix
Loss_AMP=np.zeros((Njump+1,len(α_list)),dtype=precision)
Loss_PGD=np.zeros((Njump+1,len(α_list)),dtype=precision)
Loss_RS=np.zeros((Njump+1,len(α_list)),dtype=precision)

m_AMP=np.zeros((Njump+1,len(α_list)),dtype=precision)
m_PGD=np.zeros((Njump+1,len(α_list)),dtype=precision)
m_RS=np.zeros((Njump+1,len(α_list)),dtype=precision)

q_AMP=np.zeros((Njump+1,len(α_list)),dtype=precision)
q_PGD=np.zeros((Njump+1,len(α_list)),dtype=precision)
q_RS=np.zeros((Njump+1,len(α_list)),dtype=precision)

ρ_AMP=np.zeros((Njump+1,len(α_list)),dtype=precision)
ρ_PGD=np.zeros((Njump+1,len(α_list)),dtype=precision)
ρ_RS=np.zeros((Njump+1,len(α_list)),dtype=precision)

E_AMP=np.zeros((Njump+1,len(α_list)),dtype=precision)
E_PGD=np.zeros((Njump+1,len(α_list)),dtype=precision)
E_RS=np.zeros((Njump+1,len(α_list)),dtype=precision)

V_AMP=np.zeros((Njump+1,len(α_list)),dtype=precision)
V_RS=np.zeros((Njump+1,len(α_list)),dtype=precision)

λ_list=np.zeros((Njump+1,len(α_list)),dtype=precision)



for index in range(len(α_list)):
    
    
    f = open("AMP_PGD_(α="+str(α_list[index])+",ρo="+str(ρo)+",σ="+str(σ)+").txt", "w")
    f.write("λ	m_AMP	q_AMP	E_loss_AMP	ρ_AMP	m_PGD	q_PGD	E_loss_PGD	ρ_PGD")
    f.write("\n")

          
    α=α_list[index]         
    

    ###############################################    
    ##########   xo, F and y generator   ##########
    ###############################################
    M=int(α*N)
    
    xo=np.zeros(N,dtype=precision)
    F=np.zeros((M,N),dtype=precision)
    F2=np.zeros((M,N),dtype=precision)
    Ft=np.zeros((N,M),dtype=precision)
    y=np.zeros(M,dtype=precision)
    ρo_true=1
    norm=0
    
    # We generate xo
    for k in range(N):
        if k>ρo*N:
            xo[k]=0
            ρo_true+=-1/N
        else:
            xo[k]=np.random.normal(0,σ,1)
            norm+=(xo[k]**2)/N
                
    xo=xo*(np.sqrt(ρo*σ*σ/norm))

    # We generate F and its transpose
    for i in range(M):
        print("matrix creation:",i,M)
        for j in range(N):
            F[i,j]=np.random.normal(0, 1, 1)/np.sqrt(N)
            Ft[j,i]=F[i,j]
        
    # We generate y
    y=np.dot(F,xo) 




    ################################################
    ################# Algorithms ###################
    ################################################
    
    # Initialization of AMP
    X_AMP=np.zeros(N,dtype=precision)
    g=np.zeros(M,dtype=precision)    
    X_AMP[:]=xo[:]
    A_AMP=α
    
    # Initialization of SE for AMP
    m=np.zeros(N_dt,dtype=precision)+0.6
    q=np.zeros(N_dt,dtype=precision)+0.6
    E=np.zeros(N_dt,dtype=precision)
    V=np.zeros(N_dt,dtype=precision)
    
    # Initialization of PGD
    X_PGD=np.zeros(N,dtype=precision)  
    
    # Variable that cut the increase of λ if the RS fixed point dissapear
    div=0
    
    
    
    for djump in range(Njump):
        λ=λ*np.exp((1/(Njump))*(-np.log(λmax)+np.log(λmin)))
      
        print("")
        print("λ,α",λ,α)
        
        ####################################
        ########### RS saddle  #############
        ####################################
        if div==0:
            mo=0.6
            qo=0.6
            Vo=0
            
            # We obtain the RS saddle point via SE
            for k in range(200):
                mo,qo,Vo=ϕRS_SE([mo,qo,Vo])
            
            # We can further improve the RS saddle point by solving exactly the saddle-point equation (can be discarded)
            #mo,qo,Vo=optimize.fsolve(d_ϕRS,[mo,qo,Vo])
            
            # We compute the density ρ for the RS saddle-point
            dλ=λ/10000
            ρ_rs=(  -ϕRS(α,ρo,σ,mo,qo,Vo,λ+dλ)+ϕRS(α,ρo,σ,mo,qo,Vo,λ)  )/dλ
            error_RS=d_ϕRS([mo,qo,Vo])
        
        
            # We compute the density ρ for the RS saddle-point
            if  sum( abs(error_RS[k]) for k in range(3))>1:
                div=1
            if Vo>10:
                div=1
                
            if α==0.7 and djump>10:
                div=1                

     
        
        ####################################
        ###########    AMP     #############
        ####################################

        # We set a damping parameter of for AMP and the SE
        η=0.01
        η_SE=η
        
        
        if div==0:
            
            #We re-initialize the SE for each new value of λ
            m[0]=np.sum(X_AMP*xo)/N
            q[0]=np.sum(X_AMP**2)/N
            E[0]=np.sum(xo**2)/N-2*m[0]+q[0]
            V[0]=V[N_dt-5]
            
            
            for dt in range(1,N_dt): 
                    
                # First we iterate the SE equation
                m_,q_,V_=ϕRS_SE([m[dt-1],q[dt-1],V[dt-1]])
                m[dt]=(1-η_SE)*m[dt-1]+η_SE*m_
                q[dt]=(1-η_SE)*q[dt-1]+η_SE*q_
                E[dt]=ρo*σ*σ-2*m[dt]+q[dt]
                V[dt]=(1-η_SE)*V[dt-1]+η_SE*V_


                # Then we iterate AMP
                ω = np.matmul(F,X_AMP) - g*V[dt]
                g = (y - ω) /(1+V[dt])
                A = α/(1+V[dt])
                B = np.matmul(Ft,g) + X_AMP*A
                
                X_AMP=(1-η)*X_AMP+η*prox_hard_vec(B,λ,A)
                
                # We update the observables of AMP
                ρ_amp=np.sum(dens_func(X_AMP))/N     
                m_amp=np.sum(X_AMP*xo)/N
                q_amp=np.sum(X_AMP**2)/N
                E_amp=np.sum(xo**2)/N-2*m_amp+q_amp
                error=y-np.matmul(F,X_AMP)

                loss_AMP=np.sum((error)**2)/(2*N*λ)+ρ_amp
            
                print("α,λ AMP,dt,djump,η:",α,λ,dt,djump,η)
                print("E_xo, ρ_AMP:",np.sum((xo-X_AMP)**2 )/N, ρ_amp,ρo_true)
                print("Loss, density(SE):",loss_AMP-ρo_true,-ϕRS(α,ρo,σ,m[dt],q[dt],V[dt],λ)/λ-ρo)
                print("Loss, density(RS pot):",-ϕRS(α,ρo,σ,mo,qo,Vo,λ)/λ-ρo)
                print("E,V (SE):",E[dt],V[dt],E[0])
                print("E,V (RS pot)",ρo*σ*σ-2*mo+qo,Vo,sum( abs(error_RS[k]) for k in range(3)))
                print("")
            

            
            
        
        
        ####################################
        ###########    PGD     #############
        ####################################
        
        
        # We set the learning rate
        δ=0.6
        
        if div==0:
            #We initialize PGD at the output of AMP
            X_PGD=X_AMP
        
            for dt in range(1,2000): 
                
                # We iterate PGD
                grad=np.matmul(   Ft,y-np.matmul(F,X_PGD)  )*δ
                
                X_PGD=prox_hard_vec_PGD(grad+X_PGD,λ*δ)
            
            
                # We update the observables of PGD
                ρ_pgd=np.sum(dens_func(X_PGD))/N
                m_pgd=np.sum(X_PGD*xo)/N
                q_pgd=np.sum(X_PGD**2)/N
                E_pgd=np.sum(xo**2)/N-2*m_pgd+q_pgd
                error=y-np.matmul(F,X_PGD)
                    
                loss_PGD=np.sum((error)**2)/(2*N*λ)+ρ_pgd
            
                print("λ PGD",λ,djump,dt)
                print("E_xo:",np.sum((xo-X_PGD)**2)/N)
                print("loss",loss_PGD-ρo_true)
                print("Loss, density(RS pot):",-ϕRS(α,ρo,σ,mo,qo,Vo,λ)/λ-ρo)
                print("")
        
           

        ####################################
        ###########  Results   #############
        ####################################
        
        if div==0: 
            λ_list[djump+1][index]=λ  
            
            Loss_RS[djump+1][index]=-ϕRS(α,ρo,σ,mo,qo,Vo,λ)
            Loss_AMP[djump+1][index]=loss_AMP
            Loss_PGD[djump+1][index]=loss_PGD
            
            m_RS[djump+1][index]=mo
            m_AMP[djump+1][index]=m_amp
            m_PGD[djump+1][index]=m_pgd
            
            q_RS[djump+1][index]=qo
            q_AMP[djump+1][index]=q_amp
            q_PGD[djump+1][index]=q_pgd
            
            ρ_RS[djump+1][index]=ρ_rs
            ρ_AMP[djump+1][index]=ρ_amp
            ρ_PGD[djump+1][index]=ρ_pgd
            
            E_RS[djump+1][index]=ρo*σ*σ-2*mo+qo
            E_AMP[djump+1][index]=E_amp
            E_PGD[djump+1][index]=E_pgd
            
            V_RS[djump+1][index]=Vo
            V_AMP[djump+1][index]=V[N_dt-4]
            
            
            f.write(str(λ))
            f.write("	")
            f.write(str(  m_amp  ))
            f.write("	")
            f.write(str(  q_amp  ))
            f.write("	")
            f.write(str(  loss_AMP            ))
            f.write("	")
            f.write(str(  ρ_amp               ))
            f.write("	")

            f.write(str(  m_pgd  ))
            f.write("	")
            f.write(str(  q_pgd  ))
            f.write("	")
            f.write(str(  loss_PGD            ))
            f.write("	")
            f.write(str(  ρ_pgd               ))
            f.write("	")            
            f.write("\n")
            
            
        else:
            if div==1:
                λ_list[djump+1][index]=λ_list[djump][index]
            
                Loss_RS[djump+1][index]=Loss_RS[djump][index]
                Loss_AMP[djump+1][index]=Loss_AMP[djump][index]
                Loss_PGD[djump+1][index]=Loss_PGD[djump][index]
            
                m_RS[djump+1][index]=m_RS[djump][index]
                m_AMP[djump+1][index]=m_AMP[djump][index]
                m_PGD[djump+1][index]=m_PGD[djump][index]
            
                q_RS[djump+1][index]=q_RS[djump][index]
                q_AMP[djump+1][index]=q_AMP[djump][index]
                q_PGD[djump+1][index]=q_PGD[djump][index]
                
                ρ_RS[djump+1][index]=ρ_RS[djump][index]
                ρ_AMP[djump+1][index]=ρ_AMP[djump][index]
                ρ_PGD[djump+1][index]=ρ_PGD[djump][index]
            
                V_RS[djump+1][index]=V_RS[djump][index]
                V_AMP[djump+1][index]=V_AMP[djump][index]
                
    f.close()      
    





#######################################################################
###########  Plotting different observables of interest   #############
#######################################################################
for k in range(len(α_list)):
    plt.plot(np.log(λ_list[:,k])/np.log(10),Loss_RS[:,k]/λ_list[:,k],c=color_list[k],linestyle='-.', label="Loss (RS)(α="+str(α_list[k])+")")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),Loss_AMP[:,k],s=4 ,marker='v', c=color_list[k],label="Loss AMP")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),Loss_PGD[:,k],s=4 ,marker='o', c=color_list[k],label="Loss AMP+PGD")

plt.legend()
plt.xlabel("log(λ)")
plt.ylabel("Loss/λ")
plt.savefig("AMP vs Prox vs Saddle point (Loss) (α="+str(α_list)+").png")
plt.show()




def fitting_m(x):
    offset=x[0]
    print("offset, power law(m opt):",x)
    out=0
    
    for i in range(10):
            out1=abs(m_AMP[i+1,k]-offset)
            out_=np.log(out1)/np.log(10)-power_law*np.log(λ_list[i+1,k])/np.log(10)
            out+=abs(out_)/10**(-4)
                    
            out2=abs(m_PGD[i+1,k]-offset)
            out_=np.log(out2)/np.log(10)-power_law*np.log(λ_list[i+1,k])/np.log(10)
            out+=abs(out_)/10**(-4)
    print("out:",out)
    print("")
    return out

def fitting_q(x):
    offset=x[0]
    print("offset, power law(q opt):",x)
    out=0
    

    for i in range(Njump):
            out1=abs(q_AMP[i+1,k]-offset)
            out_=np.log(out1)/np.log(10)-power_law*np.log(λ_list[i+1,k])/np.log(10)
            out+=abs(out_)/10**(-4)
                    
            out2=abs(q_PGD[i+1,k]-offset)
            out_=np.log(out2)/np.log(10)-power_law*np.log(λ_list[i+1,k])/np.log(10)
            out+=abs(out_)/10**(-4)
    print("out:",out)
    print("")
    return out

k=1
x_list=np.zeros(10)
y_list=np.zeros(10)
for i in range(10):
    x_list[i]=np.log(λ_list[i+8,k])/np.log(10)
    y_list[i]=np.log(q_RS[i+8,k]-ρo*σ*σ)/np.log(10)
 
power_law, offset=(np.polyfit(x_list, y_list, deg=1))
offset_m=optimize.fmin(fitting_m,[0.6])
offset_q=optimize.fmin(fitting_q,[0.6])



dm_RS=np.zeros((Njump+1,len(α_list)))
dm_AMP=np.zeros((Njump+1,len(α_list)))
dm_PGD=np.zeros((Njump+1,len(α_list)))

dq_RS=np.zeros((Njump+1,len(α_list)))
dq_AMP=np.zeros((Njump+1,len(α_list)))
dq_PGD=np.zeros((Njump+1,len(α_list)))


for k in range(len(α_list)):
    for i in range(Njump):                

        dm_RS[i,k]=abs(m_RS[i,k]-ρo*σ*σ)
        #dm_AMP[i,k]=abs(E_AMP[i,k]+m_AMP[i,k]-q_AMP[i,k])
        #dm_PGD[i,k]=abs(E_PGD[i,k]+m_PGD[i,k]-q_PGD[i,k])
        dm_AMP[i,k]=abs(m_AMP[i,k]-offset_m)
        dm_PGD[i,k]=abs(m_PGD[i,k]-offset_m)
        

        dq_RS[i,k]=abs(q_RS[i,k]-ρo*σ*σ)
        dq_AMP[i,k]=abs(q_AMP[i,k]-offset_q)
        dq_PGD[i,k]=abs(q_PGD[i,k]-offset_q)
        


k=1
plt.plot(np.log(λ_list[:,k])/np.log(10),np.log(abs(dm_RS[:,k]))/np.log(10),c=color_list[k],linestyle='-.', label="m-ρo*σ*σ (RS)(α="+str(α_list[k])+")")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dm_AMP[:,k]))/np.log(10),s=20 ,marker='v', c=color_list[k],label="m-ρo*σ*σ AMP")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dm_PGD[:,k]))/np.log(10),s=4 ,marker='o', c=color_list[k],label="m-ρo*σ*σ AMP+PGD")

    
plt.plot(np.log(λ_list[:,k])/np.log(10),np.log(abs(dq_RS[:,k]))/np.log(10),c=color_list[k],linestyle='--', label="q-ρo*σ*σ (RS)(α="+str(α_list[k])+")")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dq_AMP[:,k]))/np.log(10),s=20 ,marker='+', c=color_list[k],label="q-ρo*σ*σ AMP")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dq_PGD[:,k]))/np.log(10),s=4 ,marker='*', c=color_list[k],label="q-ρo*σ*σ AMP+PGD")

#plt.legend(loc=2,fontsize='xx-small')
plt.xlabel("log(λ)")
plt.ylabel("log_10(|m-ρo*σ*σ|),log_10(|q-ρo*σ*σ|)")
plt.savefig("AMP vs Prox vs Saddle point (m,q) (α="+str(α_list[k])+").png")
plt.show()



k=0
plt.plot(np.log(λ_list[:,k])/np.log(10),np.log(abs(dm_RS[:,k]))/np.log(10),c=color_list[k],linestyle='-.', label="m-ρo*σ*σ (RS)(α="+str(α_list[k])+")")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dm_AMP[:,k]))/np.log(10),s=20 ,marker='v', c=color_list[k],label="m-ρo*σ*σ AMP")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dm_PGD[:,k]))/np.log(10),s=4 ,marker='o', c=color_list[k],label="m-ρo*σ*σ AMP+PGD")

    
plt.plot(np.log(λ_list[:,k])/np.log(10),np.log(abs(dq_RS[:,k]))/np.log(10),c=color_list[k],linestyle='--', label="q-ρo*σ*σ (RS)(α="+str(α_list[k])+")")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dq_AMP[:,k]))/np.log(10),s=20 ,marker='+', c=color_list[k],label="q-ρo*σ*σ AMP")
plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(abs(dq_PGD[:,k]))/np.log(10),s=4 ,marker='*', c=color_list[k],label="q-ρo*σ*σ AMP+PGD")

#plt.legend(loc=2,fontsize='xx-small')
plt.xlabel("log(λ)")
plt.ylabel("log_10(|m-ρo*σ*σ|),log_10(|q-ρo*σ*σ|)")
plt.savefig("AMP vs Prox vs Saddle point (m,q) (α="+str(α_list[k])+").png")
plt.show()



for k in range(len(α_list)):
    plt.plot(np.log(λ_list[:,k])/np.log(10),np.log(E_RS[:,k])/np.log(10),c=color_list[k],linestyle='-.', label="E_xo (RS)(α="+str(α_list[k])+")")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(E_AMP[:,k])/np.log(10),s=4 ,marker='v', c=color_list[k],label="E_xo AMP")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),np.log(E_PGD[:,k])/np.log(10),s=4 ,marker='o', c=color_list[k],label="E_xo AMP+PGD")

plt.legend(loc=2,fontsize='xx-small')
plt.xlabel("log(λ)")
plt.ylabel("E_xo")
plt.savefig("AMP vs Prox vs Saddle point (xo distance) (α="+str(α_list)+").png")
plt.show()



for k in range(len(α_list)):
    plt.plot(np.log(λ_list[:,k])/np.log(10),(ρ_RS[:,k]),c=color_list[k],linestyle='-.', label="ρ (RS)(α="+str(α_list[k])+")")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),(ρ_AMP[:,k]),s=4 ,marker='v', c=color_list[k],label="ρ AMP")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),(ρ_PGD[:,k]),s=4 ,marker='o', c=color_list[k],label="ρ AMP+PGD")

plt.legend(fontsize='xx-small')
plt.xlabel("log(λ)")
plt.ylabel("ρ")
plt.savefig("AMP vs Prox vs Saddle point (density) (α="+str(α_list)+").png")
plt.show()




for k in range(len(α_list)):    
    plt.plot(np.log(λ_list[:,k])/np.log(10),V_RS[:,k],c=color_list[k],linestyle='--', label="V (RS)(α="+str(α_list[k])+")")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),V_AMP[:,k],s=4 ,marker='v', c=color_list[k],label="V AMP")
plt.legend()
plt.xlabel("log(λ)")
plt.ylabel("V")
plt.savefig("AMP vs Prox vs Saddle point (V) (α="+str(α_list)+").png")
plt.show()



for k in range(len(α_list)):    
    plt.plot(np.log(λ_list[:,k])/np.log(10),α_list[k]/(1+V_RS[:,k]),c=color_list[k],linestyle='--', label="A (RS)(α="+str(α_list[k])+")")
    plt.scatter(np.log(λ_list[:,k])/np.log(10),α_list[k]/(1+V_AMP[:,k]),s=4 ,marker='v', c=color_list[k],label="A AMP")
plt.legend()
plt.xlabel("log(λ)")
plt.ylabel("A")
plt.savefig("AMP vs Prox vs Saddle point (A) (α="+str(α_list)+").png")
plt.show()
