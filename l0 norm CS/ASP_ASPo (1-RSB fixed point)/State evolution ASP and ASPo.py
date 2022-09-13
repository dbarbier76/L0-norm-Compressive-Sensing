import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import math 
import random 
from multiprocessing.pool import ThreadPool as Pool
from scipy.interpolate import UnivariateSpline


pi=np.arccos(-1)


#################################################################
##### Useful partial derivatives of ϕ_in at the 1-RSB level #####
#################################################################


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



###################################################
##### Program that run the SE of ASP and ASPo #####
###################################################
def State_evolution_equations():


    def ϕ1RSB_SE_Ansatz(α,ρo,ξ,σ,m,q,V,λ): 
       A=α/(1+V)
       m̃=A
       E=ρo*σ*σ-2*m+q
       q̃=α*E/(1+V)**2
       
       
       def V_t(ρo,ξ,σ,E,V,λ):
       
            def dB2ϕfunc2(B,A,λ):
                 eps=ξ*λ
                 var3=(B-np.sqrt(2*λ*A))/eps
                 var4=(B+np.sqrt(2*λ*A))/eps               
                 ϕ_end=(1/A)*((1-0.5*special.erfc(var3))+0.5*special.erfc(var4))
                 ϕ_end+=(B/A)*(np.exp(-var3**2)/np.sqrt(pi)-np.exp(-var4**2)/np.sqrt(pi))/eps
                 return ϕ_end 

            a=q̃+(m̃*σ)**2
                      
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dB2ϕfunc2(np.sqrt(a)*zo     ,A,λ)
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dB2ϕfunc2(np.sqrt(q̃)*zo     ,A,λ)
            
            V_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            V_int+=(1-ρo)*integrate.quad (ϕint2, -np.inf, np.inf)[0]   
           
            return V_int


       def E_t(ρo,ξ,σ,E,V,λ):
       
            def dB2ϕfunc2(B,A,λ):
                eps=ξ*λ
                var3=(B-np.sqrt(2*λ*A))/eps
                var4=(B+np.sqrt(2*λ*A))/eps               
                ϕ_end=(1/A)*((1-0.5*special.erfc(var3))+0.5*special.erfc(var4))
                ϕ_end+=(B/A)*(np.exp(-var3**2)/np.sqrt(pi)-np.exp(-var4**2)/np.sqrt(pi))/eps
                return ϕ_end 
            
            def dBϕfunc2(B,A,λ):
                eps=ξ*λ
                var3=(B-np.sqrt(2*λ*A))/eps
                var4=(B+np.sqrt(2*λ*A))/eps                 
                ϕ_end=B/A
                ϕ_end=ϕ_end*((1-0.5*special.erfc(var3))+0.5*special.erfc(var4))
                return ϕ_end  


            a=q̃+(m̃*σ)**2
            
            
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dB2ϕfunc2(np.sqrt(a)*zo     ,A,λ)*m̃*(σ**2)
            m_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0] 
           
           
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dBϕfunc2(np.sqrt(a)*zo     ,A,λ)**2
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dBϕfunc2(np.sqrt(q̃)*zo     ,A,λ)**2
            q_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            q_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0] 
           
           
            E_int=ρo
            E_int+=-2*m_int
            E_int+=+q_int
           
            return [m_int,q_int]
        
        
       m_new,q_new=E_t(ρo,ξ,σ,E,V,λ)
       V_new=V_t(ρo,ξ,σ,E,V,λ)
       
       return [m_new,q_new,V_new]


    def ϕ1RSB(α,ρo,σ,m,q,V0,V1,λ,y):         
        eps=10**(-20)
        E= ρo-2*m+q
        A1=α/(1+V1)+eps
        A0=(α/y)*(1/(1+V1)-1/(1+V1+y*V0))+eps
        m̃=α/(1+V1+y*V0)
        q̃=α*E/(1+V1+y*V0)**2 

   
        def GE(ρo,σ,m,q,V0,V1,y):
            GE_int=-0.5*(ρo*σ*σ-2*m+q)/(1+V1+y*V0)+(0.5/y)*np.log((1+V1)/(1+V1+y*V0))
            return GE_int
        
        def GS(ρo,σ,m̃,q̃,A0,A1,y,λ):
        
            def ϕfunc(B,A1,A0,y,λ):
                var1=(np.sqrt(2*λ)*(abs(A1-y*A0))+np.sqrt(A1)*B)/np.sqrt(2*A0*(abs(A1-y*A0)))
                var2=(np.sqrt(2*λ)*(abs(A1-y*A0))-np.sqrt(A1)*B)/np.sqrt(2*A0*(abs(A1-y*A0)))
                var3=(B-np.sqrt(2*λ*A1))/np.sqrt(2*A0)
                var4=(B+np.sqrt(2*λ*A1))/np.sqrt(2*A0)
            
                
                ϕ2= np.exp(-y*λ)*np.sqrt(A1/(abs(A1-y*A0)))*(0.5*special.erfc(var1)+0.5*special.erfc(var2))  
                
                ϕ3= np.exp(-y*B*B/(2*(abs(A1-y*A0))))*(0.5*special.erfc(var3)-0.5*special.erfc(var4))   
                
                ϕ_end=B*B/(2*(abs(A1-y*A0))) +(1/y)*np.log(ϕ2+ϕ3)
                return ϕ_end
        

            a=q̃+(m̃*σ)**2   
            
            ϕint1=  lambda     zo: (np.exp( -(zo**2)/2)/np.sqrt(2*pi))*(ϕfunc(np.sqrt(a)*zo  ,A1,A0,y,λ))
            ϕint2=  lambda     zo: (np.exp( -(zo**2)/2)/np.sqrt(2*pi))*(ϕfunc(np.sqrt(q̃)*zo  ,A1,A0,y,λ))
            
            GS_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            GS_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]  

            
            return GS_int
        
        
        pot= 0.5*(  (V0+q)*A1  -2*m*m̃  -V1*(q̃+A0)  +y*(q*q̃-(q+V0)*(q̃+A0))  ) +GS(ρo,σ,m̃,q̃,A0,A1,y,λ) +α*GE(ρo,σ,m,q,V0,V1,y)   
        return pot


    def ϕ1RSB_SE(α,ρo,σ,m,q,V0,V1,λ,y):
       eps=10**(-20)
       E= ρo-2*m+q
       A1=α/(1+V1)+eps
       A0=(α/y)*(1/(1+V1)-1/(1+V1+y*V0))+eps
       m̃=α/(1+V1+y*V0)
       q̃=α*E/(1+V1+y*V0)**2
       
       def V0_t(ρo,σ,E,V0,V1,λ,y):
        
            a=q̃+(m̃*σ)**2
                      
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(-2*d_A1ϕ_in(np.sqrt(a)*zo,A1,A0,y,λ)-d_Bϕ_in(np.sqrt(a)*zo,A1,A0,y,λ)**2)
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(-2*d_A1ϕ_in(np.sqrt(q̃)*zo,A1,A0,y,λ)-d_Bϕ_in(np.sqrt(q̃)*zo,A1,A0,y,λ)**2)
            
            V_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            V_int+=(1-ρo)*integrate.quad (ϕint2, -np.inf, np.inf)[0]   
           
            return V_int

       def V1_t(ρo,σ,E,V0,V1,λ,y):
        
            a=q̃+(m̃*σ)**2
                      
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_Bϕ_in(np.sqrt(a)*zo,A1,A0,y,λ))
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_Bϕ_in(np.sqrt(q̃)*zo,A1,A0,y,λ))
            
            V_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            V_int+=(1-ρo)*integrate.quad (ϕint2, -np.inf, np.inf)[0]   
           
            return V_int
        
       def E_t(ρo,σ,E,V0,V1,λ,y):
            a=q̃+(m̃*σ)**2
            
            
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d2_Bϕ_in(np.sqrt(a)*zo,A1,A0,y,λ)*m̃*(σ**2)
            m_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0] 
           
           
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d_Bϕ_in(np.sqrt(a)*zo,A1,A0,y,λ)**2
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*d_Bϕ_in(np.sqrt(q̃)*zo,A1,A0,y,λ)**2
            q_int=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            q_int+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0] 
           
           
            E_int=ρo
            E_int+=-2*m_int
            E_int+=+q_int
           
            return [m_int,q_int]
       
       V0_int=V0_t(ρo,σ,E,V0,V1,λ,y)
       V1_int=V1_t(ρo,σ,E,V0,V1,λ,y)-y*V0_int
       m_int,q_int=E_t(ρo,σ,E,V0,V1,λ,y)
       return [m_int,q_int,V0_int,V1_int]


    def dy_ϕ1RSB(y,λ,V0,V1):
        dy=0.00001*y
        Σ=-(ϕ1RSB(α,ρo,σ,m_ASP,q_ASP,V0_ASP,V1_ASP,λ,y+dy)-ϕ1RSB(α,ρo,σ,m_ASP,q_ASP,V0_ASP,V1_ASP,λ,y))/dy
        print("Σ:",Σ)
        if  y>0:
            return Σ
        if  y<0:
            return 100000 
 

    def ϕ1RSB_stab_typeII(α,ρo,σ,m,q,V0,V1,λ,y):

        m̃=α/(1+V1+y*V0)
        q̃=abs(α*(ρo*σ*σ-2*m+q)/(1+V1+y*V0)**2)
        A0=abs((α/y)*( 1/(1+V1) - 1/(1+V1+y*V0) ))
        A1=α/(1+V1) 
        
        def ϕ_I(B,A0,A1,y,λ):  
            
            var1=(np.sqrt(2*λ)*(A1-y*A0)+np.sqrt(A1)*B)/np.sqrt(2*A0*(A1-y*A0))
            var2=(np.sqrt(2*λ)*(A1-y*A0)-np.sqrt(A1)*B)/np.sqrt(2*A0*(A1-y*A0))
            var3=(B-np.sqrt(2*λ*A1))/np.sqrt(2*A0)
            var4=(B+np.sqrt(2*λ*A1))/np.sqrt(2*A0)
                
                    
            ϕ2= np.exp(-y*λ)*np.sqrt(A1/(A1-y*A0))*(0.5*special.erfc(var1)+0.5*special.erfc(var2))  
                    
            ϕ3= np.exp(-y*B*B/(2*(A1-y*A0)))*(0.5*special.erfc(var3)-0.5*special.erfc(var4))   
                    
            ϕ_end=B*B/(2*(A1-y*A0)) +(1/y)*np.log(ϕ2+ϕ3)

            return ϕ_end
        
        def ϕ_II(B,A,λ):  
            return max((B**2)/(2*A)-λ,0)
         
        def d2_Bϕ_II(B,A,y,λ):
            return (1/A)*np.heaviside((B**2)/(2*A)-λ,1)
         
        
        def P_I(h,ρo,σ,q̃,m̃):  
        
            a=q̃+(m̃*σ)**2
            P=(1-ρo)*np.exp(-h**2/(2*q̃))/np.sqrt(2*pi*q̃)+ρo*np.exp(-h**2/(2*a))/np.sqrt(2*pi*a)

            return P
    
        def P_II(h,W,ρo,σ,q̃,m̃,A0,A1,y,λ):  
            h_new=h-np.sqrt(A0)*W
            P=np.exp(y*(ϕ_II(h,A1,λ)-ϕ_I(h_new,A0,A1,y,λ)))*P_I(h_new,ρo,σ,q̃,m̃)
       
            if math.isnan(P)==True or math.isinf(P)==True:
                return 0
            else:
                return P  

        instab2 =  lambda h,W: (np.exp(-(W**2)/2)/np.sqrt(2*pi))*P_II(h,W,ρo,σ,q̃,m̃,A0,A1,y,λ)*(d2_Bϕ_II(h,A1,y,λ))**2     
            
        instab22=(α/((1+V1+y*V0)**2))*integrate.nquad (instab2, [[-6,6],[-np.inf, np.inf]])[0]  
        return 1-instab22
            

    def ϕ1RSB_stab_typeI(α,ρo,σ,m,q,V0,V1,λ,y):

        m̃=α/(1+V1+y*V0)
        q̃=abs(α*(ρo*σ*σ-2*m+q)/(1+V1+y*V0)**2)
        A0=abs((α/y)*( 1/(1+V1) - 1/(1+V1+y*V0) ))
        A1=α/(1+V1)        


        a=q̃+(m̃*σ)**2
        
        ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_Bϕ_in(np.sqrt(a)*zo,A1,A0,y,λ))**2
        ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*(d2_Bϕ_in(np.sqrt(q̃)*zo,A1,A0,y,λ))**2

            
        stab=     ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
        stab+=(1-ρo)*integrate.quad(ϕint2, -np.inf, np.inf)[0]
        
        
        return 1-stab*α/(1+V1+y*V0)**2
            






 
    # Array of α's for which we will compute the free energy
    α_list=[0.7,0.73,0.76,0.83,0.86,0.9]
    
    # Parameters of the Gauss-Bernouilli distribution of the signal
    ρo=0.6
    σ=1
    
    
    for pas in range(len(α_list)):   
        
        α=α_list[pas]
        
        # We set the max/min value  for λ
        λmax=1*10**(-1)
        λmin=1*10**(-6)
        λ=λmax  
        
        # We set the discretization between λmax and λmin
        Nλ=300	
        # We set the dnumber of iteration for a given λ
        Njump=200
        Nt=Nλ*Njump
        
        # Initialization for ASP SE
        
        m_ASP=0.5371105970342697
        q_ASP=0.5867199756487088
        V0_ASP=0.052134871537376656	
        V1_ASP=0.8048949033667998
        
        y=29.5072356591726	
        dyo=0.2
        dy=dyo
        dy_ϕ1RSB_mem=1000

        
        # Initialization for ASPo SE
        m_ASPo=0
        q_ASPo=0
        V_ASPo=0
        ξ=0.7
        


        f=open("ASP_SE("+str(α)+").txt",'w')
        f.write("dt	λ	y	m	q	V0	V1	Loss	stabI	stabII")
        f.write("\n")    

        g=open("ASPo_SE("+str(α)+").txt",'w')
        g.write("dt	λ	m	q	V")
        g.write("\n") 
        
        
        #extra index for changing λ
        tmem=0
        
        for k in range(Nt):
            print("α,λ:",α,λ)
            print("ASP (m,q,V0,V1,y):",m_ASP,q_ASP,V0_ASP,V1_ASP,y)
            
            if k>tmem+Njump-1:
                λ=λ*np.exp((-1/Nλ)*(np.log(λmax)-np.log(λmin)))
                dy=dyo
                tmem=k
            
            #iterating the ASP SE
            m_ASP,q_ASP,V0_ASP,V1_ASP=ϕ1RSB_SE(α,ρo,σ,m_ASP,q_ASP,V0_ASP,V1_ASP,λ,y)
             
            # tuning of the effective temperature to remain at zero complexity
            if k>1:   
                    dy_ϕ1RSB_test=dy_ϕ1RSB(y,λ,V0_ASP,V1_ASP)
                    if dy_ϕ1RSB_test*dy_ϕ1RSB_mem<0:
                        dy=dy/1.1
                        
                    if dy_ϕ1RSB_test>0:
                        dy_ϕ1RSB_mem=dy_ϕ1RSB_test
                        y+=dy*y
                    else:
                        dy_ϕ1RSB_mem=dy_ϕ1RSB_test
                        y+=-dy*y
                        
            #iterating the ASPo SE
            m_ASPo,q_ASPo,V_ASPo=ϕ1RSB_SE_Ansatz(α,ρo,ξ,σ,m_ASPo,q_ASPo,V_ASPo,λ)
            print("ASPo(m,q,V):",m_ASPo,q_ASPo,V_ASPo)            
            print("")
                
            # We compute the loss function and the type I and II 1-RSB instabilities (discarded here)
            loss=-ϕ1RSB(α,ρo,σ,m_ASP,q_ASP,V0_ASP,V1_ASP,λ,y)
            stabI=0#ϕ1RSB_stab_typeI(α,ρo,σ,m_ASP,q_ASP,V0_ASP,V1_ASP,λ,y)
            stabII=0#ϕ1RSB_stab_typeII(α,ρo,σ,m_ASP,q_ASP,V0_ASP,V1_ASP,λ,y)
            
            f.write(str(k))
            f.write("	")
            f.write(str(λ))
            f.write("	")    
            f.write(str(y))
            f.write("	")
            f.write(str(m_ASP))
            f.write("	")
            f.write(str(q_ASP))
            f.write("	")
            f.write(str(V0_ASP))
            f.write("	")
            f.write(str(V1_ASP))
            f.write("	")
            f.write(str(loss))
            f.write("	")
            f.write(str(stabI))
            f.write("	")
            f.write(str(stabII))
            f.write("\n")


            g.write(str(k))
            g.write("	")
            g.write(str(λ))
            g.write("	")
            f.write(str(m_ASPo))
            g.write("	")
            g.write(str(q_ASPo))
            g.write("	")
            g.write(str(V_ASPo))
            g.write("\n")
    
    return 1
    

State_evolution_equations()
    
    
    