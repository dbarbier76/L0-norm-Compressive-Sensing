import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import math 
import random 
from multiprocessing.pool import ThreadPool as Pool


pi=np.arccos(-1)

#################################################################
##### Routine outputing a file with α_c as a function of ρo #####
#################################################################
def problem_optimal_y(input):
        

    def ASPo_SE(α,ρo,ξ,σ,E,V,λ): 
        A=α/(1+V)
        m̃=A
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
                      
            ϕint1=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dB2ϕfunc2(np.sqrt(a)*zo,A,λ)
            ϕint2=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dB2ϕfunc2(np.sqrt(q̃)*zo,A,λ)
            
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
            
            ϕint3=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dBϕfunc2(np.sqrt(a)*zo     ,A,λ)**2
            ϕint4=  lambda  zo: (np.exp(  -(zo**2)/2)/np.sqrt(2*pi))*dBϕfunc2(np.sqrt(q̃)*zo     ,A,λ)**2
            
            E_int=ρo*σ*σ
            E_int+=-2*ρo*integrate.quad(ϕint1, -np.inf, np.inf)[0]
            E_int+=ρo*integrate.quad(ϕint3, -np.inf, np.inf)[0]  
            E_int+=(1-ρo)*integrate.quad(ϕint4, -np.inf, np.inf)[0]
            
            return E_int
        
        eq1=V_t(ρo,ξ,σ,E,V,λ)
        eq2=E_t(ρo,ξ,σ,E,V,λ)
        return [eq1,eq2]

    ρo_max=input[1]
    ρo_min=input[2]
    
    # Parameters fixing the interval for which we look for the transition: α_c \in [α,α-dα] 
    α=input[3]
    dα=0.05
    
    # Parameter fixing the maximal value of ξ
    ξmax=input[4]
    
    # Parameter the number of points we will have between ρo_max and ρo_min
    N_run=int(input[5])
    
     
    
    # Parameter fixing the number of steps for decreasing λ
    N_λ=10000
    # Parameter fixing the maximal/minimal value of λ
    λmax=0.2
    λmin=0.00001
    


    f = open("1RSB_transition(σ="+str(σ)+", rhomax="+str(ρo_max)+", rhomin="+str(ρo_min)+").txt", "w")
    f.write("ρ	α	ξ (Nλ="+str(N_λ)+", λmax="+str(λmax)+", λmin="+str(λmin)+")")
    f.write("\n")



    for k in range(N_run):
            ρo=ρo_max-(ρo_max-ρo_min)*(k/N_run) 

            def ξ_check(α,dα,ρo,ξ):
                #######################################################################################################
                ####### This subroutine provides the optimal value of ξ for which we can expect signal recovery #######
                #######################################################################################################
                ξ_conv=ξ/4
                ξ_div=ξ
                for j in range(11):
                    ξ_test=(ξ_conv+ξ_div)/2
                    
                    λo=λmax
                    breaking=0
                
                    V_list=np.zeros(N_λ+1)
                    E_list=np.zeros(N_λ+1)
                    V_list[0]=0
                    E_list[0]=ρo
                    
                    N_check=40
                    t_check=200
                    slope_check=np.zeros(N_check)
                    conv_check=np.zeros(N_check)
                
                    for l in range(N_λ):
                        λo=λo*np.exp((-1/N_λ)*(np.log(λmax)-np.log(λmin))) 
                        V_list[l+1],E_list[l+1]=ASPo_SE(α,ρo,ξ_test,σ,E_list[l],V_list[l],λo)

                        if l>t_check:
                            slope_check=np.roll(slope_check, 1)
                            slope_check[0]=(E_list[l+1]-E_list[l+5-t_check])/abs(E_list[l+1]-E_list[l+5-t_check]) 
                            
                            conv_check=np.roll(conv_check, 1)
                            conv_check[0]=(E_list[l+1]-E_list[l+5-t_check])/abs(E_list[l+1])  
                        
                        print("α,dα,ρo,index (checking ξ):",α,dα,ρo,l,j)
                        print("λo,V,E,ξ:")
                        print(λo,V_list[l+1],E_list[l+1],ξ_test)
                        print("")
                    
                    
                        if math.isnan(V_list[l+1])==True or math.isnan(E_list[l+1])==True or abs(V_list[l+1])>10**6:
                            breaking=1
                            break
                        if l>t_check and sum(slope_check[k] for k in range(len(slope_check)))>N_check-0.5 and abs(sum(conv_check[k] for k in range(len(conv_check))))<10:
                            breaking=1
                            break
                        if l>t_check and abs(sum(conv_check[k] for k in range(len(conv_check))))<10**(-4):
                            break   
                    if breaking==1:
                        ξ_div=ξ_test
                    else:
                        ξ_conv=ξ_test
                    
                    
                return ξ_conv

                        
            ####################################################
            ####### Finding the critical α at a given ρo #######
            ####################################################
            
            # We run the SE equations for ASPo for a given α_test. If we obtain signal recovery α_test is above the critical α, if SE explode α_test is below. 
            # We perform several run to evaluate for which value of α_test SE goes from signal recovery to divergence
            
            α_conv=α
            α_div=max(α-dα,0.001)  
            
            for j in range(8):
                print("ρo,α_test,dα:",(α_conv+α_div)/2,(α_conv-α_div))
                α_test=(α_conv+α_div)/2
                
                λo=λmax
                breaking=0
                
                # Before run checking the SE of ASPo we determine 
                ξ=ξ_check(α_test,(α_conv-α_div),ρo,ξmax)
                
                
                # Initialization of the ASPo SE
                V_list=np.zeros(N_λ+1)
                E_list=np.zeros(N_λ+1)
                V_list[0]=0
                E_list[0]=ρo
                
                
                # Parameters to check very early the divergence/converge of the SE
                N_check=100
                t_check=200
                slope_check=np.zeros(N_check)
                slope_check2=np.zeros(N_check)
                
                for l in range(N_λ):
                    λo=λo*np.exp((-1/N_λ)*(np.log(λmax)-np.log(λmin))) 

                    V_list[l+1],E_list[l+1]=ASPo_SE(α_test,ρo,ξ,σ,E_list[l],V_list[l],λo)        

                    print("ρo,α_test,dα  (checking α):",ρo,(α_conv+α_div)/2,(α_conv-α_div))
                    print("λ,V,E,ξ:",λo,V_list[l+1],E_list[l+1],ξ)
                    print("")
                    
                    
                    
                    # We compute the parameters to check the divergence/convergence of the SE
                    if l>t_check:
                        slope_check=np.roll(slope_check, 1)
                        slope_check[0]=(V_list[l+1]-V_list[l+5-t_check])/abs(V_list[l+1]-V_list[l+5-t_check])                   
 
                        slope_check2=np.roll(slope_check2, 1)
                        slope_check2[0]=(E_list[l+1]-E_list[l+5-t_check])/abs(E_list[l+1]-E_list[l+5-t_check]) 
                    
                    
                    
                    
                    #### Checking the convergence ###
                    if l>t_check and sum(slope_check[k] for k in range(len(slope_check)))>N_check-0.5:
                        breaking=1
                        
                    if breaking==1 and sum(slope_check[k] for k in range(len(slope_check)))<-N_check+0.5:
                        breaking=0
                        break

                    #### Checking the divergence ###                 
                    if l>t_check and sum(slope_check[k] for k in range(len(slope_check)))>N_check-0.5 and sum(slope_check2[k] for k in range(len(slope_check2)))>N_check-0.5:
                        breaking=1
                        break                    
                    if V_list[l+1]> 10**9:  
                        breaking=1
                        break   
                    
                if breaking==1:
                    α_div=α_test
                else:
                    α_conv=α_test
                    
                
                
            α=α_conv  
    
            # Writing in the file the critical value for α
            f.write(str(ρo))
            f.write("	")    
            f.write(str(α))
            f.write("	")    
            f.write(str(ξ))
            f.write("\n")
                    
            
    f.close()   
        
    return 1
    


# Standard variation in the Gauss-Bernouilli of the signal xo
σ=1

# We set the intervals in which we look for α_c, ρo \in [ρo_finish,ρo_begin]
ρo_begin= [0.9 ,0.8  ,0.7 ,0.6 ,0.5 ,0.4 ,0.3 ,0.2 ,0.1 ]
ρo_finish=[0.8 ,0.7  ,0.6 ,0.5 ,0.4 ,0.3 ,0.2 ,0.1 ,0.0 ]
# We set (as a guess) an upper bound for α_c in each of the intervals [ρo_finish[k],ρo_begin[k]] defined above
α_begin=  [0.99,0.945,0.88,0.81,0.73,0.63,0.52,0.39,0.22] 
# We set (as a guess) an upper bound for ξ in each of the intervals [ρo_finish[k],ρo_begin[k]] defined above
ξ_begin=  [28  ,12   ,10   ,5   ,3.9 ,2.6,1.9  ,1.2 ,0.9 ]    
# We set the number of points for which we determine α_c in a given interval [ρo_finish[k],ρo_begin[k]]
N_run=5

for run in range(9):
    problem_optimal_y([1,ρo_begin[run],ρo_finish[run],α_begin[run],ξ_begin[run],N_run])
    
    