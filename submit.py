import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import itertools

# Point set generation for gaussians
#n is the number of points, d the dimension, k the number of clusters. Spread influences the size of the box containing the cluster centers, covi the covariance matrix of the gaussians.
def point_gene(n,d,k,rng,spread,covi):
    Centers=[[rng.random()*spread for i in range(d)] for j in range(k)]
    Points=[]
    for i in range(n):
        r=rng.integers(0,k)
        mean=Centers[r]
        cov=[[0 for l in range(d)] for j in range(d)]
        for j in range(d):
            cov[j][j]=covi
        Points.append(rng.multivariate_normal(mean,cov))
    return(Points)


def dist(X,Y):
    d=len(X)
    tot=0
    for i in range(d):
        tot+=(X[i]-Y[i])**2
    return(tot)

#kmeans++ initialization
def kplusplus(P,n,d,k,seed):
    rng=np.random.default_rng(191*seed-18)
    Cluster=[[0 for i in range(d)] for j in range(k)]
    init=rng.integers(0,n)
    Cluster[0]=P[init]
    nb=1
    Distances=[dist(Cluster[0],P[i]) for i in range(n)]
    while nb<k:
        Proba=[0 for i in range(n)]
        Proba[1]=Distances[0]
        for j in range(1,n):
            Proba[j]=Proba[j-1]+Distances[j]
        for j in range(n):
            Proba[j]=Proba[j]/Proba[-1]
        s=rng.random()
        temp=0
        rk=0
        while temp<s:
            temp+=Proba[rk]
            rk+=1
        rk-=1
        Cluster[nb]=P[rk]
        for j in range(n):
            Distances[j]=min(Distances[j],dist(Cluster[nb],P[j]))
        nb+=1
    return(Cluster)

# Functions to modify the threshold for Hartigan/Smartigan swap acceptance
def func1(x,i,n):
    return(max((1.5-0.5*i/n),1)*x)

def func2(x,i,n):
    return(x)


# Hartigan algorithm, currently with kmeans++ initialization. 
#Comment out the  first commented line for random initialization
def hartigan(n,k,d,seed,steps,repeats,spread,cov):
    rng=np.random.default_rng(seed)
    Vals1=[]
    Vals2=[]
    Points=point_gene(n, d, k,rng,spread,cov)
    for a in range(repeats):
        Assign=[0 for i in range(n)] 
        #Cluster=[[rng.random()*spread-1 for i in range(d)] for j in range(k)]
        #kmeans++
        Cluster=kplusplus(Points,n,d,k,seed)
        InitClus=Cluster
        for i in range(n):
            Distances=[dist(Points[i],Cluster[r]) for r in range(k)]
            s=min(Distances)
            Assign[i]=Distances.index(s)
        Count=[0 for i in range(k)]
        Cluster=[[0 for i in range(d)] for j in range(k)]
        for i in range(n):
            c=Assign[i]
            Count[c]+=1
            for j in range(d):
                Cluster[c][j]+=Points[i][j]
        
        for i in range(k):
            for j in range(d):
                if Count[i]>0:
                    Cluster[i][j]=Cluster[i][j]/Count[i]
                else:
                    Cluster[i][j]=InitClus[i][j]
        i=0
        C=[i for i in range(n)]
        while i<steps:
            i+=1
            np.random.shuffle(C)
            for r in C:
            #r=rng.integers(0,n)
                curr_clus=Assign[r]
                if Count[curr_clus]>1:
                    curr_clus_dist=Count[curr_clus]*dist(Points[r],Cluster[curr_clus])/(Count[curr_clus]-1)
                else:
                    curr_clus_dist=0
                other_clus_dist=[Count[i]*dist(Points[r],Cluster[i])/(Count[i]+1) for i in range(k)]
                other_clus_dist[curr_clus]=max(other_clus_dist)
                rival=min(other_clus_dist)
                rival_clus=other_clus_dist.index(rival)
                u=func2(curr_clus_dist,i,steps)
                if rival <u:
                    Assign[r]=rival_clus
                    Count[curr_clus]-=1
                    Count[rival_clus]+=1
                    Cluster[curr_clus]=[0 for i in range(d)]
                    Cluster[rival_clus]=[0 for i in range(d)]
                    for t in range(n):
                        if Assign[t]==curr_clus:
                            for j in range(d):
                                Cluster[curr_clus][j]+=Points[t][j]/Count[curr_clus]
                        elif Assign[t]==rival_clus:
                            for j in range(d):
                                Cluster[rival_clus][j]+=Points[t][j]/Count[rival_clus]
        tot_cost=0
        for i in range(n):
            tot_cost+=dist(Points[i],Cluster[Assign[i]])
        Vals1.append(tot_cost)
    #print(bet_cost)
    return(Vals1)

#Smartigan implementation. It could also be obtained from the previous Hartigan by simply modifying the swap acceptance check line (func2 replaced by func1)
#Also set to kmeans++ initialization. Uncomment the first comment for random initialization
def smartigan(n,k,d,seed,steps,repeats,spread,cov):
    rng=np.random.default_rng(seed)
    Vals1=[]
    Vals2=[]
    Points=point_gene(n, d, k,rng,spread,cov)
    for a in range(repeats):
        Assign=[0 for i in range(n)] 
        #Cluster=[[rng.random()*spread-1 for i in range(d)] for j in range(k)]
        #kmeans++
        Cluster=kplusplus(Points,n,d,k,seed)
        InitClus=Cluster
        for i in range(n):
            Distances=[dist(Points[i],Cluster[r]) for r in range(k)]
            s=min(Distances)
            Assign[i]=Distances.index(s)
        Count=[0 for i in range(k)]
        Cluster=[[0 for i in range(d)] for j in range(k)]
        for i in range(n):
            c=Assign[i]
            Count[c]+=1
            for j in range(d):
                Cluster[c][j]+=Points[i][j]
        
        for i in range(k):
            for j in range(d):
                if Count[i]>0:
                    Cluster[i][j]=Cluster[i][j]/Count[i]
                else:
                    Cluster[i][j]=InitClus[i][j]
        i=0
        C=[i for i in range(n)]
        while i<10:
            i+=1
            np.random.shuffle(C)
            for r in C:
            #r=rng.integers(0,n)
                curr_clus=Assign[r]
                if Count[curr_clus]>1:
                    curr_clus_dist=Count[curr_clus]*dist(Points[r],Cluster[curr_clus])/(Count[curr_clus]-1)
                else:
                    curr_clus_dist=0
                other_clus_dist=[Count[i]*dist(Points[r],Cluster[i])/(Count[i]+1) for i in range(k)]
                other_clus_dist[curr_clus]=max(other_clus_dist)
                rival=min(other_clus_dist)
                rival_clus=other_clus_dist.index(rival)
                u=func1(curr_clus_dist,i,steps)
                if rival <u:
                    Assign[r]=rival_clus
                    Count[curr_clus]-=1
                    Count[rival_clus]+=1
                    Cluster[curr_clus]=[0 for i in range(d)]
                    Cluster[rival_clus]=[0 for i in range(d)]
                    for t in range(n):
                        if Assign[t]==curr_clus:
                            for j in range(d):
                                Cluster[curr_clus][j]+=Points[t][j]/Count[curr_clus]
                        elif Assign[t]==rival_clus:
                            for j in range(d):
                                Cluster[rival_clus][j]+=Points[t][j]/Count[rival_clus]
        tot_cost=0
        for i in range(n):
            tot_cost+=dist(Points[i],Cluster[Assign[i]])
        Assign2=[0 for i in range(n)]
        for i in range(n):
            Dists=[dist(Points[i],Cluster[r]) for r in range(k)]
            Assign2[i]=Dists.index(min(Dists))
        bet_cost=0
        for i in range(n):
            bet_cost+=dist(Points[i],Cluster[Assign2[i]])
        Vals1.append(tot_cost)
        Vals2.append(bet_cost)
    print(bet_cost)
    return(Vals1,Vals2)



# Function for the Gaussian experiments.
#Seeds are included in the extra output files.
#Each block of 4 lines in the outputs corresponds to one (n,k,seed) setting below. The seed is the first element of the line (after H or S to indicate the algorithm), followed by kmeans distance output for varying initializations.
def clean_easy_2d(d,rep):
    f=open("output"+str(d)+"d_"+str(rep)+"_rand_multi.txt","w")
    rng=np.random.default_rng()
    for k in [2,10,25]:
        for n in [250,500,1000]:
            Seeds=[rng.integers(0,1000000) for i in range(50)]
            for seed in Seeds:
                V1=hartigan(n,k,d,seed,10,rep,3,0.3)
                f.write("H: "+str(seed)+" ")
                for i in range(len(V1)):
                    f.write(str(V1[i])+" ")
                f.write("\n")
                #f.write("H: "+str(seed)+" ")
                #for i in range(len(V2)):
                #    f.write(str(V2[i])+" ")
                #f.write("\n")
                W1=smartigan(n,k,d,seed,10,rep,3,0.3)
                f.write("S: "+str(seed)+" ")
                for i in range(len(W1)):
                    f.write(str(W1[i])+" ")
                f.write("\n")
                
                V1=hartigan(n,k,d,seed,10,rep,5,0.1)
                f.write("H: "+str(seed)+" ")
                for i in range(len(V1)):
                    f.write(str(V1[i])+" ")
                f.write("\n")
                W1=smartigan(n,k,d,seed,10,rep,5,0.1)
                f.write("S: "+str(seed)+" ")
                for i in range(len(W1)):
                    f.write(str(W1[i])+" ")
                f.write("\n")