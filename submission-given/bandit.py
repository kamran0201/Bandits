import numpy as np
import matplotlib.pyplot as plt
import sys, os, argparse, random, time
from math import *

# task 1 epsilon greedy
def generate_reward_epsilon_greedy_t1(arms,reward,pulls,mean,arm):
    r=random.uniform(0,1)
    pulls[arm]+=1
    if r<arms[arm]:
        reward[arm]+=1
    mean[arm]=reward[arm]/pulls[arm]

def epsilon_greedy_t1(arms, epsilon, horizon):
    narms=len(arms)
    reward=[0]*narms
    pulls=[0]*narms
    mean=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_epsilon_greedy_t1(arms,reward,pulls,mean,i)

    for i in range(horizon-narms):
        e=random.uniform(0,1)
        if e<=epsilon:
            a=random.randint(0,narms-1)
            generate_reward_epsilon_greedy_t1(arms,reward,pulls,mean,a)
        else:
            maxmean=max(mean)
            maxarms=[j for j, v in enumerate(mean) if v==maxmean]
            a=random.choice(maxarms)
            generate_reward_epsilon_greedy_t1(arms,reward,pulls,mean,a)
    
    # print(reward)
    # print(pulls)
    # print(mean)
    # print(arms)

    totalreward=sum(reward)
    regret=horizon*max(arms)-totalreward
    highs=0
    return [regret,highs]

# task 1 ucb
def generate_reward_ucb_t1(arms,reward,pulls,ucb,arm):
    r=random.uniform(0,1)
    pulls[arm]+=1
    t=sum(pulls)
    if r<arms[arm]:
        reward[arm]+=1
    for i in range(min(len(arms),t)):
        ucb[i]=reward[i]/pulls[i]+sqrt(2*log(t)/pulls[i])

def ucb_t1(arms, horizon):
    narms=len(arms)
    reward=[0]*narms
    pulls=[0]*narms
    ucb=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_ucb_t1(arms,reward,pulls,ucb,i)
    
    for i in range(horizon-narms):
        maxucb=max(ucb)
        maxarms=[j for j, v in enumerate(ucb) if v==maxucb]
        a=random.choice(maxarms)
        generate_reward_ucb_t1(arms,reward,pulls,ucb,a)
    
    # print(reward)
    # print(pulls)
    # print(ucb)
    # print(arms)
    
    totalreward=sum(reward)
    regret=horizon*max(arms)-totalreward
    highs=0
    return [regret,highs]

# task 1 kl ucb
def kl(p,q):
    if p==1 or q==1:
        return p*log(p/q)
    elif p==0 or q==0:
        return (1-p)*log((1-p)/(1-q))
    else:
        return p*log(p/q) + (1-p)*log((1-p)/(1-q))

def q_value(p,bound):
    if p==1:
        return 1
    q_range=np.arange(p,0.999,0.02)
    diff=[]
    for i in range(len(q_range)):
        temp=bound-kl(p,q_range[i])
        if bound-kl(p,q_range[i])<0:
            temp=inf
        diff.append(temp)
    mindiff=min(diff)
    return q_range[diff.index(mindiff)]

def generate_reward_kl_ucb_t1(arms,reward,pulls,klucb,arm):
    r=random.uniform(0,1)
    pulls[arm]+=1
    t=sum(pulls)
    if r<arms[arm]:
        reward[arm]+=1
    if t==1 or t==2:
        return
    for i in range(min(len(arms),t)):
        p=reward[i]/pulls[i]
        bound=(log(t)+3*log(log(t)))/pulls[i]
        klucb[i]=q_value(p,bound)

def kl_ucb_t1(arms, horizon):
    narms=len(arms)
    reward=[0]*narms
    pulls=[0]*narms
    klucb=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_kl_ucb_t1(arms,reward,pulls,klucb,i)
    
    for i in range(horizon-narms):
        maxklucb=max(klucb)
        maxarms=[j for j, v in enumerate(klucb) if v==maxklucb]
        a=random.choice(maxarms)
        generate_reward_kl_ucb_t1(arms,reward,pulls,klucb,a)
    
    # print(reward)
    # print(pulls)
    # print(klucb)
    # print(arms)
    
    totalreward=sum(reward)
    regret=horizon*max(arms)-totalreward
    highs=0
    return [regret,highs]

# task 1 thompson sampling
def generate_reward_thomson_sampling_t1(arms,reward,pulls,ts,arm):
    r=random.uniform(0,1)
    pulls[arm]+=1
    if r<arms[arm]:
        reward[arm]+=1
    for i in range(len(arms)):
        ts[i]=np.random.beta(reward[i]+1,pulls[i]-reward[i]+1)

def thompson_sampling_t1(arms,horizon):
    narms=len(arms)
    reward=[0]*narms
    pulls=[0]*narms
    ts=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_thomson_sampling_t1(arms,reward,pulls,ts,i)

    for i in range(horizon-narms):
        maxts=max(ts)
        maxarms=[j for j, v in enumerate(ts) if v==maxts]
        a=random.choice(maxarms)
        generate_reward_thomson_sampling_t1(arms,reward,pulls,ts,a)
    
    # print(reward)
    # print(pulls)
    # print(ts)
    # print(arms)
    
    totalreward=sum(reward)
    regret=horizon*max(arms)-totalreward
    highs=0
    return [regret,highs]

# task 2 ucb
def generate_reward_ucb_t2(arms,reward,pulls,ucb,arm,scale):
    r=random.uniform(0,1)
    pulls[arm]+=1
    t=sum(pulls)
    if r<arms[arm]:
        reward[arm]+=1
    for i in range(min(len(arms),t)):
        ucb[i]=reward[i]/pulls[i]+sqrt(scale*log(t)/pulls[i])

def ucb_t2(arms, horizon, scale):
    narms=len(arms)
    reward=[0]*narms
    pulls=[0]*narms
    ucb=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_ucb_t2(arms,reward,pulls,ucb,i,scale)
    
    for i in range(horizon-narms):
        maxucb=max(ucb)
        maxarms=[j for j, v in enumerate(ucb) if v==maxucb]
        a=random.choice(maxarms)
        generate_reward_ucb_t2(arms,reward,pulls,ucb,a,scale)
    
    # print(reward)
    # print(pulls)
    # print(ucb)
    # print(arms)
    
    totalreward=sum(reward)
    regret=horizon*max(arms)-totalreward
    highs=0
    return [regret,highs]

# task 3
def generate_reward_alg_t3(cons_arms,support,reward,pulls,ts,arm):
    r=random.uniform(0,1)
    pulls[arm]+=1
    for i in range(len(support)):
        if r<cons_arms[arm][i] and i==0:
            reward[arm]+=support[i]
        elif r<cons_arms[arm][i] and r>=cons_arms[arm][i-1]:
            reward[arm]+=support[i]
    for i in range(len(cons_arms)):
        ts[i]=np.random.beta(reward[i]+1,pulls[i]-reward[i]+1)

def alg_t3(arms,support,horizon):
    narms=len(arms)
    nsupp=len(support)
    cons_arms=[]
    for i in range(narms):
        cons_arm=[]
        y=0
        for j in range(nsupp):
            y+=arms[i][j]
            cons_arm.append(y)
        cons_arms.append(cons_arm)
    
    # print(cons_arms)

    reward=[0]*narms
    pulls=[0]*narms
    ts=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_alg_t3(cons_arms,support,reward,pulls,ts,i)

    for i in range(horizon-narms):
        maxts=max(ts)
        maxarms=[j for j, v in enumerate(ts) if v==maxts]
        a=random.choice(maxarms)
        generate_reward_alg_t3(cons_arms,support,reward,pulls,ts,a)
    
    # print(reward)
    # print(pulls)
    # print(ts)
    # print(arms)
    
    totalreward=sum(reward)
    sa=np.array(arms).dot(np.array(support))
    regret=horizon*max(sa)-totalreward
    highs=0
    return [regret,highs]

# task 4
def generate_reward_alg_t4(cons_arms,support,high,pulls,ts,arm,threshold):
    r=random.uniform(0,1)
    pulls[arm]+=1
    reward=0
    for i in range(len(support)):
        if r<cons_arms[arm][i] and i==0:
            reward=support[i]
        elif r<cons_arms[arm][i] and r>=cons_arms[arm][i-1]:
            reward=support[i]
    if reward>threshold:
        high[arm]+=1
    for i in range(len(cons_arms)):
        ts[i]=np.random.beta(high[i]+1,pulls[i]-high[i]+1)

def alg_t4(arms,support,horizon,threshold):
    narms=len(arms)
    nsupp=len(support)
    cons_arms=[]
    for i in range(narms):
        cons_arm=[]
        y=0
        for j in range(nsupp):
            y+=arms[i][j]
            cons_arm.append(y)
        cons_arms.append(cons_arm)
    
    # print(cons_arms)

    high=[0]*narms
    pulls=[0]*narms
    ts=[0]*narms
    for i in range(min(narms,horizon)):
        generate_reward_alg_t4(cons_arms,support,high,pulls,ts,i,threshold)

    for i in range(horizon-narms):
        maxts=max(ts)
        maxarms=[j for j, v in enumerate(ts) if v==maxts]
        a=random.choice(maxarms)
        generate_reward_alg_t4(cons_arms,support,high,pulls,ts,a,threshold)
    
    # print(high)
    # print(pulls)
    # print(ts)
    # print(arms)
    
    highs=sum(high)
    x1=[i for i in range(nsupp) if support[i]>threshold]
    x2=[[arm[i] for i in range(nsupp) if i in x1] for arm in arms]
    x3=[sum(i) for i in x2]
    regret=horizon*max(x3)-highs
    return [regret,highs]

# main
instance=None
algorithm=None
randomSeed=0
epsilon=0.02
scale=2
threshold=0
horizon=0
reg=0
high=0
inp=sys.argv
inplen=len(inp)
for i in range(1,(inplen+1)//2):
    if inp[2*i-1] == "--instance":
        instance=inp[2*i]
    elif inp[2*i-1] == "--algorithm":
        algorithm=inp[2*i]
    elif inp[2*i-1] == "--randomSeed":
        randomSeed=int(inp[2*i])
    elif inp[2*i-1] == "--epsilon":
        epsilon=float(inp[2*i])
    elif inp[2*i-1] == "--scale":
        scale=float(inp[2*i])
    elif inp[2*i-1] == "--threshold":
        threshold=float(inp[2*i])
    elif inp[2*i-1] == "--horizon":
        horizon=int(inp[2*i])

random.seed(randomSeed)
np.random.seed(randomSeed)

if algorithm == "epsilon-greedy-t1" or algorithm == "ucb-t1" or algorithm == "kl-ucb-t1" or algorithm == "thompson-sampling-t1" or algorithm == "ucb-t2":
    file=open(instance,"r")
    arms=file.readlines()
    arms=[float(arm.rstrip()) for arm in arms]
    file.close()

if algorithm == "alg-t3" or algorithm == "alg-t4":
    file=open(instance,"r")
    temp=file.readlines()
    support=temp[0].rstrip().split()
    support=[float(s) for s in support]
    arms=temp[1:]
    arms=[arm.rstrip().split() for arm in arms]
    arms=[[float(a) for a in arm] for arm in arms]
    file.close()

if algorithm == "epsilon-greedy-t1":
    reghigh=epsilon_greedy_t1(arms, epsilon, horizon)
    reg=reghigh[0]
    high=reghigh[1]
elif algorithm == "ucb-t1":
    reghigh=ucb_t1(arms, horizon)
    reg=reghigh[0]
    high=reghigh[1]
elif algorithm == "kl-ucb-t1":
    reghigh=kl_ucb_t1(arms, horizon)
    reg=reghigh[0]
    high=reghigh[1]
elif algorithm == "thompson-sampling-t1":
    reghigh=thompson_sampling_t1(arms, horizon)
    reg=reghigh[0]
    high=reghigh[1]
elif algorithm == "ucb-t2":
    reghigh=ucb_t2(arms, horizon, scale)
    reg=reghigh[0]
    high=reghigh[1]
elif algorithm == "alg-t3":
    reghigh=alg_t3(arms, support, horizon)
    reg=reghigh[0]
    high=reghigh[1]
elif algorithm == "alg-t4":
    reghigh=alg_t4(arms, support, horizon, threshold)
    reg=reghigh[0]
    high=reghigh[1]


print(instance,algorithm,randomSeed,epsilon,scale,threshold,horizon,reg,high,sep=", ")
