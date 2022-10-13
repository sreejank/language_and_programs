"""
Code to calculate performance metric (run NN heuristic 1000 times, then z-score white tiles revealed by the distribution of performance.)
"""
import numpy as np 

def random_norm_stats(maze,start):
    p=[]
    for _ in range(1000):
        board=np.ones(maze.shape)*-1
        board[start[0],start[1]]=1
        performance=0
        while np.sum(board==1)<np.sum(maze==1):
            idxs=np.where(board==-1)
            r=np.random.choice(np.arange(len(idxs[0])))
            board[idxs[0][r],idxs[1][r]]=maze[idxs[0][r],idxs[1][r]]
            if maze[idxs[0][r],idxs[1][r]]==0:
                performance+=1
        p.append(performance)
    return np.mean(p),np.std(p,ddof=1)

def get_random_proximal_tile(board):

    n=board.shape[0]
    legal=np.zeros((n,n))
    distance=0
    while np.sum(legal)==0:
        #print(distance)
        distance+=1
        red_idx=np.where(board==1)
        for i in range(len(red_idx[0])):
            r=red_idx[0][i]
            c=red_idx[1][i]
            legal[max(r-distance,0):min(r+(distance+1),n),max(c-(distance),0):min(c+(distance+1),n)]=1
        legal=legal*(board==-1).astype('int')
    idxs=np.where(legal==1)
    r=np.random.choice(np.arange(len(idxs[0])))
    return (idxs[0][r],idxs[1][r])

def proximal_norm_stats(maze,start):
    p=[]
    n=maze.shape[0]
    for _ in range(1000):
        board=np.ones(maze.shape)*-1
        board[start[0],start[1]]=1
        performance=0
        while np.sum(board==1)<np.sum(maze==1):
            choice=get_random_proximal_tile(board)
            board[choice[0],choice[1]]=maze[choice[0],choice[1]]        
            if maze[choice[0],choice[1]]==0:
                performance+=1
        p.append(performance)
    return np.mean(p),np.std(p,ddof=1)

def calculate_performance(performances_comp,performances_noncomp,comp_mazes,null_mazes,comp_starts,null_starts):

    normc=[]
    normnc=[]
    normalize_fn=proximal_norm_stats

    for i in range(len(comp_mazes)):
        print(i)
        comp_maze=comp_mazes[i]
        comp_start=comp_starts[i]
        normc.append(normalize_fn(comp_maze,comp_start))
        if i<len(null_mazes):
            null_start=null_starts[i]
            null_maze=null_mazes[i]
            normnc.append(normalize_fn(null_maze,null_start))

    normc=np.asarray(normc)
    normnc=np.asarray(normnc)

    for i in range(performances_comp.shape[0]):
        for j in range(performances_comp.shape[1]):
            norm=normc[j]
            performances_comp[i,j]=(performances_comp[i,j]-norm[0])/(norm[1])

    for i in range(performances_noncomp.shape[0]):
        for j in range(performances_noncomp.shape[1]):
            norm=normnc[j]
            performances_noncomp[i,j]=(performances_noncomp[i,j]-norm[0])/(norm[1])
    return performances_comp,performances_noncomp



