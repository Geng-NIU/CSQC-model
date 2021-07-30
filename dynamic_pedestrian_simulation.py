import numpy as np
import math
'''
Module presents a pedestrian model that reflect the dynamic participants/sensors observations.
'''


def update_XY_position(N_participants, T, X_Lim, Y_Lim):
    """
    N_participants is the number of participants that will move;
    T is the number of time steps;
    X_Lim, Y_Lim are the boundary of the study area. 40  80;
    L_max is the maximum distance that one participant can move within one time step;
    p_move is the probability that an agent moves;
    L_min is the minimum distance that one participant can move within one time step.
    """
    #  ============ first, randomly set the initial position ====================
    X = np.zeros((T, N_participants))
    Y = np.zeros((T, N_participants))
    X[0, :] = np.random.randint(0, 39, size=N_participants)
    Y[0, :] = np.random.randint(0, 79, size=N_participants)
    # ================== L_max & p_move are vector, L_min is constant ===============
    L_max = np.random.uniform(2, 10, size=N_participants)
    p_move = np.random.uniform(.5, .8, size=N_participants)
    L_min = 1
    for t in range(1, T):
        temp0 = np.random.uniform(0, 1, size=N_participants)
        is_move = []
        for i in range(N_participants):
            if temp0[i] >= (1 - p_move[i]):
                move = 1
            else:
                move = 0
            is_move.append(move)
        temp1 = np.random.uniform(0, 1, size=N_participants)
        L = L_max * temp1
        L = L * is_move  # the distance that agents moved within one time step
        theta = 2 * math.pi * np.random.uniform(0, 1, size=N_participants)  # the directions that agents moved to
        X[t, :] = X[t - 1, :] + L * np.cos(theta)
        Y[t, :] = Y[t - 1, :] + L * np.sin(theta)  # update X and Y according to L and theta
        is_inbound_X, is_inbound_Y = [], []
        for i in range(N_participants):
            if 0 <= X[t, i] <= X_Lim:  # logics determing if X is within limit
                inbound_X = 1
            else:
                inbound_X = 0
            if 0 <= Y[t, i] <= Y_Lim:  # logics determing if Y is within limit
                inbound_Y = 1
            else:
                inbound_Y = 0
            is_inbound_X.append(inbound_X)
            is_inbound_Y.append(inbound_Y)
        while (sum(is_inbound_X) + sum(is_inbound_Y)) < (N_participants * 2):
            ID_out_bound = []
            ID = []
            for i in range(N_participants):
                if is_inbound_X[i] == 0 or is_inbound_Y[i] == 0:
                    out_bound_temp = 1
                    ID_temp = i
                    ID_out_bound.append(out_bound_temp)
                    ID.append(ID_temp)
                else:
                    continue
            alphas = np.random.uniform(0, 1, size=sum(ID_out_bound))
            teltas = 0.5 * math.pi * np.random.uniform(0, 1, size=sum(ID_out_bound))
            # adjust the L and theta to avoid crossing boundary
            X[t, ID] = X[t - 1, ID] + alphas * L[ID] * np.sin(theta[ID] + teltas)
            Y[t, ID] = Y[t - 1, ID] + alphas * L[ID] * np.cos(theta[ID] + teltas)
            is_inbound_X, is_inbound_Y = [], []
            for i in range(N_participants):
                if 0 <= X[t, i] <= X_Lim:  # logics determing if X is within limit
                    inbound_X = 1
                else:
                    inbound_X = 0
                if 0 <= Y[t, i] <= Y_Lim:  # logics determing if Y is within limit
                    inbound_Y = 1
                else:
                    inbound_Y = 0
                is_inbound_X.append(inbound_X)
                is_inbound_Y.append(inbound_Y)
        else:
            continue
    X_path, Y_path = [], []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            xx = math.ceil(X[i][j])
            yy = math.ceil(Y[i][j])
            X_path.append(xx)
            Y_path.append(yy)
    X_path = np.array(X_path).reshape(X.shape[0], X.shape[1])
    Y_path = np.array(Y_path).reshape(X.shape[0], X.shape[1])
    return X_path, Y_path
