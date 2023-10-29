import torch


def softmax_loss_naive(W, X, y, reg, device):

    def softmax_naive(y):
        '''
        y: (n, )
        '''
        y_soft_max = y.clone()
        # find max{s_i}
        max_val = torch.tensor(float('-inf'))
        for s_i in y_soft_max:
            if s_i > max_val:
                max_val = s_i.item()
        # sigma(exp(s_i - max))
        sigma = 0
        for s_i in y_soft_max:
            sigma += torch.exp(s_i - max_val)
        # exp(s_i - max) / sigma
        for i in range(len(y)):
            y_soft_max[i] = torch.exp(y[i] - max_val) / sigma
            # print(f'{i},{y[i]},max:,{max_val},{y[i] - max_val},{torch.exp(y[i] - max_val)}')
        return y_soft_max

    def R_W_naive(weights):
        # print(f'weights: {weights}')
        r_w = torch.tensor(0.0, dtype=torch.float, device=device)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                r_w += weights[i, j] * weights[i, j]
        return r_w

    def matmul_naive(X, W):
        Y = torch.zeros((X.shape[0], W.shape[1]),device=device)
        for i in range(X.shape[0]):
            for k in range(W.shape[0]):
                for j in range(W.shape[1]):
                    Y[i, j] += X[i, k] * W[k, j]
        return Y
    

    if W.device.type != device.type: W = W.to(device)
    if X.device.type != device.type: X = X.to(device)
    if y.device.type != device.type: y = y.to(device)

    loss = 0.0
    dW = torch.zeros_like(W,device=device)

    Y = matmul_naive(X, W)
    Y_softmax = torch.zeros_like(Y,device=device)  # (N, C)
    for i in range(Y.shape[0]):
        Y_softmax[i] = softmax_naive(Y[i])
    # print(f'Y_soft_max: {Y_softmax}')

    # loss_per_sample (N,)
    loss_per_sample = torch.zeros(Y_softmax.shape[0])
    for i in range(Y_softmax.shape[0]):
        loss_per_sample[i] = -torch.log(Y_softmax[i, y[i]])
    # print(f'loss_per_samp: {loss_per_sample}')
    loss = torch.mean(loss_per_sample) + reg * R_W_naive(W)
    
    # return loss
    T = torch.zeros_like(Y,device=device)
    dY = torch.zeros_like(Y,device=device)
    for i in range(len(y)):
        T[i, y[i]] = 1

    for i in range(Y_softmax.shape[0]):
        for j in range(Y_softmax.shape[1]):
            dY[i, j] = (Y_softmax[i, j] - T[i, j]) / X.shape[0]

    X_trans = X.t()
    dW = matmul_naive(X_trans, dY)
    rows, cols = W.shape
    for i in range(rows):
        for j in range(cols):
            dW[i][j] += 2 * reg * W[i][j]
    return loss, dW