import torch
from torch.profiler import profile, record_function, ProfilerActivity

def softmax_loss_vectorized(W, X, y, reg, device):
    loss = 0.0
    dW = torch.zeros_like(W, device=device)
    if W.device.type != device.type: W = W.to(device)
    if X.device.type != device.type: X = X.to(device)
    if y.device.type != device.type: y = y.to(device)  
    Y = torch.matmul(X, W)  # Y = XW
    max_val = torch.max(Y, dim=1, keepdim=True).values
    Y_exp = torch.exp(Y - max_val)
    Y_softmax = Y_exp / torch.sum(Y_exp, dim=1, keepdim=True)  # softmax(Y)

    indices = y.view(-1, 1)
    cross_entrophy = -torch.log(Y_softmax.gather(1, indices).squeeze())

    loss = torch.mean(cross_entrophy)
    loss += reg * torch.sum(W*W)

    Y_softmax.scatter_(1, indices, Y_softmax.gather(1, indices) - 1)
    Y_softmax /= len(y)
    dW = torch.mm(X.t(), Y_softmax)
    dW += 2 * reg * W

    return loss, dW


if __name__ == "__main__":
    import os
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    import sys
    sys.path.append(f'{os.getcwd()}\..')

    from check import check_loss_n_dW

    check_loss_n_dW(softmax_function=softmax_loss_vectorized, test_size=3000, test_times=50)