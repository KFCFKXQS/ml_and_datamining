import numpy as np
import matplotlib.pyplot as plt

def function_1(x):
    return 4 * x[0]**2 - 4 * x[0] * x[1] + 2 * x[1]**2


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # compute the partial_derivative for each element in x
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        
        x[idx] = float(tmp_val) - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()
    return grad


def gradient_descent(f, init_x, ax, lr=0.01, step_num=100):
    x = init_x.copy()
    x_values = [x.copy()]
    
    # gradient descent
    for _ in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        x_values.append(x.copy())
    
    x_values = np.array(x_values)

    # plot
    ax.plot(x_values[:, 0], x_values[:, 1], 'ro-', markersize=2)
    if lr > 0.1:
        ax.annotate(f'lr = {lr}, Minimum value: {f(x):.2e}', xy=(0.5, 1.47), xycoords='axes fraction',
                fontsize=12, ha='center')
        ax.annotate(f'Steps = {step_num}', xy=(0.5, 1.32), xycoords='axes fraction',
                fontsize=12, ha='center')
    else:
        ax.annotate(f'lr = {lr}, Minimum value: {f(x):.2e}', xy=(0.5, 1.25), xycoords='axes fraction',
                fontsize=12, ha='center')
        ax.annotate(f'Steps = {step_num}', xy=(0.5, 1.10), xycoords='axes fraction',
                fontsize=12, ha='center')
    
    # end point of optimization
    end_x, end_y = x_values[-1]
    if np.abs(end_x) > 1e4 or np.abs(end_y) > 1e4:
        coord_str = f'end: ({end_x:.2e}, {end_y:.2e})'
    else:
        coord_str = f'end: ({end_x:.4f}, {end_y:.4f})'

    ax.annotate(coord_str,
                xy=(end_x, end_y), xycoords='data',
                textcoords='offset points',
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1),
                horizontalalignment='right',
                verticalalignment='bottom',
                xytext=(0,10))

    return x



if __name__ == "__main__":
    # (x_0, y_0) = (2, 3)
    init_x = np.array([2.0, 3.0])
    
    fig, axs = plt.subplots(4, 4, figsize=(20, 10), gridspec_kw={'hspace': 0.8})
    axs = axs.flatten()
    
    # try various combinations of lrs and steps
    learning_rates = [0.5, 0.1, 0.01, 0.001]
    steps = [10, 100, 1000, 10000]
    
    x0 = np.arange(-20, 20, 0.1)
    x1 = np.arange(-20, 20, 0.1)
    X, Y = np.meshgrid(x0, x1)
    Z = np.array([function_1(np.array([xx, yy])) for xx, yy in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    
    counter = 0
    for lr in learning_rates:
        for step in steps:
            ax = axs[counter]
            ax.contour(X, Y, Z, levels=np.logspace(0, 3, 10))
            gradient_descent(function_1, init_x, ax, lr=lr, step_num=step)
            counter += 1

    plt.show()
