import torch
import torch.nn as nn
import torch.optim as optim
import copy

# for checking correctness
def seed_init(seed=0):
    import numpy as np
    import random
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

def check_loss_n_dW_on_device(softmax_function, device, test_times = 100, test_size = 20): 
    import numpy as np
    import torch.nn as nn
    import time
    from tqdm import tqdm

    total_time = 0.0
    for i in tqdm(range(test_times)):
        N = np.random.randint(1, test_size+1)
        D = np.random.randint(1, test_size+1)
        C = np.random.randint(1, test_size+1)
        X = torch.rand(N, D).to(device)
        W = torch.rand(D, C, requires_grad=True).to(device)
        y = torch.randint(0, C, (N,)).to(device)
        X_clone = X.clone()
        W_clone = W.clone()
        y_clone = y.clone()
        reg = np.random.rand()
        #print(f'reg: {reg}')
        
        start_time = time.time()
        # results from my function:
        loss, dW = softmax_function(W=W, X=X, y=y, reg=reg, device=device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        #print(f'device {X.device}')
        assert (X.device.type == device.type and
                W.device.type == device.type and 
                y.device.type == device.type and 
                torch.equal(X, X_clone) and
                torch.equal(W, W_clone) and
                torch.equal(y, y_clone)      
                ), f"data changed accidentally"

        # pytorch's func
        W.retain_grad()
        inputs = torch.matmul(X, W).to(device)
        #softmax_outputs = F.softmax(inputs, dim=1)
        loss_function = nn.CrossEntropyLoss().to(device)
        base_loss = loss_function(inputs, y)

        # L2 Regularization
        l2_loss = reg * torch.sum(W * W)

        # Combined Loss
        combined_loss = base_loss + l2_loss
        combined_loss.backward()

        torch_dW = W.grad

        assert torch.abs(loss - combined_loss).item() < 1e-2, f"{i} Loss mismatch {loss} \n {combined_loss}"
        assert torch.allclose(dW.to(device), torch_dW, atol=1e-2), f"{i} Gradient mismatch"
    
    average_time = total_time / test_times
    print(f"All checks pass on device '{device}' |  avg: {average_time:.5f} s/iter")
    return True


def check_loss_n_dW(softmax_function, test_times = 100, test_size = 20):
    device0 = torch.device('cpu')
    check_loss_n_dW_on_device(softmax_function=softmax_function, device=device0, test_times = test_times, test_size = test_size)
    if not torch.cuda.is_available():
        print('cuda not found')
    else : 
        device1 = torch.device('cuda')
        check_loss_n_dW_on_device(softmax_function=softmax_function, device=device1, test_times = test_times, test_size = test_size)



class SoftmaxClassifierTorch(nn.Module):
    def __init__(self, input_dim, num_classes=10,
                 learning_rate=1e-2, reg=1e-5,
                 device=torch.device('cpu'), seed=None):
        super(SoftmaxClassifierTorch, self).__init__()

        
        self.fc = nn.Linear(input_dim,num_classes,  bias=False,
                            device=device)  # Softmax without bias
        seed_init(seed)
        self.fc.weight.data = torch.randn(input_dim, num_classes,device=device).t()
        self.fc.weight.data *= 0.001

        self.lr = learning_rate
        self.reg = reg
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.seed = seed

    def forward(self, x):
        return self.fc(x)

    def train_model(self,
        X_train, Y_train, X_test, Y_test,
        num_iters=100, batch_size=200,
        verbose=True, early_stopping=False, patience=10):
        optimizer = optim.SGD(
            self.parameters(), lr=self.lr, weight_decay=0.0)

        train_loss_history = []
        test_loss_history = []
        train_acc_history = []
        test_acc_history = []

        if early_stopping:
            best_test_accuracy = 0.0
            patience = patience
            wait_counter = 0

        from tqdm import tqdm
        seed_init(self.seed)
        for iteration in tqdm(range(num_iters)):
            batch_indices = torch.randint(0, len(X_train), (batch_size,))
            X_batch = X_train[batch_indices].to(self.device)
            Y_batch = Y_train[batch_indices].to(self.device)
            optimizer.zero_grad()

            outputs = self(X_batch)
            loss = self.criterion(outputs, Y_batch) + self.reg * torch.sum(self.fc.weight ** 2)
            #print(loss)
            loss.backward()
            #print(self.fc.weight.grad)
            optimizer.step()

            test_accuracy = 0.0
            test_preds = self.predict(X_test)
            test_accuracy = self.accuracy(test_preds, Y_test)
            if early_stopping:
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_model_weights = copy.deepcopy(self.fc.weight.data)
                    wait_counter = 0
                else:
                    wait_counter += 1

            if early_stopping and wait_counter >= patience:
                tqdm.write(f"Early stopping triggered with best acc {best_test_accuracy}")
                self.W = best_model_weights  # Load the best weights
                break
            if iteration == 0 or (iteration+1) % 100 == 0:
                train_preds = self.predict(X_train)
                train_accuracy = self.accuracy(train_preds, Y_train)

                test_loss = self.criterion(
                    self(X_test), Y_test) + self.reg * torch.sum(self.fc.weight.data**2)

                train_loss_history.append(loss.item())
                test_loss_history.append(test_loss.item())
                train_acc_history.append(train_accuracy)
                test_acc_history.append(test_accuracy)

                if verbose:
                    print(
                        f"Iteration {iteration+1}/{num_iters} | train_loss: {loss.item():.5f} | test_loss: {test_loss.item():.5f} | train_acc: {train_accuracy:.5f} | test_acc: {test_accuracy:.5f}")

        return train_loss_history, test_loss_history, train_acc_history, test_acc_history

    def predict(self, X):
        outputs = self.forward(X)
        _, preds = torch.max(outputs, dim=1)
        return preds

    def accuracy(self, Y_pred, Y_true):
        return (Y_pred == Y_true).float().mean().item()

#SOFTMAX FUNCTIONS CHECK
############################################################################################################
if __name__ == "__main__":
    seed_init(0)
    import os
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    import sys
    sys.path.append(f'{os.getcwd()}')

    from check import check_loss_n_dW
    from softmax.softmax_naive import softmax_loss_naive
    from softmax.softmax_vectorized import softmax_loss_vectorized
    print("SOFTMAX_LOSS_NAIVE TEST")
    check_loss_n_dW(softmax_function=softmax_loss_naive, test_size=15,test_times=50)
    print("SOFTMAX_LOSS_VECTORIZED TEST")
    check_loss_n_dW(softmax_function=softmax_loss_vectorized, test_size=3000, test_times=50)
    print('###########################################')


#MODEL CHECK
############################################################################################################
if __name__ == "__main__":
    seed_init(0)
    from model import SoftmaxClassifier
    # test: fitting argmax
    device = torch.device('cuda')
    
    # My classifier
    softmax_classifier = SoftmaxClassifier(
        input_dim=5, num_classes=5, learning_rate=0.1, device=device, seed=0)
    # Torch classifier
    softmax_classifier_torch = SoftmaxClassifierTorch(
        input_dim=5, num_classes=5, learning_rate=0.1, device=device, seed=0)
    
    X_train = torch.rand(10000, 5).to(device)
    Y_train = torch.argmax(X_train, dim=1).to(device)
    X_test = torch.rand(2000, 5).to(device)
    Y_test = torch.argmax(X_test, dim=1).to(device)

    loss_history, test_loss_history, train_acc_history, test_acc_history, _ = softmax_classifier.train(
        X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, batch_size=500, num_iters=5000,verbose=False)

    loss_history_torch, test_loss_history_torch ,train_acc_history_torch, test_acc_history_torch= softmax_classifier_torch.train_model(
        X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, batch_size=500, num_iters=5000, verbose=False)
    
    print(test_loss_history_torch[:5])
    print(test_loss_history[:5])
    assert all(abs(a - b) < 1e-2 for a, b in zip(test_loss_history, test_loss_history_torch)), "model is incorrect"
    assert all(abs(a - b) < 1e-2 for a, b in zip(test_acc_history, test_acc_history_torch)), "model is incorrect"
    print("MODEL TEST PASSED")
    # Plotting the loss histories
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    plt.plot(test_loss_history, color='magenta', linestyle='--', alpha=0.7, label='My Test Loss')
    plt.plot(test_loss_history_torch, color='cyan', alpha=0.7, label='PyTorch Test Loss')
    plt.plot(test_acc_history, color='red', linestyle='--', alpha=0.7, label='My Test ACC')
    plt.plot(test_acc_history_torch, color='blue', alpha=0.7, label='PyTorch Test ACC')

    plt.legend()
    plt.xlabel('Iterations (in hundreds)')
    plt.ylabel('Loss')
    plt.title('Comparison of Original and PyTorch Softmax Classifiers')
    plt.show()


# DATA LOAD CHECK
#########################################################################################
if __name__ == "__main__":
    seed_init(0)
    from dataloader import *
    #labels reading test
    print('###########################################')
    print('LABEL READING TEST')
    label_names = load_labels('./dataset/cifar10/batches.meta')
    print(label_names[b'num_cases_per_batch'],'*',label_names[b'num_vis'])
    print('labels: ',label_names[b'label_names'])
    labels = label_names[b'label_names']
    print(labels, '\n')


    #batch loading test.
    print('###########################################')
    print('BATCH LOADING TEST')
    batch_test = load_batch('./dataset/cifar10/data_batch_1')
    print(batch_test.keys())
    print('batch_labels:', batch_test[b'batch_label'])
    import numpy as np
    print('labels in batch_test:', np.array(batch_test[b'labels']).shape, batch_test[b'labels'][0:10])
    for i in range(10):
        if not (i in batch_test[b'labels']):
            print(f'lack of {i}')

    print(np.array(batch_test[b'data']).shape)
    print('filenames in batch_test:', np.array(batch_test[b'filenames']).shape, batch_test[b'filenames'][0:3])

    # cifar10 load test:
    print('###########################################')
    print('CIFAR 10 LOADING TEST')
    X_train, Y_train,X_val, Y_val, X_test, Y_test = load_cifar_10()
    print('training set')
    print(np.array(X_train).shape)
    print(np.array(Y_train).shape)
    print('validation set')
    print(np.array(X_val).shape)
    print(np.array(Y_val).shape)
    print('test set')
    print(np.array(X_test).shape)
    print(np.array(Y_test).shape)

    #image loading test
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3)
    plt.suptitle('CIFAR 10 LOADING TEST')
    for i, ax in enumerate(axes.flat):
        # get images and label
        image = get_image(X_val[i])
        label = labels[Y_val[i]].decode('utf-8')
        ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')
    plt.show()
    print("DATALOADER TEST PASSED")

