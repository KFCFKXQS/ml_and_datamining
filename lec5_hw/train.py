from model import SoftmaxClassifier, display
from dataloader import load_cifar_10,download_and_extract_cifar10
import numpy as np
import torch

SEED=0
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATASET_DIR = "./dataset/cifar10"

if __name__ == "__main__":
    download_and_extract_cifar10(url=CIFAR10_URL, dataset_dir=DATASET_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar_10(seed=SEED)

    X_train = torch.FloatTensor(X_train).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.int64).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.int64).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.int64).to(device)

    learning_rates = [0.001, 0.002, 0.005]
    batch_sizes = [1, 1000, 5000, 10000]
    regs = [0, 1e-5, 1e-4, 1e-3]

    best_accuracy = 0.0
    best_params = {}
    best_model = None

    for lr in learning_rates:
        for bs in batch_sizes:
            for reg in regs:
                if lr != 0.001 or (lr == 0.001 and bs == 10000):
                    print(f"Training with learning rate {lr}  batch size {bs}  reg {reg}")
                    softmax_classifier = SoftmaxClassifier(
                        input_dim=X_train.shape[1], num_classes=10, learning_rate=lr,reg=reg, device=device, seed=SEED)
                    
                    train_loss_history, val_loss_history, train_acc_history, val_acc_history, current_accuracy = softmax_classifier.train(
                        X_train=X_train, Y_train=Y_train, X_test=X_val, Y_test=Y_val, 
                        num_iters=200000, batch_size=bs, 
                        early_stopping=True, wait=5000, verbose=False)
                    
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_params = {'learning_rate': lr, 'batch_size': bs, 'reg': reg}
                        best_model = softmax_classifier
                    if reg == 0:
                        reg_str = "0"
                    else:
                        reg_str = f"1e{int(np.log10(reg))}"

                    filename = f'.\checkpoints\weights_lr_{lr}_bs_{bs}_reg_{reg_str}_valacc_{current_accuracy:.4f}.pth'

                    best_model.save(filename)

    print(f"Best accuracy: {best_accuracy} with parameters {best_params}")
    display(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
