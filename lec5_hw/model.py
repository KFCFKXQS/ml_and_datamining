import torch

import os
import sys
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
sys.path.append(f'{os.getcwd()}softmax/')
from softmax.softmax_vectorized import softmax_loss_vectorized
from check import seed_init

class SoftmaxClassifier:
    def __init__(self, input_dim, num_classes=10,
                 learning_rate=1e-2, reg=1e-5,
                 device='cpu', seed=None):
        # set seed
        seed_init(seed)
        self.W = torch.randn(input_dim, num_classes, device=device) * 0.001
        self.lr = learning_rate
        self.reg = reg
        self.device = device
        self.seed = seed

    def train(self,
              X_train, Y_train, X_test, Y_test,
              num_iters=100, batch_size=200,
              verbose=True, early_stopping = False, wait=10):

        num_train, _ = X_train.shape
        train_loss_history = []
        test_loss_history = []
        train_acc_history = []
        test_acc_history = []
        best_test_accuracy = 0.0
        if early_stopping:
            best_test_accuracy = 0.0 
            patience = wait # epochs to wait
            wait = 0  # counter
        seed_init(self.seed)
        from tqdm import tqdm
        for iteration in tqdm(range(num_iters)):
            batch_indices = torch.randint(0, num_train, (batch_size, ))
            # print(batch_indices)
            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            loss, dW = softmax_loss_vectorized(
                self.W, X_batch, Y_batch, self.reg, self.device)
            
            self.W -= self.lr * dW
            test_preds = self.predict(X_test)
            test_accuracy = self.accuracy(test_preds, Y_test)
            if early_stopping:
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_model_weights = self.W.clone()  # Save the best model weights
                    wait = 0
                else:
                        wait += 1

            if early_stopping and wait >= patience:
                tqdm.write(f"Early stopping triggered with best acc {best_test_accuracy}")
                self.W = best_model_weights  # Load the best weights
                break
            if iteration == 0 or (iteration+1) % 100 == 0:
                train_preds = self.predict(X_train)
                train_accuracy = self.accuracy(train_preds, Y_train)
                
                test_preds = self.predict(X_test)
                test_accuracy = self.accuracy(test_preds, Y_test)
                
                test_loss = self.test(X_test=X_test, Y_test=Y_test)

                train_loss_history.append(loss.item())
                test_loss_history.append(test_loss.item())
                train_acc_history.append(train_accuracy)
                test_acc_history.append(test_accuracy)
                if verbose :
                    print(
                    f"Iteration {iteration+1}/{num_iters} | train_loss: {loss.item():.5f} | test_loss: {test_loss.item():.5f} | train_acc: {train_accuracy:.5f} | test_acc: {test_accuracy:.5f}")

                
        return train_loss_history, test_loss_history, train_acc_history, test_acc_history, best_test_accuracy

    def predict(self, X):
        if X.device.type != self.device:
            X = X.to(self.device)
        Y = torch.matmul(X, self.W)
        _, preds = torch.max(Y, dim=1)
        return preds
    
    def accuracy(self, Y_pred, Y_true):
        return torch.mean((Y_pred == Y_true.to(self.device)).float()).item()

    def test(self, X_test, Y_test):
        test_loss, _ = softmax_loss_vectorized(
            self.W, X_test, Y_test, self.reg, self.device)
        return test_loss

    def save(self, path="softmax_classifier_weights.pth"):
        """Save the model weights."""
        torch.save(self.W, path)

    def load(self, path="softmax_classifier_weights.pth"):
        """Load the model weights."""
        self.W = torch.load(path)

def display(train_loss_history, test_loss_history, train_acc_history, test_acc_history):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and test loss on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Iterations (in hundreds)')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_loss_history, color='blue', label='Train Loss')
    ax1.plot(test_loss_history, color='red', label='Test Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(train_acc_history, color='green', label='Train Acc')
    ax2.plot(test_acc_history, color='orange', label='Test Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and show the plot
    plt.title('Softmax Classifier')
    fig.tight_layout()
    ax1.legend()
    ax2.legend()
    plt.show()

