from model import SoftmaxClassifier
from dataloader import load_cifar_10, get_image, load_labels, download_and_extract_cifar10
from check import seed_init
from train import SEED
import torch
import numpy as np
import matplotlib.pylab as plt


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATASET_DIR = "./dataset/cifar10"

# Create `./dataset/cifar10`  and put all 7 .pth files in folder `cifar10`
download_and_extract_cifar10(CIFAR10_URL, DATASET_DIR)

if __name__ == '__main__':
    seed_init(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = SoftmaxClassifier(input_dim=3072, num_classes=10)
    classifier.load('./checkpoints/weights_lr_0.005_bs_5000_reg_1e-3_valacc_0.4120.pth')
    
    classifier.device=device
    _, _, _, _, X_test, Y_test = load_cifar_10(seed=SEED)
    
    X_test = torch.FloatTensor(X_test).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.int64).to(device)
    Y_pred = classifier.predict(X_test)
    print(f'Accuracy on TestBatch: {torch.sum(torch.eq(Y_pred, Y_test)).item() / len(Y_test)}')

    labels = load_labels()[b'label_names']
    labels = [label.decode('ASCII') for label in labels ]
    
    correct_indices = np.where(Y_pred.cpu().numpy() == Y_test.cpu().numpy())[0]
    incorrect_indices = np.where(Y_pred.cpu().numpy() != Y_test.cpu().numpy())[0]

    # randomly show 8 corrects & 8 incorrects
    seed_init(SEED)
    selected_correct = np.random.choice(correct_indices, 8, replace=False)
    seed_init(SEED)
    selected_incorrect = np.random.choice(incorrect_indices, 8, replace=False)

    all_selected_indices = np.concatenate([selected_correct, selected_incorrect])

    # 显示选择的图片
    plt.figure(figsize=(12, 12))
    for i, index in enumerate(all_selected_indices):
        plt.subplot(4, 4, i+1)
        im = np.array(get_image(X_test[index].to('cpu')))
        plt.imshow(im)
        if i < 8:
            title_prefix = "Correct: "
            title_color = "green"
        else:
            title_prefix = "Incorrect: "
            title_color = "red"
        plt.title(f"{title_prefix}Real: {labels[Y_test[index].item()]}, Pred: {labels[Y_pred[index].item()]}",
                  color=title_color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()