from network import reconize_network
from Augmentation import augmentation_dataset
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dump_model(model):
    with open('../model1.pkl', 'wb') as f:
        pickle.dump(model, f)

transformer = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ToTensorV2(p=1.0)
])


def evaluate_model(model, test_loader, device):
    # Set model to evaluation mode
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            # Collect labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Binarize the labels for ROC calculations
    classes = np.unique(all_labels)
    all_labels_binarized = label_binarize(all_labels, classes=classes)

    # Compute macro and micro ROC AUC
    macro_auc = roc_auc_score(all_labels_binarized, all_probs, average='macro', multi_class='ovr')
    micro_auc = roc_auc_score(all_labels_binarized, all_probs, average='micro', multi_class='ovr')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.figure(figsize=(5,5))
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("Confusion_Matrix.png")
    plt.close()


    # Compute ROC curves for each class
    n_classes = len(classes)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(5,5))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], np.array(all_probs)[:, i])
        roc_auc[i] = roc_auc_score(all_labels_binarized[:, i], np.array(all_probs)[:, i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig("ROC.png")

    plt.close()

    return {
        'accuracy': accuracy,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc
    }



model = reconize_network()
model.load_state_dict(torch.load("save.pt"))
model = model.to(device)
test_set = augmentation_dataset(root='./data', split='test', download=True,transform=transformer)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

results = evaluate_model(model,test_loader,device)
print("Accuracy:", results['accuracy'])
print("Macro AUC:", results['macro_auc'])
print("Micro AUC:", results['micro_auc'])
print("Confusion Matrix:\n", results['confusion_matrix'])

