
import os 
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def stochastic_gradient_sampling(model, optimizer, train_loader, criterion, device, ckpt_folder, use_tqdm = False):
    model.train()
    os.makedirs(ckpt_folder, exist_ok=True)
    if use_tqdm:
        train_loader = tqdm(train_loader, desc="Training", leave=False)
    else:
        train_loader = iter(train_loader)
        
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        gradient = {name: param.grad.clone() for name, param in model.named_parameters()}
        
        torch.save(gradient, os.path.join(ckpt_folder, f'gradients_{i}.pt'))
    

# === Обучение: одна эпоха ===
def train(model, criterion, optimizer, train_loader, full_loader, n_steps, device, base_folder, gradient_sampling = False, model_save = False):
    loss_trajectory = []
    model.train()

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        if i >= n_steps:
            break
        ckpt_folder = f'{base_folder}/trajectory_step_{i}'
        if gradient_sampling:
            stochastic_gradient_sampling(model, optimizer, full_loader, criterion, device, ckpt_folder)

        # Переходим в режим обучения
        model.train()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Прямой проход (forward pass)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()


        # Сохраняем веса модели на текущем шаге
        if model_save:
            torch.save(model.state_dict(), os.path.join(ckpt_folder, 'model.pt'))
        # Сохраняем значения лосса и точности
        loss_trajectory.append(loss.item())

    return loss_trajectory

def evaluate_sgd_update(model, gradient, lr, test_loader, criterion, device):
    model_copy = torch.nn.Module()  # Создаем копию модели
    model_copy.load_state_dict(model.state_dict())
    
    with torch.no_grad():
        for name, param in model_copy.named_parameters():
            if name in gradient:
                param -= lr * gradient[name]
    
    model_copy.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_copy(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)

def compute_gradient_distances(ckpt_folder):
    gradient_files = sorted(f for f in os.listdir(ckpt_folder) if f.startswith("gradients_") and f.endswith(".pt"))
    distance_dict = {}
    
    for name in torch.load(os.path.join(ckpt_folder, gradient_files[0])).keys():
        distance_dict[name] = torch.zeros((len(gradient_files), len(gradient_files)))
    
    for i in tqdm(range(len(gradient_files)), desc=f"Processing gradients (i)"):
        gradient_i = torch.load(os.path.join(ckpt_folder, gradient_files[i]))
        
        for j in tqdm(range(i + 1, len(gradient_files)), desc=f"Processing gradients (j) for i={i}", leave=False):
            gradient_j = torch.load(os.path.join(ckpt_folder, gradient_files[j]))
            
            for name in gradient_i:
                cos_sim = F.cosine_similarity(gradient_i[name].flatten(), gradient_j[name].flatten(), dim=0)
                distance = 1 - cos_sim.item()
                distance_dict[name][i, j] = distance
                distance_dict[name][j, i] = distance  # Симметричное заполнение
    return distance_dict

def visualize_mds(distance_matrix):
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', label='Gradients')
    
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=9, ha='right')
    
    plt.xlabel("MDS Component 1")
    plt.ylabel("MDS Component 2")
    plt.title("MDS Visualization of Gradient Distances")
    plt.legend()
    plt.grid(True)
    plt.show()

def extract_gradients(ckpt_folder, flattened = True):
    gradient_files = sorted(f for f in os.listdir(ckpt_folder) if f.startswith("gradients_") and f.endswith(".pt"))
    gradients = {}
    
    for i, file in enumerate(tqdm(gradient_files, desc="Loading gradients")):
        gradient = torch.load(os.path.join(ckpt_folder, file))
        
        for name, tensor in gradient.items():
            if name not in gradients:
                gradients[name] = []
            
            gradients[name].append(tensor.flatten().cpu().detach().numpy() if flattened else tensor.cpu().detach().numpy())
            
    return gradients

def visualize_mds_from_gradients(gradient_vectors):
    gradient_matrix = torch.stack(gradient_vectors).cpu().numpy()  # Перемещение на CPU
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
    coords = mds.fit_transform(gradient_matrix)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', label='Gradients')
    
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=9, ha='right')
    
    plt.xlabel("MDS Component 1")
    plt.ylabel("MDS Component 2")
    plt.title("MDS Visualization of Gradient Vectors")
    plt.legend()
    plt.grid(True)
    plt.show()