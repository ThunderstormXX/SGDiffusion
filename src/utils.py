import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pickle
import os 

import torch.nn.functional as F


class Logger:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self._ticks = []
        self._tick_value = 0
        self._tick_step = 1
        self.dict = dict()

    def _synchronize(self):
        """Синхронизация устройства перед измерением времени."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

    def tick(self,):
        """Метод для шага обновления данных"""
        self._synchronize()
        self._tick_value += 1
        if self._tick_value % self._tick_step == 0:
            self._ticks.append(self._tick_value)
            return True
        return False      

    def set_tick_step(self, step: int):
        self._tick_step = step        


    def update(self, name: str, value, tiks = True):
        if not self.tick() and tiks:
            return None

        self._synchronize()
        if name not in self.dict:
            self.dict[name] = [value]
        else:
            self.dict[name].append(value)

    # Если хочу залогировать сразу несколько, но тик должен выполниться однажды
    def updates(self, names, values):
        if not self.tick():
            return None

        self._synchronize()
        for name, value in zip(names, values):
            self.update(name, value, tiks = False)
            

    def save(self, filename):
        """Метод для сохранения логов в файл."""
        with open(filename, "wb") as f:
            pickle.dump(self.dict, f)

    def load(self, filename):
        """Метод для загрузки логов из файла."""
        with open(filename, "rb") as f:
            self.dict = pickle.load(f)

LOGGER = Logger()


def MNIST(batch_size = 64, sample_size = None):
    # Загрузка и подготовка данных
    transform = transforms.Compose([
        # transforms.Resize(32),  # Увеличиваем размер изображений до 32x32
        # transforms.Grayscale(3),  # Преобразуем в 3 канала
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))  # Нормализация для одного канала, дублируется для всех трех
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Если указано sample_size, то берем подвыборку из train_dataset
    if sample_size is not None:
        indices = torch.randperm(len(train_dataset)).tolist()[:sample_size]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader


def load_similar_mnist_data(batch_size, sample_size=1000):
    """Загружает данные Similar MNIST из файлов для обучения и классический MNIST для теста"""
    from torchvision import transforms, datasets
    from torch.utils.data import Subset
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Загружаем похожие данные для обучения
    x_data = torch.load(os.path.join(data_dir, "mnist_similar_x.pt"))
    y_data = torch.load(os.path.join(data_dir, "mnist_similar_y.pt"))
    
    # Балансировка классов
    samples_per_class = sample_size // 10
    balanced_indices = []
    for class_id in range(10):
        class_indices = (y_data == class_id).nonzero(as_tuple=True)[0]
        if len(class_indices) >= samples_per_class:
            selected = class_indices[torch.randperm(len(class_indices))[:samples_per_class]]
        else:
            selected = class_indices
        balanced_indices.extend(selected.tolist())
    
    balanced_indices = torch.tensor(balanced_indices)
    x_data = x_data[balanced_indices]
    y_data = y_data[balanced_indices]
    
    # Создаем обучающий датасет из похожих данных
    train_dataset = torch.utils.data.TensorDataset(x_data, y_data)
    
    # Загружаем классический MNIST для теста
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Создаем загрузчики
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader

def CIFAR(batch_size=64, sample_size=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    if sample_size is not None:
        indices = torch.randperm(len(train_dataset)).tolist()[:sample_size]
        train_dataset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')


def compute_full_gradient(model, data_loader, criterion, device, prepare=None):
    """
    Вычисляет полный градиент по всему датасету до обновления параметров.
    """
    model.zero_grad()
    total_loss = 0.0
    total_samples = 0

    for inputs, labels in data_loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Только если размерность меток больше 1, применяем squeeze
        if labels.dim() > 1:
            labels = torch.squeeze(labels)
        
        if prepare is not None:
            inputs = prepare(inputs, labels)
        
        output = model(inputs)
        loss = criterion(output, labels.long())
        total_loss += loss.item() * len(labels)
        total_samples += len(labels)
        loss.backward()
    
    full_gradient = {name: param.grad.clone().detach() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad()  # Сбрасываем градиенты после вычисления
    return full_gradient, total_loss / total_samples


def train_epoch(model, optimizer, train_loader, criterion, device, prepare=None, regularizer=None, logging=False, gradient_logging=False):
    model.train()
    for batch in tqdm(enumerate(train_loader)):
        it, traindata = batch 
        train_inputs, train_labels = traindata
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)
        if train_labels.dim() > 1:
            train_labels = torch.squeeze(train_labels)

        # print(train_labels.shape)
        if gradient_logging:
            full_gradient, _ = compute_full_gradient(model, train_loader, criterion, device, prepare)
        
        model.zero_grad()        
        if prepare is not None:
            train_inputs = prepare(train_inputs, train_labels)

        output = model(train_inputs)
        if regularizer is not None: 
            loss = criterion(output, train_labels.long()) + regularizer(model)
        else:
            loss = criterion(output, train_labels.long())
        
        
        if logging:
            names = []
            values = []
            for name, param in model.named_parameters():
                names.append(name)
                values.append(param.clone().detach().cpu().numpy())
            names.append('loss')
            values.append(loss.clone().detach().cpu().numpy())
            LOGGER.updates(names, values)
        
        loss.backward()

        if gradient_logging:
            stochastic_gradients = {name: param.grad.clone().detach() for name, param in model.named_parameters() if param.grad is not None}

            names = ['full_' + name for name in full_gradient.keys()] + ['stochastic_' + name for name in stochastic_gradients.keys()]
            values = list(full_gradient.values()) + list(stochastic_gradients.values())
            LOGGER.updates(names, values)

        optimizer.step()

def evaluate_loss_acc(loader, model, criterion, device, regularizer = None, logging = False):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    if regularizer is not None:
        total_reg_loss = 0
    total = 0.0
    for it, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device) 
        labels = labels.to(device)
        if labels.dim() > 1:
            labels = torch.squeeze(labels)

        output = model(inputs) # pay attention here!
        loss = criterion(output, labels.long())
        total_loss += loss.item()

        if regularizer is not None:
            reg = regularizer(model)
            loss += reg
            total_reg_loss += reg.item()
            
        pred = output.argmax(dim=1)
        correct = pred == labels.byte()
        total_acc += torch.sum(correct).item() / len(correct)

    total = it + 1
    if regularizer is not None:
        return total_loss / total, total_acc / total, total_reg_loss / total
    return total_loss / total, total_acc / total

def train(model, opt, train_loader, test_loader, criterion, n_epochs, \
          device, verbose=True, prepare = None, regularizer=None, logging = False, gradient_logging = False , logging_tick_step = 15):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    if logging or gradient_logging:
        LOGGER.reset()
        LOGGER.set_tick_step(step = logging_tick_step)

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device, prepare = prepare, regularizer=regularizer, logging = logging, gradient_logging = gradient_logging)
        if regularizer is not None:
            train_loss, train_acc, train_reg_loss = evaluate_loss_acc(train_loader,
                                                  model, criterion,
                                                  device, regularizer = regularizer)
            val_loss, val_acc, val_reg_loss = evaluate_loss_acc(test_loader, model,
                                                criterion, device, regularizer = regularizer)
        else:
            train_loss, train_acc = evaluate_loss_acc(train_loader,
                                                    model, criterion,
                                                    device, regularizer = regularizer)
            val_loss, val_acc = evaluate_loss_acc(test_loader, model,
                                                criterion, device, regularizer = regularizer)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)

        if verbose:
            if regularizer is not None:
                    print (('Epoch [%d/%d], Loss (train/test): %.4f/%.4f,'+\
                        ' Acc (train/test): %.4f/%.4f, Reg loss (train/test): %.4f/%.4f' )
                        %(epoch+1, n_epochs, \
                        train_loss, val_loss, train_acc, val_acc, train_reg_loss, val_reg_loss))
            else:
                print (('Epoch [%d/%d], Loss (train/test): %.4f/%.4f,'+\
                ' Acc (train/test): %.4f/%.4f' )
                    %(epoch+1, n_epochs, \
                        train_loss, val_loss, train_acc, val_acc))

    return train_log, train_acc_log, val_log, val_acc_log


def total_variation_loss(weights):
    """
    TV loss для весов нейросети. Подходит для 2D тензоров (например, матриц весов линейных слоёв).
    """
    diff_x = torch.abs(weights[:, :-1] - weights[:, 1:])
    diff_y = torch.abs(weights[:-1, :] - weights[1:, :])
    return torch.sum(diff_x) + torch.sum(diff_y)


def total_variation_loss_model(model):
    loss = 0
    for param in model.parameters():
        if len(param.shape) == 2:  # Только для матриц весов
            loss += total_variation_loss(param)
    return loss

