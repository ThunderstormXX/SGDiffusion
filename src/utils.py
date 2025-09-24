import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import pickle
import os 
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from torch.utils.data import DataLoader, TensorDataset

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


def MNIST(batch_size=64, sample_size=None):
    # Загрузка и подготовка данных
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Если указан sample_size, берем первые элементы
    if sample_size is not None:
        train_dataset = Subset(train_dataset, range(sample_size))
        test_dataset = Subset(test_dataset, range(sample_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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
    
    # Балансировка классов (берём первые samples_per_class для каждого класса)
    samples_per_class = sample_size // 10
    balanced_indices = []
    for class_id in range(10):
        class_indices = (y_data == class_id).nonzero(as_tuple=True)[0]
        selected = class_indices[:samples_per_class]  # ← без случайности
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



def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# random batches (with replacement), all batches exactly batch_size
def load_data_with_replacement(dataset_train: str, batch_size: int, sample_size: int, seed: int):
    if dataset_train == "mnist":
        train_dataset, test_dataset, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, test_dataset, _, _ = load_similar_mnist_data(
            batch_size=batch_size, sample_size=sample_size or 1000
        )

    N = len(train_dataset)
    eff = (N // batch_size) * batch_size  # multiple of batch_size
    g = torch.Generator().manual_seed(seed)

    sampler = RandomSampler(train_dataset, replacement=True, num_samples=eff, generator=g)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )
    return train_dataset, test_dataset, train_loader, val_loader

def load_data(dataset_train: str, batch_size: int, sample_size: int, seed: int):
    if dataset_train == "mnist":
        train_dataset, test_dataset, _, _ = MNIST(batch_size=batch_size, sample_size=sample_size)
    else:
        train_dataset, test_dataset, _, _ = load_similar_mnist_data(
            batch_size=batch_size, sample_size=sample_size or 1000
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=seed_worker
    )

    return train_dataset, test_dataset, train_loader, val_loader

def load_saved_data(train_path: str, test_path: str, batch_size: int, shuffle_train: bool = True, replacement: bool = False, seed: int = 42 ):
    """Загружает сохранённые (X, y) тензоры и возвращает датасеты + лоадеры"""
    X_train, y_train = torch.load(train_path, weights_only=False)
    X_test, y_test = torch.load(test_path, weights_only=False)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    if replacement:
        N = len(train_ds)
        eff = (N // batch_size) * batch_size  # multiple of batch_size
        if eff != N:
            raise Exception('Выборка кратна должна юыть батчу иначе не сойдется с GD')
        
        g = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(train_ds, replacement=True, num_samples=eff, generator=g)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler
        )

        val_loader = DataLoader(
            test_ds,
            batch_size=512,
            shuffle=False,
        )
        return train_ds, test_ds, train_loader, val_loader

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_ds, test_ds, train_loader, test_loader


def load_shakespeare_data(train_path, val_path, batch_size, replacement=False, seed=42):
    """Load Shakespeare dataset with optional replacement sampling"""
    X_train, Y_train = torch.load(train_path)
    X_val, Y_val = torch.load(val_path)
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    if replacement:
        N = len(train_dataset)
        eff = (N // batch_size) * batch_size  # multiple of batch_size
        if eff != N:
            raise Exception('Выборка кратна должна быть батчу иначе не сойдется с GD')
        
        g = torch.Generator().manual_seed(seed)
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=eff, generator=g)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            worker_init_fn=seed_worker, 
            num_workers=0
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                 worker_init_fn=seed_worker, num_workers=0)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           worker_init_fn=seed_worker, num_workers=0)
    
    return train_dataset, val_dataset, train_loader, val_loader
