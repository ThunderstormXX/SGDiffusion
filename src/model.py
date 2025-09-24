import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleMLP(nn.Module):
    def __init__(
        self,
        input_dim=28 * 28,
        output_dim=10,
        hidden_dim=512,
        num_hidden_layers=2,
        dropout_list=None,
        use_relu_list=None,
        input_downsample=None  # <-- новое: сторона квадрата после понижения, напр. 7 → 7x7=49
    ):
        super(FlexibleMLP, self).__init__()

        self.input_downsample = input_downsample

        if self.input_downsample is not None:
            self.downsampler = nn.AdaptiveAvgPool2d((input_downsample, input_downsample))
            input_dim = input_downsample * input_downsample
        else:
            self.downsampler = None
            input_dim = input_dim  # = 784 по умолчанию

        if dropout_list is None:
            dropout_list = [0.0] * num_hidden_layers
        if use_relu_list is None:
            use_relu_list = [True] * num_hidden_layers

        assert len(dropout_list) == num_hidden_layers, "dropout_list must match num_hidden_layers"
        assert len(use_relu_list) == num_hidden_layers, "use_relu_list must match num_hidden_layers"

        layers = []
        in_dim = input_dim

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_relu_list[i]:
                layers.append(nn.SiLU())
            if dropout_list[i] > 0:
                layers.append(nn.Dropout(dropout_list[i]))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))  # выходной слой без активации
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.downsampler is not None:
            x = self.downsampler(x)  # [B, 1, 28, 28] → [B, 1, d, d]
        x = x.view(x.size(0), -1)    # flatten
        return self.model(x)

class FlexibleCNN(nn.Module):
    def __init__(
        self,
        # --- ввод ---
        in_channels: int = 1,
        input_downsample: int | None = None,  # если задано, сначала сожмём картинку до d x d
        # --- свёрточная часть ---
        conv_channels: list[int] = [32, 64],  # число каналов после каждого Conv
        conv_kernels: list[int] | None = None,  # ядра (если None → все 3)
        conv_strides: list[int] | None = None,  # шаги (если None → все 1)
        conv_use_relu_list: list[bool] | None = None,  # активация после conv-блока
        conv_dropouts: list[float] | None = None,  # dropout после активации
        conv_use_bn: bool = True,  # BatchNorm2d после Conv
        pool_after: list[bool] | None = None,  # ставить ли MaxPool2d(2) после блока
        # --- neck / агрегирование ---
        gap_size: int = 1,  # AdaptiveAvgPool2d к (gap_size x gap_size), 1 = global avg pool
        # --- MLP (классификатор) ---
        mlp_hidden_dim: int = 256,           # если mlp_hidden_dims=None → берём это
        mlp_num_layers: int = 1,
        mlp_hidden_dims: list[int] | None = None,  # переопределяет mlp_hidden_dim/mlp_num_layers
        mlp_use_relu_list: list[bool] | None = None,
        mlp_dropouts: list[float] | None = None,
        # --- выход ---
        output_dim: int = 10,
    ):
        super().__init__()

        # ---------- настройки по умолчанию для conv ----------
        L = len(conv_channels)
        if conv_kernels is None:      conv_kernels = [3] * L
        if conv_strides is None:      conv_strides = [1] * L
        if conv_use_relu_list is None: conv_use_relu_list = [True] * L
        if conv_dropouts is None:     conv_dropouts = [0.0] * L
        if pool_after is None:        pool_after = [False] * L

        assert len(conv_kernels) == L
        assert len(conv_strides) == L
        assert len(conv_use_relu_list) == L
        assert len(conv_dropouts) == L
        assert len(pool_after) == L

        # ---------- настройки по умолчанию для mlp ----------
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [mlp_hidden_dim] * mlp_num_layers
        M = len(mlp_hidden_dims)
        if mlp_use_relu_list is None: mlp_use_relu_list = [True] * M
        if mlp_dropouts is None:      mlp_dropouts = [0.0] * M

        assert len(mlp_use_relu_list) == M
        assert len(mlp_dropouts) == M

        # ---------- препроцессинг входа ----------
        self.downsampler = (
            nn.AdaptiveAvgPool2d((input_downsample, input_downsample))
            if input_downsample is not None else None
        )

        # ---------- свёрточные блоки ----------
        conv_layers = []
        c_in = in_channels
        for i in range(L):
            k = conv_kernels[i]
            s = conv_strides[i]
            # «same»-подобная паддинга для нечётных k
            pad = k // 2

            conv_layers.append(nn.Conv2d(c_in, conv_channels[i], kernel_size=k, stride=s, padding=pad, bias=not conv_use_bn))
            if conv_use_bn:
                conv_layers.append(nn.BatchNorm2d(conv_channels[i]))
            if conv_use_relu_list[i]:
                conv_layers.append(nn.SiLU())
            if conv_dropouts[i] > 0:
                conv_layers.append(nn.Dropout2d(conv_dropouts[i]))
            if pool_after[i]:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            c_in = conv_channels[i]

        self.conv = nn.Sequential(*conv_layers)

        # ---------- neck: адаптивный пул к фикс. размеру ----------
        self.neck = nn.AdaptiveAvgPool2d((gap_size, gap_size)) if gap_size is not None else nn.Identity()

        # ---------- MLP классификатор ----------
        mlp_layers = []
        in_dim = c_in * (gap_size if gap_size is not None else 1) ** 2

        for j in range(M):
            mlp_layers.append(nn.Linear(in_dim, mlp_hidden_dims[j]))
            if mlp_use_relu_list[j]:
                mlp_layers.append(nn.SiLU())
            if mlp_dropouts[j] > 0:
                mlp_layers.append(nn.Dropout(mlp_dropouts[j]))
            in_dim = mlp_hidden_dims[j]

        mlp_layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        if self.downsampler is not None:
            x = self.downsampler(x)
        x = self.conv(x)         # [B, C*, H*, W*]
        x = self.neck(x)         # [B, C*, gap, gap]  (gap=1 по умолчанию)
        x = x.flatten(1)         # [B, C* * gap * gap]
        return self.mlp(x)

class CNN(nn.Module):
    def __init__(self, k=1):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)  #;print(x.shape)
        x = self.layer2(x) #;print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x) #;print(x.shape)
        x = self.relu(x) #;print(x.shape)
        x = self.fc1(x) #;print(x.shape)
        x = self.relu1(x) #;print(x.shape)
        x = self.fc2(x) #;print(x.shape)
        return x
    
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn64(self.conv1(x))))
        x = self.pool(F.relu(self.bn128(self.conv2(x))))
        x = F.relu(self.bn256(self.conv3(x)))
        x = self.pool(F.relu(self.bn256(self.conv4(x))))
        x = F.relu(self.bn512(self.conv5(x)))
        x = self.pool(F.relu(self.bn512(self.conv6(x))))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size = 28*28, num_classes = 10, hidden_dim = 32, num_layers = 2):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.network(x)


class CNNLayerNorm(nn.Module):
    def __init__(self, k=1):
        super(CNNLayerNorm, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([6, 28, 28]),  # LayerNorm после свертки
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([6, 14, 14])  # LayerNorm после MaxPool2d
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.LayerNorm([16, 10, 10]),  # LayerNorm после свертки
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([16, 5, 5])  # LayerNorm после MaxPool2d
        )
        self.fc = nn.Linear(400, 120)
        self.ln1 = nn.LayerNorm(120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.ln2 = nn.LayerNorm(84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        self.ln3 = nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.ln2(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.ln3(x)
        return x
    
# Определение модели
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(2, 1, bias=True)
        
    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)


class CausalSelfAttention(nn.Module):
    """Simplified causal self-attention for NanoGPT"""
    
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Key, Query, Value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Causal mask
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        
        # Output projection
        y = self.c_proj(y)
        return y


class TransformerMLP(nn.Module):
    """Simple MLP for transformer block"""
    
    def __init__(self, n_embd, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(mlp_ratio * n_embd)
        self.c_fc = nn.Linear(n_embd, hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, n_embd, bias=False)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block"""
    
    def __init__(self, n_embd, n_head, block_size, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = TransformerMLP(n_embd, mlp_ratio)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class NanoGPT(nn.Module):
    """
    Simplified GPT model for experiments
    Target ~1000 parameters with:
    n_layer=1, n_head=1, n_embd=8, vocab_size=25, block_size=16, mlp_ratio=1
    """
    
    def __init__(self, vocab_size=25, n_embd=8, n_head=1, n_layer=1, 
                 block_size=16, mlp_ratio=1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        
        # Token and position embeddings
        self.wte = nn.Embedding(vocab_size, n_embd)  # token embeddings
        self.wpe = nn.Embedding(block_size, n_embd)  # position embeddings
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, mlp_ratio) 
            for _ in range(n_layer)
        ])
        
        # Final layer norm and language modeling head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm and projection to vocab
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten for cross entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens"""
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
