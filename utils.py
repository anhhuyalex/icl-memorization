import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import torch
import torchvision
import torchvision.models as models


import numpy as np
import pickle
import functools
import shutil
import io
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    """default dict"""
    def __init__(self, *args, **kwargs):
        if 'default' in kwargs:
            self.default = kwargs['default']
            del kwargs['default']
        else:
            self.default = None
        dict.__init__(self, *args, **kwargs)
        
    def __repr__(self):
        return 'defaultdict(%s, %s)' % (self.default,
                                        dict.__repr__(self))

    def __missing__(self, key):
        if self.default is not None:
            return self.default
        else:
            raise KeyError(key)
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)
    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            return super(DictionaryLike, self).__getattr__(key)
        return self.__getitem__(key)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def try_or(expr, default=None, expected_exc=(Exception,)):
    try:
        return expr
    except expected_exc:
        return default
    
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def freeze_layer(net, *layers):
    for layer_name in layers:
        layer = rgetattr(net, layer_name)
        for param in layer.parameters():
            param.requires_grad = False


################# METRICS ########################
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pearson_corr_lastdim(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Batched Pearson correlation along the last dimension.

    Args:
        x: Tensor with shape (..., L)
        y: Tensor broadcastable to x with shape (..., L)
        eps: Small constant for numerical stability.

    Returns:
        Tensor with shape (...) containing Pearson correlations.
    """
    # Center over last dim
    x0 = x - x.mean(dim=-1, keepdim=True)
    y0 = y - y.mean(dim=-1, keepdim=True)

    # Normalize variance (unbiased=False matches common deep-learning usage)
    x0 = x0 / (x0.std(dim=-1, keepdim=True, unbiased=False) + eps)
    y0 = y0 / (y0.std(dim=-1, keepdim=True, unbiased=False) + eps)

    # Correlation is mean of z-scored products along last dim
    return (x0 * y0).mean(dim=-1)


@torch.no_grad()
def corr_heads_at_ts(
    attn: torch.Tensor,
    step: int = 5,
    ts: torch.Tensor | None = None,
    eps: float = 1e-8,
    return_ts: bool = False,
    fill_nan: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-batch head-head correlation matrices for several values of t using causal prefixes.

    For each t in `ts` (default: arange(0, H, step) clipped to < N), form:
        X_t = attn[:, :, t, :t]   with shape (B, H, t)
    and compute Pearson correlation across heads (treating each head's length-t vector as a sample):
        corr_t[b] = corrcoef_over_lastdim(X_t[b])  -> (H, H)

    Args:
        attn: (B, H, N, N) attention probabilities.
        step: Step size when ts is None.
        ts: Optional 1D LongTensor of t indices. If None, uses torch.arange(0, H, step).
            Note: `t` indexes the sequence dimension N, so we always clip to ts < N.
        eps: Numerical stability for std normalization.
        return_ts: If True, return (corr, ts_valid). Otherwise return corr only.
        fill_nan: If True, fill undefined t<=1 slices with NaN; else fill with 0.

    Returns:
        corr: (B, T, H, H) where T = len(ts_valid)
        ts_valid (optional): (T,) the actually-used t indices (ts clipped to < N)
    """
    if attn.ndim != 4:
        raise ValueError(f"attn must have shape (B,H,N,N); got {tuple(attn.shape)}")
    B, H, N, N2 = attn.shape
    if N != N2:
        raise ValueError(f"attn must be square in last 2 dims; got {(N, N2)}")

    device = attn.device
    if ts is None:
        if step <= 0:
            raise ValueError(f"step must be > 0; got {step}")
        ts = torch.arange(0, N, step, device=device, dtype=torch.long)
    else:
        if not isinstance(ts, torch.Tensor):
            ts = torch.as_tensor(ts, device=device)
        ts = ts.to(device=device, dtype=torch.long)
        if ts.ndim != 1:
            raise ValueError(f"ts must be 1D; got shape {tuple(ts.shape)}")

    ts_valid = ts[ts < N]
    T = int(ts_valid.numel())
    fill = float("nan") if fill_nan else 0.0
    out = attn.new_full((B, T, H, H), fill)

    # Loop over a small number of t values; inside is batched over B and all heads.
    for ti in range(T):
        t = int(ts_valid[ti].item())
        if t <= 1:
            continue

        # X: (B,H,t)
        X = attn[:, :, t, :t]

        # Center/standardize over last dim (t), then compute correlation via dot products.
        X = X - X.mean(dim=-1, keepdim=True)
        X = X / (X.std(dim=-1, keepdim=True, unbiased=False) + eps)
        out[:, ti] = torch.matmul(X, X.transpose(-1, -2)) / X.shape[-1]

    return (out, ts_valid) if return_ts else out


@torch.no_grad()
def corr_attn_prefix_rows(
    attn: torch.Tensor,
    step: int = 5,
    positions: torch.Tensor | None = None,
    eps: float = 1e-8,
    chunk_t: int | None = None,
    return_nan_matrix: bool = True,
) -> torch.Tensor:
    """
    Compute correlations between prefix-normalized attention rows on a coarse (step) grid.

    Given attention `attn` with shape (B, H, N, N), for a set of positions `pos` (default:
    0, step, 2*step, ...), define:

      s_positions = pos                         (i.e., 0, step, 2*step, ...)
      t_positions = pos + 1                     (i.e., 1, step+1, 2*step+1, ...)

    so that `s` and `t` are always distinct (modulo valid range), and compute for each
    pair (s, t) with s in s_positions and t in t_positions and t > s:

      x = attn[..., s, :s] normalized to sum 1
      y = attn[..., t, :s] normalized to sum 1
      corr(s,t) = PearsonCorr(x, y) over the last dimension (length s)

    This matches the common strictly-causal setting where row t attends only to indices < t,
    so the meaningful prefix for row s is :s (excluding the diagonal).

    Args:
        attn: (B, H, N, N) attention probabilities (typically softmaxed).
        step: Step size used when `positions` is None.
        positions: Optional 1D LongTensor of positions in [0, N). If provided, overrides step.
        eps: Numerical stability for renormalization and std.
        chunk_t: Optional chunk size for t-positions to reduce peak memory.
        return_nan_matrix: If True, fill invalid entries (t <= s or s==0) with NaN.
            If False, fill them with 0.

    Returns:
        corr: (B, H, P_s, P_t) where:
            - P_s = len(positions) (s_positions index i)
            - P_t = number of valid t_positions = (positions + 1) < N (t_positions index j)
          Entry (i,j) corresponds to s = positions[i], t = positions[j] + 1.
          Only entries with j >= i and valid indices are populated; others are NaN or 0
          depending on `return_nan_matrix`.
    """
    if attn.ndim != 4:
        raise ValueError(f"attn must have shape (B,H,N,N); got {tuple(attn.shape)}")
    B, H, N, N2 = attn.shape
    if N != N2:
        raise ValueError(f"attn must be square in last 2 dims; got {(N, N2)}")

    device = attn.device
    if positions is None:
        if step <= 0:
            raise ValueError(f"step must be > 0; got {step}")
        positions = torch.arange(0, N, step, device=device, dtype=torch.long)
    else:
        if not isinstance(positions, torch.Tensor):
            positions = torch.as_tensor(positions, device=device)
        positions = positions.to(device=device, dtype=torch.long)
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1D; got shape {tuple(positions.shape)}")

    P = int(positions.numel())
    fill = float("nan") if return_nan_matrix else 0.0

    # Precompute valid t positions: t_positions = positions + 1, but must satisfy t < N.
    # Since positions is increasing, validity is a prefix.
    t_positions = positions + 1
    P_t = int((t_positions < N).sum().item())
    t_positions = t_positions[:P_t]

    out = attn.new_full((B, H, P, P_t), fill)

    # One Python loop over s; inside is fully batched over B,H and all valid t (shifted by +1).
    for i in range(P):
        if i >= P_t:
            break  # no columns remain with j >= i
        s = int(positions[i].item())
        if s <= 0:
            continue
        if s >= N:
            continue

        # x: (B,H,s) -> normalize to sum 1 -> (B,H,1,s)
        x = attn[:, :, s, :s]
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
        x = x.unsqueeze(2)  # (B,H,1,s)

        # Columns correspond to t_positions = positions + 1, and we only fill j >= i.
        tpos = t_positions[i:]  # (T,)
        if tpos.numel() == 0:
            continue
        valid_len = int(tpos.numel())

        # Optionally chunk over t to cap memory.
        if chunk_t is None or int(chunk_t) <= 0:
            y = attn[:, :, tpos, :s]  # (B,H,T,s)
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
            corr = pearson_corr_lastdim(x, y, eps=eps)  # (B,H,T)
            out[:, :, i, i : (i + valid_len)] = corr
        else:
            T = int(tpos.numel())
            for j0 in range(0, T, int(chunk_t)):
                j1 = min(T, j0 + int(chunk_t))
                tpos_chunk = tpos[j0:j1]
                y = attn[:, :, tpos_chunk, :s]  # (B,H,Tc,s)
                y = y / (y.sum(dim=-1, keepdim=True) + eps)
                corr = pearson_corr_lastdim(x, y, eps=eps)  # (B,H,Tc)
                out[:, :, i, (i + j0) : (i + j1)] = corr

    return out
    
################# MODELS ########################


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation, keep_rate=1.0, N_CLASSES = 10, batch_norm = False,
                randomfeatures = False):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
       
        # Output layer
        if self.hidden_sizes is not None:
            self.out = torch.nn.Linear(self.hidden_sizes[-1], N_CLASSES)
             # Set up perceptron layers and add dropout
            self.fc1 = torch.nn.Linear(self.input_size,
                                       self.hidden_sizes[0])
            
            if randomfeatures == True: 
                assert len(hidden_sizes) == 1, f"Random features require a 2-layer network, get a {len(hidden_sizes) + 1}-layer network"
                self.fc1.requires_grad_(False)
              
            self.dropout = torch.nn.Dropout(1 - keep_rate)
            self.hidden = nn.ModuleList()
            for k in range(len(self.hidden_sizes) - 1):
                self.hidden.append(nn.Linear(self.hidden_sizes[k], self.hidden_sizes[k+1]))
             
            # Batch norm
            self.batch_norm_layers = nn.ModuleList()
            self.batch_norm = batch_norm
            if batch_norm == True:
                for k in range(len(self.hidden_sizes)):
                    self.batch_norm_layers.append(nn.BatchNorm1d(self.hidden_sizes[k]))

        else:
            self.out = torch.nn.Linear(self.input_size, N_CLASSES)
     
    def forward(self, x):
        x = x.reshape(x.size(0), -1) # flatten if needed
        if self.hidden_sizes is not None:
            if self.activation == "tanh":
                x = self.fc1(x)
                Tanh = torch.nn.Tanh()
                if self.batch_norm == True:
                    x = self.batch_norm_layers[0](x)
                x = Tanh(x)
            elif self.activation == "relu":
                x = self.fc1(x)
                if self.batch_norm == True:
                    x = self.batch_norm_layers[0](x)
                x = torch.nn.functional.relu(x)
            elif self.activation == "line":
                x = self.fc1(x)
                if self.batch_norm == True:
                    x = self.batch_norm_layers[0](x)
                

            x = self.dropout(x)
            for k, lay in enumerate(self.hidden):
                x = lay(x)
                if self.batch_norm == True:
                    x = self.batch_norm_layers[k+1](x)
                if self.activation == "tanh":
                    x = Tanh(x)
                elif self.activation == "relu":
                    x = torch.nn.functional.relu(x)
                
                x = self.dropout(x)
            
        # Else just return the linear transformation
        return (self.out(x))
    
class ResNet(MLP):
    def forward(self, x):
        x = x.view(-1, self.input_size)
        if self.activation == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        for lay in self.hidden:
            if self.activation == "sigmoid":
                x = x + sigmoid(lay(x))
            elif self.activation == "relu":
                x = x + torch.nn.functional.relu(lay(x))
            x = self.dropout(x)
        return (self.out(x))
    
class CNN(nn.Module):
    def __init__(self, conv1_chans = 6, conv2_chans = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_chans, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_chans, conv2_chans, 5)
        self.fc1 = nn.Linear(conv2_chans * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, n_classes):
        """ Alexnet Classifier
        Borrowed from https://github.com/ayshrv/cs7641-img-classification/blob/master/models/alexnet.py
        Arguments:
            n_classes (int): Number of classes to score
        """
        super(AlexNet, self).__init__()

        self.alexnet = models.alexnet(pretrained=False, num_classes=n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        """
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width
        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        unnormalised_scores = self.alexnet(images)
        #logits = self.softmax(unnormalised_scores)
        return unnormalised_scores
    
class VGG(nn.Module):
    def __init__(self, model_name, n_classes):
        """ VGG11 Classifier
        Arguments:
            n_classes (int): Number of classes to score
        """
        super(VGG, self).__init__()
        if model_name == "vgg11":
            self.VGG = models.vgg11(pretrained=False, num_classes=n_classes)

    def forward(self, images):
        """
        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width
        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        """
        unnormalised_scores = self.VGG(images)
        return unnormalised_scores

class Ensemble(nn.Module):
    def __init__(self, num_models, model_func, model_kwargs = {}, voting_strategy = "average_logits"):
        
        super(Ensemble, self).__init__()
        
        self.models = nn.ModuleList([model_func(**model_kwargs) for _ in range(num_models)])
        self.voting_strategy = voting_strategy 
    def forward(self, x):
        submodel_scores = [f(x) for f in self.models]
        if self.voting_strategy == "average_logits":
            unnormalised_scores = torch.mean(torch.stack(submodel_scores), dim=0)
            return unnormalised_scores, submodel_scores
        elif self.voting_strategy == "average_softmax":
            softmaxes = F.softmax(torch.stack(submodel_scores), dim=2)
            average_softmax =  torch.mean(softmaxes, dim=0)
            return average_softmax, submodel_scores
        
class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True,):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1,1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim = 1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X.T @ X 
        rhs = X.T @ y
        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0], device=X.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution
            
    def predict(self, X: torch.tensor) -> None:
        # print ("w", self.w.shape)
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim = 1)
        return X @ self.w
    
models_registry = dict(
    MLP = MLP,
    ResNet = ResNet,
    CNN = CNN,
    AlexNet = AlexNet,
    VGG = VGG,
    Ensemble = Ensemble
)
########### DATASETS ##################################################
class ShuffledCIFAR10:
    def __init__(self, *, 
                 pixel_shuffled = None, image_width = IMAGE_WIDTH, train = True, download=False, transform = None,
                    fix_permutation = None):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                            download=download, transform=transform)
        self.pixel_shuffled = pixel_shuffled
        if pixel_shuffled == True:
            self.image_width = image_width
            self.perm = torch.randperm(image_width * image_width)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        inputs, labels = self.dataset[idx]
        if self.pixel_shuffled == True:
            inputs = inputs.view(IMAGE_CHANNELS, self.image_width * self.image_width)
            inputs = inputs[:, self.perm].view(IMAGE_CHANNELS, self.image_width, self.image_width)
        return inputs, labels
    
########### TRAINERS ##################################################
class BaseTrainer:
    def __init__(self, data_params = dict(),
                       train_params = dict(),
                       model_optim_params = dict(),
                       save_params = dict()):
        
        self.data_params = data_params
        self.train_params = train_params
        self.model_optim_params = model_optim_params
        self.save_params = save_params
        self.setup()
        
    def setup(self):
        self.build_data_loader()
        self.build_model_optimizer()
        self.initialize_record()
        
    def save_record(self):
        # Save record
        fname = f"{self.save_params['save_dir']}/{self.save_params['exp_name']}"
        save_file_pickle(fname, self.record)
        
class CIFAR_trainer(BaseTrainer):
    def build_data_loader(self):
        batch_size = self.train_params['batch_size']
        
        trainset = ShuffledCIFAR10(train=True, **self.data_params) 
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

        testset = ShuffledCIFAR10(train=False, **self.data_params) 
        
        if self.data_params["pixel_shuffled"] == True:
            if self.data_params["fix_permutation"] == True:
                try:
                    trainset.perm = torch.load(f"{self.save_params['save_dir']}/{self.model_optim_params['model_name']}_quenched_permutation.pt")
                except:
                    torch.save(trainset.perm, f"{self.save_params['save_dir']}/{self.model_optim_params['model_name']}_quenched_permutation.pt")
            testset.perm = trainset.perm

        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    def build_model_optimizer(self):
        self.model = self.model_optim_params["model"]
        self.criterion = self.model_optim_params["criterion"]
        self.optimizer = self.model_optim_params["optimizer"]
        
    def initialize_record(self):
        self.record = dict(
            metrics = dict(
                train_loss_prog = [],
                train_acc_prog = [],
                test_loss_prog = [],
                test_acc_prog = [],
                test_loss = 0.0,
                test_accuracy = 0.0
            ),    
            print_interval = 5000 // self.train_params['batch_size'],
            data_params = self.data_params,
            train_params = self.train_params,
            model_optim_params = self.model_optim_params,
            save_params = self.save_params,
            model = None
        )
        
    def before_train_epoch(self):
        # after a certain epoch, freeze convolutional layers if desired
        if self.epoch > self.train_params["freeze_params"]["epoch"]:
            freeze_layer(self.model, *self.train_params["freeze_params"]["epoch"])
            for n, param in self.model.named_parameters():         
                print(n, param.requires_grad)
    def after_train_iter(self, running_loss, running_accuracy, outputs, labels):
        self.loss.backward()
        self.optimizer.step()
        
        # get top-1 accuracy
        pred_class = torch.argmax(outputs, dim=1)
        running_accuracy += torch.mean((pred_class == labels).float()).item()

        # print statistics
        running_loss += self.loss.item()
        if self.iter % self.record["print_interval"] == self.record["print_interval"] - 1:    # print every self.record["print_interval"] mini-batches
            running_loss /= self.record["print_interval"]
            running_accuracy /= self.record["print_interval"]
            self.record["metrics"]["train_loss_prog"].append(running_loss)
            self.record["metrics"]["train_acc_prog"].append(running_accuracy)
            print(f'[{self.epoch + 1}, {self.iter + 1:5d}] loss: {running_loss:.3f}, accuracy:  {running_accuracy:.3f}')
            running_loss = 0.0
            running_accuracy = 0.0
        return running_loss, running_accuracy
    
    def after_train_epoch(self):
        # get test loss and accuracy
        test_loss, test_accuracy = self.test()
        self.record["metrics"]["test_loss_prog"].append(test_loss)
        self.record["metrics"]["test_acc_prog"].append(test_accuracy)
        
        # put model back into train mode
        self.model.train()
        
        
    def train(self):
        num_epochs = self.train_params['num_epochs']
        for self.epoch in range(num_epochs):  # loop over the dataset multiple times
            self.before_train_epoch()
            running_loss, running_accuracy = 0.0, 0.0
            for self.iter, data in enumerate(self.trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                self.loss = self.criterion(outputs, labels)
                
                running_loss, running_accuracy = self.after_train_iter(running_loss, running_accuracy, outputs, labels)

            self.after_train_epoch()

        print('Finished Training')
        self.after_run()
        
    def test(self):
        running_loss, running_accuracy = 0.0, 0.0
        self.model.eval()
        for i, data in enumerate(self.testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # forward and evaluate
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()
            
            # get top-1 accuracy
            pred_class = torch.argmax(outputs, dim=1)
            running_accuracy += torch.mean((pred_class == labels).float()).item()

        # print loss
        running_loss /= len(self.testloader)
        running_accuracy /= len(self.testloader)
        print(f'Test loss: {running_loss:.3f}, accuracy: {running_accuracy:.3f}')
        return running_loss, running_accuracy
    
    def after_run(self):
        # Get final test loss and accuracy
        test_loss, test_accuracy = self.test()
        self.record["metrics"]["test_loss"] = test_loss
        self.record["metrics"]["test_accuracy"] = test_accuracy
        
        # Save model
        self.record["model"] = list(self.model.named_parameters())
        
        # Save record
        self.save_record()

        

################# METERS ########################
from enum import Enum
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries),flush=True)
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



################# SAVING FILES ########################
def save_file_pickle(fname, file):
    with open(f"{fname}.pkl", 'wb') as f:
        pickle.dump(file, f)
        
def save_checkpoint(state, save_dir = "./", filename='checkpoint.pth.tar'):
    torch.save(state, f"{save_dir}/{filename}")
    
                               
def load_file_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
                               
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])
                               
                               
# from scipy import interpolate
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from matplotlib.colors import LogNorm


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def show_plt_if_in_notebook(title=None):
    if in_notebook():
        plt.show()
    else:
        plt.savefig(title)

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def sci_tex(x):
    exp = int(np.floor(np.log10(x)))
    mant = x / (10 ** exp)
    return rf"${mant:.1f}\times 10^{{{exp}}}$"

def set_plot_style():
    sns.set(
        context="talk",
        font_scale=1.0,
        color_codes=True,
        palette="deep",
        style="ticks",
        rc={
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.5,
            "figure.dpi": 70,
            "text.usetex": True,
            "font.size": 28,
            "legend.fontsize": 14,
            "legend.title_fontsize": 16,
        },
    )

def add_colorbar(fig, ax, norm, cmap="rocket", label=None, log_ticks=False):
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    if label is not None:
        cbar.set_label(label)
    if log_ticks:
        cbar.formatter = FuncFormatter(
            lambda x, _pos: sci_tex(x) if x > 0 else "0"
        )
        cbar.update_ticks()
    return cbar
