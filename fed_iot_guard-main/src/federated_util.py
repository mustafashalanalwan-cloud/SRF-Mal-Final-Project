from copy import deepcopy
from types import SimpleNamespace
from typing import List, Set, Tuple, Callable, Optional

import numpy as np
import torch
from context_printer import ContextPrinter as Ctp, Color
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from architectures import NormalizingModel
from ml import set_models_sub_divs


def federated_averaging(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    with torch.no_grad():
        state_dict_mean = global_model.state_dict()
        for key in state_dict_mean:
            state_dict_mean[key] = torch.stack([model.state_dict()[key] for model in models], dim=-1).mean(dim=-1)
        global_model.load_state_dict(state_dict_mean)


# For 8 clients this is equivalent to federated trimmed mean 3
def federated_median(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    n_excluded_down = (len(models) - 1) // 2
    n_included = 2 if (len(models) % 2 == 0) else 1

    with torch.no_grad():
        state_dict_median = global_model.state_dict()
        for key in state_dict_median:
            # It seems that it's much faster to compute the median by manually sorting and narrowing onto the right element
            # rather than using torch.median.
            sorted_tensor, _ = torch.sort(torch.stack([model.state_dict()[key] for model in models], dim=-1), dim=-1)
            trimmed_tensor = torch.narrow(sorted_tensor, -1, n_excluded_down, n_included)
            state_dict_median[key] = trimmed_tensor.mean(dim=-1)
        global_model.load_state_dict(state_dict_median)


def federated_min_max(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    subs = torch.stack([model.sub for model in models])
    sub, _ = torch.min(subs, dim=0)
    divs = torch.stack([model.div for model in models])
    max_values = divs + subs
    max_value, _ = torch.max(max_values, dim=0)
    div = max_value - sub
    global_model.set_sub_div(sub, div)


# Shortcut for __federated_trimmed_mean(global_model, models, 1) so that it's easier to set the aggregation function as a single param
def federated_trimmed_mean_1(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    __federated_trimmed_mean(global_model, models, 1)


# Shortcut for __federated_trimmed_mean(global_model, models, 2) so that it's easier to set the aggregation function as a single param
def federated_trimmed_mean_2(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    __federated_trimmed_mean(global_model, models, 2)


def __federated_trimmed_mean(global_model: torch.nn.Module, models: List[torch.nn.Module], trim_num_up: int) -> None:
    n = len(models)
    n_remaining = n - 2 * trim_num_up

    with torch.no_grad():
        state_dict_result = global_model.state_dict()
        for key in state_dict_result:
            sorted_tensor, _ = torch.sort(torch.stack([model.state_dict()[key] for model in models], dim=-1), dim=-1)
            trimmed_tensor = torch.narrow(sorted_tensor, -1, trim_num_up, n_remaining)
            state_dict_result[key] = trimmed_tensor.mean(dim=-1)
        global_model.load_state_dict(state_dict_result)


# As defined in https://arxiv.org/pdf/2006.09365.pdf
def s_resampling(models: List[torch.nn.Module], s: int) -> Tuple[List[torch.nn.Module], List[List[int]]]:
    T = len(models)
    c = [0 for _ in range(T)]
    output_models = []
    output_indexes = []
    for t in range(T):
        j = [-1 for _ in range(s)]
        for i in range(s):
            while True:
                j[i] = np.random.randint(T)
                if c[j[i]] < s:
                    c[j[i]] += 1
                    break
        output_indexes.append(j)
        with torch.no_grad():
            g_t_bar = deepcopy(models[0])
            sampled_models = [models[j[i]] for i in range(s)]
            federated_averaging(g_t_bar, sampled_models)
            output_models.append(g_t_bar)

    return output_models, output_indexes


def model_update_scaling(global_model: torch.nn.Module, malicious_clients_models: List[torch.nn.Module], factor: float) -> None:
    with torch.no_grad():
        for model in malicious_clients_models:
            new_state_dict = {}
            for key, original_param in global_model.state_dict().items():
                param_delta = model.state_dict()[key] - original_param
                param_delta = param_delta * factor
                new_state_dict.update({key: original_param + param_delta})
            model.load_state_dict(new_state_dict)


def model_canceling_attack(global_model: torch.nn.Module, malicious_clients_models: List[torch.nn.Module], n_honest: int) -> None:
    factor = - n_honest / len(malicious_clients_models)
    with torch.no_grad():
        for normalizing_model in malicious_clients_models:
            new_state_dict = {}
            for key, original_param in global_model.model.state_dict().items():
                new_state_dict.update({key: original_param * factor})
            normalizing_model.model.load_state_dict(new_state_dict)
            # We only change the internal model of the NormalizingModel. That way we do not actually attack the normalization values
            # because they are not supposed to change throughout the training anyway.


def select_mimicked_client(params: SimpleNamespace) -> Optional[int]:
    honest_client_ids = [client_id for client_id in range(len(params.clients_devices)) if client_id not in params.malicious_clients]
    if params.model_poisoning == 'mimic_attack':
        mimicked_client_id = np.random.choice(honest_client_ids)
        Ctp.print('The mimicked client is {}'.format(mimicked_client_id))
    else:
        mimicked_client_id = None
    return mimicked_client_id


# Attack in which all malicious clients mimic the model of a single good client. The mimicked client should always be the same throughout
# the federation rounds.
def mimic_attack(models: List[torch.nn.Module], malicious_clients: Set[int], mimicked_client: int) -> None:
    with torch.no_grad():
        for i in malicious_clients:
            models[i].load_state_dict(models[mimicked_client].state_dict())


def init_federated_models(train_dls: List[DataLoader], params: SimpleNamespace, architecture: Callable):
    # Initialization of a global model
    n_clients = len(params.clients_devices)
    global_model = NormalizingModel(architecture(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                                    sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))

    if params.cuda:
        global_model = global_model.cuda()

    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    if params.normalization == 'min-max':
        federated_min_max(global_model, models)
    else:
        federated_averaging(global_model, models)

    models = [deepcopy(global_model) for _ in range(n_clients)]
    return global_model, models


def model_poisoning(global_model: torch.nn.Module, models: List[torch.nn.Module], params: SimpleNamespace,
                    mimicked_client_id: Optional[int] = None, verbose: bool = False) -> List[torch.nn.Module]:
    malicious_clients_models = [model for client_id, model in enumerate(models) if client_id in params.malicious_clients]
    n_honest = len(models) - len(malicious_clients_models)

    # Model poisoning attacks
    if params.model_poisoning is not None:
        if params.model_poisoning == 'cancel_attack':
            model_canceling_attack(global_model=global_model, malicious_clients_models=malicious_clients_models, n_honest=n_honest)
            if verbose:
                Ctp.print('Performing cancel attack')
        elif params.model_poisoning == 'mimic_attack':
            mimic_attack(models, params.malicious_clients, mimicked_client_id)
            if verbose:
                Ctp.print('Performing mimic attack on client {}'.format(mimicked_client_id))
        else:
            raise ValueError('Wrong value for model_poisoning: ' + str(params.model_poisoning))

    # Rescale the model updates of the malicious clients (if any)
    model_update_scaling(global_model=global_model, malicious_clients_models=malicious_clients_models, factor=params.model_update_factor)
    return models


# Aggregates the model according to params.aggregation_function, potentially using s-resampling, and distributes the global model back to the clients
def model_aggregation(global_model: torch.nn.Module, models: List[torch.nn.Module], params: SimpleNamespace, verbose: bool = False)\
        -> Tuple[torch.nn.Module, List[torch.nn.Module]]:

    if params.resampling is not None:
        models, indexes = s_resampling(models, params.resampling)
        if verbose:
            Ctp.print(indexes)
    params.aggregation_function(global_model, models)

    # Distribute the global model back to each client
    models = [deepcopy(global_model) for _ in range(len(params.clients_devices))]

    return global_model, models

# ══════════════════════════════════════════════════════════════════════════════
# Added by SRF-Mal — Jessmon & Alwan, University of Galway, 2025
# Extending Rey et al. (2022) future work: Krum and Bulyan aggregation
# + gradient noise and sign flip attack simulation
# ══════════════════════════════════════════════════════════════════════════════
 
def federated_krum(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    """
    Krum aggregation — Blanchard et al. (2017).
    Selects the single model update with the smallest sum of squared
    distances to its m nearest neighbours.
    Identified as future work in Rey et al. (2022).
    """
    n = len(models)
    m = max(1, n - 2)
 
    # Flatten each model's parameters into a 1-D vector
    vecs = []
    for model in models:
        params_flat = torch.cat([p.data.view(-1) for p in model.parameters()])
        vecs.append(params_flat)
 
    # Compute pairwise squared distances
    dist = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sum((vecs[i] - vecs[j]) ** 2)
            dist[i, j] = dist[j, i] = d
 
    # Score = sum of m nearest neighbour distances (skip self = 0)
    scores = []
    for i in range(n):
        sorted_dists, _ = torch.sort(dist[i])
        scores.append(sorted_dists[1: m + 1].sum().item())
 
    # Select the model with the lowest score
    best = int(np.argmin(scores))
    global_model.load_state_dict(models[best].state_dict())
 
 
def federated_bulyan(global_model: torch.nn.Module, models: List[torch.nn.Module]) -> None:
    """
    Bulyan aggregation — El Mhamdi et al. (2018).
    Step 1: Multi-Krum selects c most trustworthy models.
    Step 2: Coordinate-wise trimmed mean on those c models.
    Requires n >= 4f + 3 (f = number of Byzantine clients).
    Identified as future work in Rey et al. (2022).
    """
    n = len(models)
    c = max(3, n // 2)
    m = max(1, n - 2)
 
    # Flatten parameters to vectors
    vecs = []
    for model in models:
        params_flat = torch.cat([p.data.view(-1) for p in model.parameters()])
        vecs.append(params_flat)
 
    # Pairwise squared distances
    dist = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sum((vecs[i] - vecs[j]) ** 2)
            dist[i, j] = dist[j, i] = d
 
    # Multi-Krum: select c models with lowest scores
    scores = []
    for i in range(n):
        sorted_dists, _ = torch.sort(dist[i])
        scores.append(sorted_dists[1: m + 1].sum().item())
 
    selected_idx = sorted(range(n), key=lambda i: scores[i])[:c]
    selected_models = [models[i] for i in selected_idx]
 
    # Trimmed mean on selected models
    trim_k = max(0, (c - 2) // 4) if c > 2 else 0
 
    with torch.no_grad():
        state_dict_result = global_model.state_dict()
        for key in state_dict_result:
            stack = torch.sort(
                torch.stack([m.state_dict()[key] for m in selected_models], dim=-1),
                dim=-1
            )[0]
            trimmed = torch.narrow(stack, -1, trim_k, c - 2 * trim_k) \
                if 2 * trim_k < c else stack
            state_dict_result[key] = trimmed.mean(dim=-1)
        global_model.load_state_dict(state_dict_result)
 
 
def gradient_noise_attack(models: List[torch.nn.Module], malicious_clients: Set[int], std: float = 1.0) -> None:
    """
    Gradient Noise Attack — SRF-Mal (2025).
    Adds Gaussian noise N(0, std) to all weights of malicious clients
    after local training. Weight-level attack.
    """
    with torch.no_grad():
        for client_id in malicious_clients:
            for param in models[client_id].parameters():
                param.data += torch.randn_like(param.data) * std
 
 
def sign_flip_attack(models: List[torch.nn.Module], malicious_clients: Set[int]) -> None:
    """
    Sign Flip Attack — SRF-Mal (2025).
    Negates all model weights of malicious clients after local training.
    Weight-level attack — maximally disrupts aggregation.
    """
    with torch.no_grad():
        for client_id in malicious_clients:
            for param in models[client_id].parameters():
                param.data = -param.data
 
 
def label_flipping(y: np.ndarray) -> np.ndarray:
    """
    Label Flip Attack — SRF-Mal (2025).
    Flips binary labels 0->1 and 1->0 before local training.
    Data-level attack — applied in the experiment loop before training.
    """
    return 1.0 - y