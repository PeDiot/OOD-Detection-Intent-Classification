from collections import defaultdict
from typing import List, Dict, Any, Optional, Callable

from tqdm import tqdm

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from torch import Tensor

from Todd import ScorerType

def prepare_detectors(
        detectors: List[ScorerType], model, loader: DataLoader, tokenizer
) -> List[ScorerType]:
    """
    Fit the detectors on the behavior of the model on the (in) validation set
    :param detectors: List of detectors to fit
    :param model: Model to evaluate
    :param loader: Dataloader (reference set) to evaluate the model on
    :return: List of fitted detectors
    """

    loop = tqdm(loader, position=0, leave=True)
    loop.set_description("Collecting hidden states...")

    for batch in loop:

        inputs = tokenizer(
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model(
            inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True
        )
        output['encoder_hidden_states'] = output['hidden_states']

        for detector in detectors:
            detector.accumulate(output)

    loop = tqdm(detectors, position=0, leave=True)
    loop.set_description("Fitting detectors...")

    for detector in loop:
        detector.fit()

    return detectors

def flatten_dict(d):
    """
    Flatten a dictionary
    :param d: Dictionary to flatten
    :return: Flattened dictionary
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update({f"{key}+{k}": v for k, v in flatten_dict(value)})
        else:
            result[key] = value

    return result


def evaluate_batch(output, detectors: List[ScorerType]) -> Dict[str, torch.Tensor]:
    scores = {}
    for detector in detectors:
        s = detector.compute_scores_benchmark(output)
        scores |= {f"{detector}+{k}": v for k, v in flatten_dict(s).items()}

    return scores

def energy(logits: Tensor, temperature: float) -> Tensor: 
    """Description. Energy function for multi-class classification."""

    logits = logits.detach()
    score = torch.sum(torch.exp(logits / temperature), dim=-1) 
    score = - temperature * torch.log(score)

    return score

def msp(likelihood: Tensor) -> Tensor: 
    """Description. Maximum softmax probability score."""

    likelihood = likelihood.detach()
    likelihood_max = likelihood.max(dim=-1).values
    score = 1 - torch.exp(likelihood_max)

    return score

def evaluate_dataloader(
        model,
        data_loader: DataLoader,
        tokenizer,
        detectors: List[ScorerType],
        energy_temperature: float=1.
) -> Dict[str, List]:
    # Initialize the scores dictionary
    records: Dict[str, List] = {
        f"{detector}+{score_name}": []
        for detector in detectors
        for score_name in detector.score_names
    }

    records["msp"] = []
    records["energy"] = []
    records["pred_label"] = []
    records["true_label"] = []
    records["correct"] = []

    loop = tqdm(data_loader, position=0, leave=True)
    loop.set_description("Evaluating detectors...")

    for batch_idx, batch in enumerate(loop):

        inputs = tokenizer(
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        )
        labels = batch["labels"]

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        output = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
        output['encoder_hidden_states'] = output['hidden_states']

        # Should be a dictionary with keys ood scores,
        # each containing a numpy array of shape (batch_size, num_return_sequences))

        ood_scores = evaluate_batch(output, detectors)

        ood_scores = {
            k: (scores.tolist() if not isinstance(scores, list) else scores)
            for k, scores in ood_scores.items()
        }

        for k, scores in ood_scores.items():
            records[k].extend(scores)

        logits = output.logits
        energy_scores = energy(logits, energy_temperature)

        # log-softmax is advantageous over softmax for improved numerical performance and gradient optimization
        likelihood = torch.nn.functional.log_softmax(logits, dim=-1)
        msp_scores = msp(likelihood)

        records["msp"].extend([s.item() for s in msp_scores])
        records["energy"].extend([s.item() for s in energy_scores])

        pred_labels = torch.argmax(likelihood, dim=-1)

        if isinstance(pred_labels, Tensor) and isinstance(labels, Tensor):
            correct = (pred_labels == labels).tolist()
            labels = labels.tolist()
            pred_labels = pred_labels.tolist()
        else: 
            correct = [i==j for i, j in zip(pred_labels, labels)]            

        records["pred_label"].extend(pred_labels)
        records["true_label"].extend(labels)
        records["correct"].extend(correct)

    return records
