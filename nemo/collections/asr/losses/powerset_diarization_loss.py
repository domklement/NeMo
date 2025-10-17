from functools import cached_property, singledispatch, partial
from itertools import combinations, permutations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.special
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["PowerSetDiarizationLoss"]


@singledispatch
def permutate(y1, y2, cost_func: Optional[Callable] = None, return_cost: bool = False):
    """Find cost-minimizing permutation

    Parameters
    ----------
    y1 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_classes_1)
    y2 : np.ndarray or torch.Tensor
        (num_samples, num_classes_2) or (batch_size, num_samples, num_classes_2)
    cost_func : callable
        Takes two (num_samples, num_classes) sequences and returns (num_classes, ) pairwise cost.
        Defaults to computing mean squared error.
    return_cost : bool, optional
        Whether to return cost matrix. Defaults to False.

    Returns
    -------
    permutated_y2 : np.ndarray or torch.Tensor
        (batch_size, num_samples, num_classes_1)
    permutations : list of tuple
        List of permutations so that permutation[i] == j indicates that jth speaker of y2
        should be mapped to ith speaker of y1.  permutation[i] == None when none of y2 speakers
        is mapped to ith speaker of y1.
    cost : np.ndarray or torch.Tensor, optional
        (batch_size, num_classes_1, num_classes_2)
    """
    raise TypeError()


def mse_cost_func(Y, y, **kwargs):
    """Compute class-wise mean-squared error

    Parameters
    ----------
    Y, y : (num_frames, num_classes) torch.tensor

    Returns
    -------
    mse : (num_classes, ) torch.tensor
        Mean-squared error
    """
    return torch.mean(F.mse_loss(Y, y, reduction="none"), axis=0)


def mae_cost_func(Y, y, **kwargs):
    """Compute class-wise mean absolute difference error

    Parameters
    ----------
    Y, y: (num_frames, num_classes) torch.tensor

    Returns
    -------
    mae : (num_classes, ) torch.tensor
        Mean absolute difference error
    """
    return torch.mean(torch.abs(Y - y), axis=0)


@permutate.register
def permutate_torch(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_classes_1 = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes)."
        raise ValueError(msg)

    batch_size_, num_samples_, num_classes_2 = y2.shape
    if batch_size != batch_size_ or num_samples != num_samples_:
        msg = f"Shape mismatch: {tuple(y1.shape)} vs. {tuple(y2.shape)}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    if return_cost:
        costs = []

    permutated_y2 = torch.zeros(y1.shape, device=y2.device, dtype=y2.dtype)

    for b, (y1_, y2_) in enumerate(zip(y1, y2)):
        # y1_ is (num_samples, num_classes_1)-shaped
        # y2_ is (num_samples, num_classes_2)-shaped
        with torch.no_grad():
            cost = torch.stack(
                [
                    cost_func(y2_, y1_[:, i : i + 1].expand(-1, num_classes_2))
                    for i in range(num_classes_1)
                ],
            )

        if num_classes_2 > num_classes_1:
            padded_cost = F.pad(
                cost,
                (0, 0, 0, num_classes_2 - num_classes_1),
                "constant",
                torch.max(cost) + 1,
            )
        else:
            padded_cost = cost

        permutation = [None] * num_classes_1
        for k1, k2 in zip(*linear_sum_assignment(padded_cost.cpu())):
            if k1 < num_classes_1:
                permutation[k1] = k2
                permutated_y2[b, :, k1] = y2_[:, k2]
        permutations.append(tuple(permutation))

        if return_cost:
            costs.append(cost)

    if return_cost:
        return permutated_y2, permutations, torch.stack(costs)

    return permutated_y2, permutations


@permutate.register
def permutate_numpy(
    y1: np.ndarray,
    y2: np.ndarray,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int]]]:

    output = permutate(
        torch.from_numpy(y1),
        torch.from_numpy(y2),
        cost_func=cost_func,
        return_cost=return_cost,
    )

    if return_cost:
        permutated_y2, permutations, costs = output
        return permutated_y2.numpy(), permutations, costs.numpy()

    permutated_y2, permutations = output
    return permutated_y2.numpy(), permutations


class Powerset(nn.Module):
    """Powerset to multilabel conversion, and back.

    Parameters
    ----------
    num_classes : int
        Number of regular classes.
    max_set_size : int
        Maximum number of classes in each set.
    """
    def __init__(self, num_classes: int, max_set_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.max_set_size = max_set_size

        self.register_buffer("mapping", self.build_mapping(), persistent=False)
        self.register_buffer("cardinality", self.build_cardinality(), persistent=False)

    @cached_property
    def num_powerset_classes(self) -> int:
        # compute number of subsets of size at most "max_set_size"
        # e.g. with num_classes = 3 and max_set_size = 2:
        # {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}
        return int(
            sum(
                scipy.special.binom(self.num_classes, i)
                for i in range(0, self.max_set_size + 1)
            )
        )

    def build_mapping(self) -> torch.Tensor:
        """Compute powerset to regular mapping

        Returns
        -------
        mapping : (num_powerset_classes, num_classes) torch.Tensor
            mapping[i, j] == 1 if jth regular class is a member of ith powerset class
            mapping[i, j] == 0 otherwise

        Example
        -------
        With num_classes == 3 and max_set_size == 2, returns

            [0, 0, 0]  # none
            [1, 0, 0]  # class #1
            [0, 1, 0]  # class #2
            [0, 0, 1]  # class #3
            [1, 1, 0]  # classes #1 and #2
            [1, 0, 1]  # classes #1 and #3
            [0, 1, 1]  # classes #2 and #3

        """
        mapping = torch.zeros(self.num_powerset_classes, self.num_classes)
        powerset_k = 0
        for set_size in range(0, self.max_set_size + 1):
            for current_set in combinations(range(self.num_classes), set_size):
                mapping[powerset_k, current_set] = 1
                powerset_k += 1

        return mapping

    def build_cardinality(self) -> torch.Tensor:
        """Compute size of each powerset class"""
        return torch.sum(self.mapping, dim=1)

    def to_multilabel(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """Convert predictions from powerset to multi-label

        Parameter
        ---------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            Soft predictions in "powerset" space.
        soft : bool, optional
            Return soft multi-label predictions. Defaults to False (i.e. hard predictions)
            Assumes that `powerset` are "log probabilities".

        Returns
        -------
        multi_label : (batch_size, num_frames, num_classes) torch.Tensor
            Predictions in "multi-label" space.
        """

        if soft:
            powerset_probs = torch.exp(powerset)
        else:
            powerset_probs = torch.nn.functional.one_hot(
                torch.argmax(powerset, dim=-1),
                self.num_powerset_classes,
            ).float()

        return torch.matmul(powerset_probs, self.mapping)

    def forward(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """Alias for `to_multilabel`"""
        return self.to_multilabel(powerset, soft=soft)

    def to_powerset(self, multilabel: torch.Tensor) -> torch.Tensor:
        """Convert (hard) predictions from multi-label to powerset

        Parameter
        ---------
        multi_label : (batch_size, num_frames, num_classes) torch.Tensor
            Prediction in "multi-label" space.

        Returns
        -------
        powerset : (batch_size, num_frames, num_powerset_classes) torch.Tensor
            Hard, one-hot prediction in "powerset" space.

        Note
        ----
        This method will not complain if `multilabel` is provided a soft predictions
        (e.g. the output of a sigmoid-ed classifier). However, in that particular
        case, the resulting powerset output will most likely not make much sense.
        """
        return F.one_hot(
            torch.argmax(torch.matmul(multilabel, self.mapping.T), dim=-1),
            num_classes=self.num_powerset_classes,
        )

    def _permutation_powerset(
        self, multilabel_permutation: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Helper function for `permutation_mapping` property

        Takes a (num_classes,)-shaped permutation in multilabel space and returns
        the corresponding (num_powerset_classes,)-shaped permutation in powerset space.
        This does not cache anything and only works on one single permutation at a time.

        Parameters
        ----------
        multilabel_permutation : tuple of int
            Permutation in multilabel space.

        Returns
        -------
        powerset_permutation : tuple of int
            Permutation in powerset space.

        Example
        -------
        >>> powerset = Powerset(3, 2)
        >>> powerset._permutation_powerset((1, 0, 2))
        # (0, 2, 1, 3, 4, 6, 5)

        """

        permutated_mapping: torch.Tensor = self.mapping[:, multilabel_permutation]

        arange = torch.arange(
            self.num_classes, device=self.mapping.device, dtype=torch.int
        )
        powers_of_two = (2**arange).tile((self.num_powerset_classes, 1))

        # compute the encoding of the powerset classes in this 2**N space, before and after
        # permutation of the columns (mapping cols=labels, mapping rows=powerset classes)
        before = torch.sum(self.mapping * powers_of_two, dim=-1)
        after = torch.sum(permutated_mapping * powers_of_two, dim=-1)

        # find before-to-after permutation
        powerset_permutation = (before[None] == after[:, None]).int().argmax(dim=0)

        # return as tuple of indices
        return tuple(powerset_permutation.tolist())

    @cached_property
    def permutation_mapping(self) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        """Mapping between multilabel and powerset permutations

        Example
        -------
        With num_classes == 3 and max_set_size == 2, returns

        {
            (0, 1, 2): (0, 1, 2, 3, 4, 5, 6),
            (0, 2, 1): (0, 1, 3, 2, 5, 4, 6),
            (1, 0, 2): (0, 2, 1, 3, 4, 6, 5),
            (1, 2, 0): (0, 2, 3, 1, 6, 4, 5),
            (2, 0, 1): (0, 3, 1, 2, 5, 6, 4),
            (2, 1, 0): (0, 3, 2, 1, 6, 5, 4)
        }
        """
        permutation_mapping = {}

        for multilabel_permutation in permutations(
            range(self.num_classes), self.num_classes
        ):
            permutation_mapping[
                tuple(multilabel_permutation)
            ] = self._permutation_powerset(multilabel_permutation)

        return permutation_mapping

class PowerSetDiarizationLoss(nn.Module):
    def __init__(self, num_speakers: int, max_overlapping_speakers: Optional[int] = None, temperature: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.num_speakers = num_speakers
        self.max_overlapping_speakers = max_overlapping_speakers
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.powerset = Powerset(num_classes=num_speakers, max_set_size=max_overlapping_speakers)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, ps_logits, ml_targets, target_lens, return_batch_accuracy: bool = False):
        ml_preds = self.powerset.to_multilabel(ps_logits)
        perm_targets, _ = permutate(ml_preds, ml_targets)
        perm_ps_targets = self.powerset.to_powerset(perm_targets.float())
        desired_logits = torch.concat([ps_logits[i, :target_lens[i], :] for i in range(len(target_lens))] , dim=0)
        desired_targets = torch.concat([perm_ps_targets[i, :target_lens[i], :] for i in range(len(target_lens))] , dim=0)
        # The loss is already divided by the number of frames
        loss = self.loss(desired_logits, desired_targets.float())
        
        if return_batch_accuracy:
            with torch.no_grad():
                batch_accuracy = (desired_targets.argmax(dim=-1) == desired_logits.argmax(dim=-1)).float().mean()
            return loss, batch_accuracy
        else:
            return loss


if __name__ == '__main__':
    pset = Powerset(num_classes=4, max_set_size=2)
    # 1. get multilabel from pset logits
    # 2. permutate labels
    # 3. Create permutated pset labels
    # 4. Compute CE loss between ps logits and permutated ps labels.
    ps_labels = torch.randint(0, 11, (1, 20))
    ps_onehot = F.one_hot(ps_labels, num_classes=pset.num_powerset_classes)
    ml_labels = pset.to_multilabel(ps_onehot)
    ps_logits = torch.rand(1, 20, pset.num_powerset_classes)

    ml_preds = pset.to_multilabel(ps_logits)
    perm_targets, _ = permutate(ml_preds, ml_labels)
    perm_ps_targets = pset.to_powerset(perm_targets.float())
    print('')
