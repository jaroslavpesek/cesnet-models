"""
Module doc string.
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DistanceLayer(nn.Module):
    """
    Takes representation of the features and calculates the distance 
    between the representation and the learned prototypes.
    """
    def __init__(self, in_features: int, n_prototypes: int, p_norm_distance: float = 2.0):
        super(DistanceLayer, self).__init__()
        self.prototype = nn.Parameter(torch.empty(n_prototypes, in_features), requires_grad=True)
        init.kaiming_uniform_(self.prototype, a=math.sqrt(5))
        self.p_norm_distance = p_norm_distance

    def forward(self, x: torch.Tensor):
        """
        Forward function to compute distances between input and prototypes.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Distances between input features and prototypes,
                          shape (batch_size, n_prototypes).
        """
        # Expand dimensions for broadcasting
        x_extended = x.unsqueeze(1)  # Shape: (batch_size, 1, in_features)
        prototypes_expanded = self.prototype.unsqueeze(0)  # Shape: (1, n_prototypes, in_features)

        distances = torch.cdist(x_extended, prototypes_expanded, p=self.p_norm_distance)
        return distances


class SupportLayer(nn.Module):
    """
    Calculates the distance-based support values given the distances.
    """
    def __init__(self, n_prototypes: int):
        super(SupportLayer, self).__init__()
        # Initialize alpha in (0, 1)
        self.alpha = nn.Parameter(torch.rand(n_prototypes), requires_grad=True)
        # Initialize eta with normal distribution
        self.eta = nn.Parameter(torch.randn(n_prototypes), requires_grad=True)

    def forward(self, distances: torch.Tensor):
        """
        Forward function to compute support values.
        
        Args:
            distances (torch.Tensor): Distance tensor of shape (batch_size, n_prototypes).

        Returns:
            torch.Tensor: Support values s^i for each input and prototype
            shape (batch_size, n_prototypes).
        """
        alpha = torch.sigmoid(self.alpha)
        eta_distances_squared = (self.eta * distances) ** 2
        support_values = alpha * torch.exp(-eta_distances_squared)

        return support_values


class MassFunctionLayer(nn.Module):
    """
    Custom module that performs element-wise multiplication of the input tensor
    """
    def __init__(self, n_prototypes: int, n_classes: int):
        super(MassFunctionLayer, self).__init__()
        self.h = nn.Parameter(torch.rand(n_prototypes, n_classes), requires_grad=True)
        self.h.data = F.normalize(self.h.data, p=1, dim=1)

    def forward(self, s: torch.Tensor):
        """
        Transform the batch of support values to mass functions.
        """
        self.h.data = F.normalize(self.h.data, p=1, dim=1)
        s = s.view(s.size(0), -1, 1)
        sh = s * self.h.unsqueeze(0)
        one_minus_s = 1 - s
        m_i = torch.cat((sh, one_minus_s), dim=2)

        return m_i



class DempsterAggregationLayer(nn.Module):
    """
    Aggregates mass functions using Dempster's rule with a vectorized approach.
    """
    def __init__(self, n_prototypes, num_class):
        super(DempsterAggregationLayer, self).__init__()
        self.n_prototypes = n_prototypes
        self.num_class = num_class

    def forward(self, inputs):
        """
        Dempster's rule of combination.
        """
        m1 = inputs[..., 0, :]
        omega1 = torch.unsqueeze(inputs[..., 0, -1], -1)
        for i in range(self.n_prototypes - 1):
            m2 = inputs[..., (i + 1), :]
            omega2 = torch.unsqueeze(inputs[..., (i + 1), -1], -1)
            combine1 = torch.mul(m1, m2)
            combine2 = torch.mul(m1, omega2)
            combine3 = torch.mul(omega1, m2)
            combine1_2 = combine1 + combine2
            combine2_3 = combine1_2 + combine3
            combine2_3 = combine2_3 / torch.sum(combine2_3, dim=-1, keepdim=True)
            m1 = combine2_3
            omega1 = torch.unsqueeze(combine2_3[..., -1], -1)
        return m1


class NormalizeMassFunction(nn.Module):
    """
    Custom module to normalize mass functions.
    """
    def __init__(self):
        super(NormalizeMassFunction, self).__init__()

    def forward(self, mass_functions: torch.Tensor):
        """
        Forward function to normalize mass functions.
        :param mass_functions: Tensor of shape (batch_size, n_classes+1)
                               where the last column represents the uncertainty.
        :return: Normalized mass functions.
        """
        return mass_functions / torch.sum(mass_functions, dim=1, keepdim=True)


class DMPignistic(nn.Module):
    """
    Get pignistic probability from mass function.
    """
    def __init__(self, num_classes: int):
        super(DMPignistic, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor):
        """
        Forward function to compute pignistic probabilities.
        :param mass_functions: Tensor of shape (batch_size, n_classes+1)
                               where the last column represents the uncertainty.
        :return: Pignistic probabilities.
        """
        average_pignistic = inputs[..., -1] / self.num_classes
        average_pignistic = average_pignistic.unsqueeze(-1)

        # Extract the mass functions for each class
        mass_class = inputs[..., :-1]

        # Add the average pignistic value to each class mass
        pignistic_prob = mass_class + average_pignistic

        return pignistic_prob


class DM(nn.Module):
    """
    Decision maker.
    """
    def __init__(self, nu, num_class):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class = num_class

    def forward(self, inputs: torch.Tensor):
        """
        Decision making with pessimistic decision maker.
        """
        # Calculate the upper term
        upper = (1 - self.nu) * inputs[:, -1].unsqueeze(-1)
        upper = upper.expand(-1, self.num_class + 1)

        outputs = inputs + upper
        outputs = outputs[:, 0:-1]

        return outputs
