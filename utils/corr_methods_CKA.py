from typing import Callable
import json
import pickle
from itertools import product as p
from os.path import basename, dirname
import logging
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F



"""
See this repository:
- https://github.com/johnmwu/contextual-corr-analysis/blob/HEAD/corr_methods.py
"""

def reshape_Conv2d(tensor, size=(1,1), reshape_method="AveragePool", compare_dim="Channel",
                ):
    if reshape_method == "AveragePool":
        tensor_reshape = F.adaptive_avg_pool2d(tensor, size)
    elif reshape_method == "Interpolate":
        tensor_reshape = F.interpolate(tensor, size=size, mode='bilinear')
        # tensor_reshape = F.interpolate(tensor, size=size, mode='linear')
    elif reshape_method == "OnlyReshape":
        tensor_reshape = tensor
    else:
        raise NotImplementedError

    if compare_dim == "Channel":
        # [N, C, H, W] -> [C, N, H, W]
        tensor_reshape = tensor_reshape.permute(1, 0, 2, 3)
        # [C, N, H, W] -> [C, N*H*W]
        tensor_reshape = tensor_reshape.reshape(
            tensor_reshape.shape[0], -1)
        # # [C, N*H*W] -> [N*H*W, C]
        # tensor_reshape = tensor_reshape.permute(
        #     1, 0)
    elif compare_dim == "Data":
        # [N, C, H, W] -> [N, C*H*W]
        tensor_reshape = tensor_reshape.reshape(
            tensor_reshape.shape[0], -1)
    else:
        raise NotImplementedError
    return tensor_reshape

def compare_tensor_size(tensor1, tensor2):
    wh_acts1 = (tensor1.shape[2], tensor1.shape[3])
    wh_acts2 = (tensor2.shape[2], tensor2.shape[3])
    size1 = wh_acts1[0] * wh_acts1[1]
    size2 = wh_acts2[0] * wh_acts2[1]

    if size1 < size2:
        smaller = 1
    elif size1 > size2:
        smaller = 2
    else:
        if tensor1.shape[2] < tensor2.shape[2]:
            smaller = 1
        elif tensor1.shape[2] > tensor2.shape[2]:
            smaller = 2
        else:
            smaller = 0
            logging.info("Detect acts1 and acts2 have the same size......")
    return smaller, wh_acts1, wh_acts2



def reshape_2tensors(copy_acts1, copy_acts2,
        if_same_layer=None, reshape_method="Interpolate"):

    acts1_module_type = asign_type(copy_acts1)
    acts2_module_type = asign_type(copy_acts2)

    print("acts1_module_type: {}, acts2_module_type: {}".format(
        acts1_module_type, acts2_module_type
    ))
    if acts1_module_type in ["Conv2d", "image"] and acts2_module_type in ["Conv2d", "image"]:
        smaller, reshape_size1, reshape_size2 = compare_tensor_size(copy_acts1, copy_acts2)
        if if_same_layer and smaller != 0:
            logging.error("ERROR!! same layer indicator is true but the size of them is different!!!!! ")
            raise NotImplementedError

        if reshape_method == "AveragePool":
            if if_same_layer:
                logging.info("Same layer reshape, using AveragePool")
                copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="AveragePool", compare_dim="Channel")
                copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="AveragePool", compare_dim="Channel")
            else:
                logging.info("Diff layer reshape, using AveragePool")
                if smaller in [0, 1, 2]:
                    copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="AveragePool", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="AveragePool", compare_dim="Data")
                else:
                    raise NotImplementedError

        elif reshape_method == "Subsampling":
            if if_same_layer:
                copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Channel")
                copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Channel")
            else:
                if smaller == 0:
                    copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Data")
                elif smaller == 1:
                    copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, size=reshape_size1, reshape_method="Interpolate", compare_dim="Data")
                elif smaller == 2:
                    copy_acts1 = reshape_Conv2d(copy_acts1, size=reshape_size2, reshape_method="Interpolate", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Data")
                else:
                    raise NotImplementedError

        elif reshape_method == "Interpolate":
            if if_same_layer:
                copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Channel")
                copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Channel")
            else:
                if smaller == 0:
                    copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Data")
                elif smaller == 1:
                    copy_acts1 = reshape_Conv2d(copy_acts1, size=reshape_size2, reshape_method="Interpolate", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Data")
                elif smaller == 2:
                    copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Data")
                    copy_acts2 = reshape_Conv2d(copy_acts2, size=reshape_size1, reshape_method="Interpolate", compare_dim="Data")
                else:
                    raise NotImplementedError

        elif reshape_method == "OnlyReshape":
            if if_same_layer:
                copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Channel")
                copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Channel")
            else:
                copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape", compare_dim="Data")
                copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape", compare_dim="Data")
        else:
            raise NotImplementedError

    elif acts1_module_type in ["Conv2d", "image"] and acts2_module_type == "Linear":
        # [N, C, H, W] -> [N, C]
        copy_acts1 = reshape_Conv2d(copy_acts1, size=(1,1), reshape_method="AveragePool", compare_dim="Data")
    elif acts1_module_type == "Linear" and acts2_module_type in ["Conv2d", "image"]:
        # [N, C, H, W] -> [N, C]
        copy_acts2 = reshape_Conv2d(copy_acts2, size=(1,1), reshape_method="AveragePool", compare_dim="Data")
    elif acts1_module_type == "Linear" and acts2_module_type == "Linear":
        pass
    else:
        raise NotImplementedError
    return copy_acts1, acts1_module_type, copy_acts2, acts2_module_type


# def reshape_2tensors(copy_acts1, copy_acts2, reshape_method="Interpolate"):

#     acts1_module_type = asign_type(copy_acts1)
#     acts2_module_type = asign_type(copy_acts2)

#     print("acts1_module_type: {}, acts2_module_type: {}".format(
#         acts1_module_type, acts2_module_type
#     ))
#     if acts1_module_type in ["Conv2d", "image"] and acts2_module_type in ["Conv2d", "image"]:
#         wh_acts1 = (copy_acts1.shape[2], copy_acts1.shape[3])
#         wh_acts2 = (copy_acts2.shape[2], copy_acts2.shape[3])
#         size1 = wh_acts1[0] * wh_acts1[1]
#         size2 = wh_acts2[0] * wh_acts2[1]

#         if size1 < size2:
#             reshape_size = (copy_acts1.shape[2], copy_acts1.shape[3])
#             copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape")
#             copy_acts2 = reshape_Conv2d(copy_acts2, size=reshape_size, reshape_method=reshape_method)
#         elif size1 > size2:
#             reshape_size = (copy_acts2.shape[2], copy_acts2.shape[3])
#             copy_acts1 = reshape_Conv2d(copy_acts1, size=reshape_size, reshape_method=reshape_method)
#             copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape")
#         else:
#             if copy_acts1.shape[2] < copy_acts2.shape[2]:
#                 reshape_size = (copy_acts1.shape[2], copy_acts1.shape[3])
#                 copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape")
#                 copy_acts2 = reshape_Conv2d(copy_acts2, size=reshape_size, reshape_method=reshape_method)

#             elif copy_acts1.shape[2] > copy_acts2.shape[2]:
#                 reshape_size = (copy_acts2.shape[2], copy_acts2.shape[3])
#                 copy_acts1 = reshape_Conv2d(copy_acts1, size=reshape_size, reshape_method=reshape_method)
#                 copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape")
#             else:
#                 copy_acts1 = reshape_Conv2d(copy_acts1, reshape_method="OnlyReshape")
#                 copy_acts2 = reshape_Conv2d(copy_acts2, reshape_method="OnlyReshape")
#                 logging.info("Detect acts1 and acts2 have the same size......")

#     elif acts1_module_type in ["Conv2d", "image"] and acts2_module_type == "Linear":
#         # [N, C, H, W] -> [N, C]
#         copy_acts1 = reshape_Conv2d(copy_acts1, size=(1,1), reshape_method="AveragePool")
#     elif acts1_module_type == "Linear" and acts2_module_type in ["Conv2d", "image"]:
#         # [N, C, H, W] -> [N, C]
#         copy_acts2 = reshape_Conv2d(copy_acts2, size=(1,1), reshape_method="AveragePool")
#     elif acts1_module_type == "Linear" and acts2_module_type == "Linear":
#         pass
#     else:
#         raise NotImplementedError
#     return copy_acts1, acts1_module_type, copy_acts2, acts2_module_type


def asign_type(tensor):
    if tensor.dim() == 4:
        tensor_type = "Conv2d"
    elif tensor.dim() == 2:
        tensor_type = "Linear"
    else:
        raise NotImplementedError
    return tensor_type


def two_same_models_compare(num_neurons_d, layers_list,
        activation_X, activation_Y, model_type="image", reshape_method="Interpolate"):

    activation_X_type = {}
    activation_Y_type = {}
    same_layer_indicator = {}
    for keyX in layers_list:
        assert keyX in activation_X
        assert keyX in activation_Y
        activation_X_type[keyX] = model_type
        activation_Y_type[keyX] = model_type
        same_layer_indicator[keyX] = {}
        for keyY in layers_list:
            if keyX == keyY:
                same_layer_indicator[keyX][keyY] = 1
            else:
                same_layer_indicator[keyX][keyY] = 0

    cca = CCA(num_neurons_d, activation_X, activation_X_type, 
                activation_Y, activation_Y_type, same_layer_indicator, 
                reshape_method=reshape_method)
    cca.compute_correlations()
    # print(cca_ab.corrs)
    logging.info("Correlation similarity: {}".format(cca.sv_similarities))
    return cca

class Method(object):
    """
    Abstract representation of a correlation method. 

    Input:
        representations_X: dict,
        representations_X_type: dict,
        representations_Y: dict,
        representations_Y_type: dict,
        reshape_method: "AveragePool" or "Interpolate" or "OnlyReshape".

    Y can be None, then calculate corr between X and itself.
        And the last corr matrix is dict[keyX][keyX].

    Example instances are MaxCorr, MinCorr, MaxLinReg, MinLinReg, CCA,
    LinCKA.
    """
    def __init__(self, num_neurons_d, 
                representations_X, representations_X_type, 
                representations_Y=None, representations_Y_type=None,
                same_layer_indicator=None,
                device="cpu",
                reshape_method="AveragePool"):
        self.num_neurons_d = num_neurons_d
        self.representations_X = representations_X
        self.representations_X_type = representations_X_type
        self.representations_Y = representations_Y
        self.representations_Y_type = representations_Y_type
        # self.same_layer_indicator = same_layer_indicator
        # assert len(self.same_layer_indicator) == len(self.representations_X)

        self.device = device
        self.reshape_method = reshape_method

        if self.representations_Y is None:
            assert self.representations_Y_type is None
            self.inner_corr = True
        else:
            self.inner_corr = False

        logging.info("Similarity comparing. under {} mode ".format(
            "Inner correlations" if self.inner_corr else "Outer correlations"
        ))


class CCA(Method):
    """
    Abstract representation of a correlation method. 

    Input:
        representations_X: dict,
        representations_X_type: dict,
        representations_Y: dict,
        representations_Y_type: dict,
        reshape_method: "AveragePool" or "Interpolate" or "OnlyReshape".

    Y can be None, then calculate corr between X and itself.
        And the last corr matrix is dict[keyX][keyX].

    Example instances are MaxCorr, MinCorr, MaxLinReg, MinLinReg, CCA,
    LinCKA.
    """
    def __init__(self, num_neurons_d, 
                representations_X, representations_X_type, 
                representations_Y=None, representations_Y_type=None,
                same_layer_indicator=None,
                device="cpu",
                percent_variance=0.99, normalize_dimensions=True,
                save_cca_transforms=False, reshape_method="AveragePool"):
        super().__init__(num_neurons_d, representations_X, representations_X_type, 
                representations_Y, representations_Y_type,
                same_layer_indicator,
                device,
                reshape_method=reshape_method
            )
        self.percent_variance = percent_variance
        self.normalize_dimensions = normalize_dimensions
        self.save_cca_transforms = save_cca_transforms

        self.normd_representations_X = self.normalize(self.representations_X)
        self.normd_representations_Y = self.normalize(self.representations_Y) \
            if self.representations_Y is not None else None

        self.transforms = {}
        self.transforms["X"] = {network: {} for network in self.representations_X}
        self.transforms["Y"] = {network: {} for network in self.representations_Y} \
            if self.representations_Y is not None else None
        # Set `whitening_transforms`, `pca_directions`
        # {network: whitening_tensor}
        self.whitening_transforms = {}
        self.whitening_transforms["X"] = {network: {} for network in self.representations_X}
        self.whitening_transforms["Y"] = {network: {} for network in self.representations_Y} \
            if self.representations_Y is not None else None

        self.pca_directions = {}
        self.pca_directions["X"] = {network: {} for network in self.representations_X}
        self.pca_directions["Y"] = {network: {} for network in self.representations_Y} \
            if self.representations_Y is not None else None


    def normalize(self, representations):
        nrepresentations_d = {}
        if self.normalize_dimensions:
            for network in tqdm(representations, desc='mu, sigma'):
                t = representations[network]
                means = t.mean(0, keepdims=True)
                stdevs = t.std(0, keepdims=True)

                nrepresentations_d[network] = (t - means) / stdevs
        else:
            nrepresentations_d = representations
        return nrepresentations_d

    def whitening_transform(self, network, other_network, tensor, which="X"):

        for trial in range(50):
            try:
                U, S, V = torch.svd(tensor)
                assert U.shape!=(0,0)
                print("U=",U)
                print("S=",S)
                print("V=",V)
                break
            except:
                print("SVD calculating {} error, adding noise, the norm is {}, \
                    detect NaN value: {}, detect INF value: {}, try {} times".format(
                        network, tensor.norm(),
                        "NaN existing!!" if True in torch.isnan(tensor) else "Nan not exists.",
                        "INF existing!!" if True in torch.isinf(tensor) else "INF not exists.",
                        trial))

                if True in torch.isnan(tensor):
                    tensor[torch.isnan(tensor)] = 0.
                    isnan = True
                else:
                    isnan=False

                if True in torch.isinf(tensor):
                    tensor[torch.isinf(tensor)] = 0.
                    isinf = True
                else:
                    isinf=False

                if (not isnan) and (not isinf):
                    # acts1 = acts1*1e-1 + np.random.normal(size=acts1.shape)*epsilon
                    # X = X*1e-1 + torch.rand(X.shape)*1e-6
                    tensor = tensor*1e-1 + torch.rand(tensor.shape)*1e-6
                    # tensor = torch.rand(tensor.shape)
                    # X = X + np.random.normal(size=X.shape)*1e-3
                logging.info("SVD calculating {}-{} failed, data shape is {}, new tensor is {}, \
                    the norm is {}, detect NaN value: {}, detect INF value: {}".format(
                    network, other_network, tensor.shape, tensor[:5, :10], tensor.norm(),
                    "NaN existing!!" if True in torch.isnan(tensor) else "Nan not exists.",
                    "INF existing!!" if True in torch.isinf(tensor) else "INF not exists.",
                ))
                if trial + 1 == 50:
                    print("SVD calculating %s failed, data shape is %s" % 
                        (network, tensor.shape))
                    raise

        var_sums = torch.cumsum(S.pow(2), 0)
        wanted_size = torch.sum(var_sums.lt(var_sums[-1] *
                                            self.percent_variance)).item()

        print('For network', network, 'wanted size is', wanted_size)

        if self.save_cca_transforms:
            whitening_transform = torch.mm(V, torch.diag(1/S))
            self.whitening_transforms[which][network][other_network] = \
                whitening_transform[:, :wanted_size]
        self.pca_directions[which][network][other_network] = U[:, :wanted_size]



    def compute_correlations(self):
        # Set 
        # `self.transforms`: {network: {other: svcca_transform}}
        # `self.corrs`: {network: {other: canonical_corrs}}
        # `self.pw_alignments`: {network: {other: unnormalized pw weights}}
        # `self.pw_corrs`: {network: {other: pw_alignments*corrs}}
        # `self.sv_similarities`: {network: {other: svcca_similarities}}
        # `self.pw_similarities`: {network: {other: pwcca_similarities}}

        self.corrs = {network: {} for network in self.representations_X}
        self.pw_alignments = {network: {} for network in
                            self.representations_X}
        self.pw_corrs = {network: {} for network in self.representations_X}
        self.sv_similarities = {network: {} for network in
                                self.representations_X}
        self.pw_similarities = {network: {} for network in
                                self.representations_X}

        if self.inner_corr:
            for network, other_network in tqdm(p(self.representations_X,
                                                self.representations_X),
                                            desc='cca',
                                            total=len(self.representations_X)**2):
                if network == other_network:
                    continue

                # if other_network in self.transforms[network]: 
                #     continue

                if other_network in self.sv_similarities[network]: 
                    continue

                logging.info("Calculating {} and {} similarity...".format(network, other_network))
                self.get_cca_similarity(
                    network,
                    self.normd_representations_X[network],
                    None,
                    other_network,
                    self.normd_representations_X[other_network],
                    None,
                    device=self.device,
                    reshape_method=self.reshape_method,
                )

        else:
            for network, other_network in tqdm(p(self.representations_X,
                                                self.representations_Y),
                                            desc='cca',
                                            total=len(self.representations_X)*len(self.representations_Y)):
                logging.info("Calculating {} and {} similarity...".format(network, other_network))
                self.get_cca_similarity(
                    network,
                    self.normd_representations_X[network],
                    None,
                    other_network,
                    self.normd_representations_Y[other_network][:len(self.normd_representations_X[network])],
                    None,
                    device=self.device,
                    reshape_method=self.reshape_method,
                )

                # Compute `self.pw_alignments`, `self.pw_corrs`, `self.pw_similarities`. 
                # This is not symmetric
                # pass this

    def get_cca_similarity(self, acts1_name, acts1, acts1_module_type, acts2_name, acts2, acts2_module_type,
                    device="cpu",
                    reshape_method="AveragePool",
                ):
        # assert acts1_module_type in ("Conv2d", "Linear", "image")
        # assert acts2_module_type in ("Conv2d", "Linear", "image")
        if device not in ("cpu", "cuda"):
            raise RuntimeError(f"Unknown device name {device}")

        copy_acts1 = deepcopy(acts1)
        copy_acts2 = deepcopy(acts2)

        copy_acts1 = copy_acts1.to(device)
        copy_acts2 = copy_acts2.to(device)

        # if self.same_layer_indicator[acts1_name][acts2_name]:
        #     if_same_layer = True
        # else:
        #     if_same_layer = False

        copy_acts1, acts1_module_type, copy_acts2, acts2_module_type = reshape_2tensors(
            copy_acts1, copy_acts2,
            # if_same_layer=self.same_layer_indicator[acts1_name][acts2_name],
            reshape_method=reshape_method
        )
        logging.info("tensor : {} has shape: {}, tensor: {} has shape: {}".format(
            acts1_name, copy_acts1.shape, acts2_name, copy_acts2.shape
        ))

        self.whitening_transform(acts1_name, acts2_name, copy_acts1, which="X")
        if self.representations_Y is not None:
            self.whitening_transform(acts2_name, acts1_name, copy_acts2, which="Y")
        else:
            self.whitening_transform(acts2_name, acts1_name, copy_acts2, which="X")

        X = self.pca_directions["X"][acts1_name][acts2_name]
        if self.representations_Y is not None:
            Y = self.pca_directions["Y"][acts2_name][acts1_name]
        else:
            Y = self.pca_directions["X"][acts1_name][acts2_name]

        # Perform SVD for CCA.
        # u s vt = Xt Y
        # s = ut Xt Y v
        for trial in range(50):
            try:
                tensor=torch.mm(X.t(), Y)
                u, s, v = torch.svd(tensor)
                break
            except:
                print("SVD calculating error, adding noise, the norm is {}, \
                    detect NaN value: {}, detect INF value: {}, try {} times".format(
                        tensor.norm(),
                        "NaN existing!!" if True in torch.isnan(tensor) else "Nan not exists.",
                        "INF existing!!" if True in torch.isinf(tensor) else "INF not exists.",
                        trial))
                isnan = False
                isinf = False
                if True in torch.isnan(tensor):
                    tensor[torch.isnan(tensor)] = 0.
                    isnan = True

                if True in torch.isinf(tensor):
                    tensor[torch.isinf(tensor)] = 0.
                    isinf = True

                if (not isnan) and (not isinf):
                    # acts1 = acts1*1e-1 + np.random.normal(size=acts1.shape)*epsilon
                    # X = X*1e-1 + torch.rand(X.shape)*1e-6
                    tensor = tensor*1e-1 + torch.rand(tensor.shape)*1e-5
                    # tensor = torch.rand(tensor.shape)
                    # X = X + np.random.normal(size=X.shape)*1e-3
                logging.info("SVD calculating failed, data shape is {}, new tensor is {}, \
                    the norm is {}, detect NaN value: {}, detect INF value: {}".format(
                    tensor.shape, tensor[:5, :10], tensor.norm(),
                    "NaN existing!!" if True in torch.isnan(tensor) else "Nan not exists.",
                    "INF existing!!" if True in torch.isinf(tensor) else "INF not exists.",
                ))
                if trial + 1 == 50:
                    print("SVD calculating failed, data shape is %s" % 
                        (tensor.shape))
                    raise
        # `self.transforms`, `self.corrs`, `self.sv_similarities`
        if self.save_cca_transforms:
            self.transforms["X"][acts1_name][acts2_name] = torch.mm(
                self.whitening_transforms["X"][acts1_name][acts2_name], u).cpu().numpy()
            if self.representations_Y is not None:
                self.transforms["Y"][acts2_name][acts1_name] = torch.mm(
                    self.whitening_transforms["Y"][acts2_name][acts1_name], v).cpu().numpy()
            else:
                self.transforms["X"][acts2_name][acts1_name] = torch.mm(
                    self.whitening_transforms["X"][acts2_name][acts1_name], v).cpu().numpy()

        self.corrs[acts1_name][acts2_name] = s.cpu().numpy()
        if not self.inner_corr:
            self.corrs[acts2_name][acts1_name] = s.cpu().numpy()

        similarity = s.mean().item()
        self.sv_similarities[acts1_name][acts2_name] = similarity
        if not self.inner_corr:
            self.sv_similarities[acts2_name][acts1_name] = similarity

        logging.info("tensor : {} and tensor: {} has similarity: {}".format(
            acts1_name, acts2_name, similarity
        ))
        return similarity
        # # For X
        # H = torch.mm(X, u)
        # Z = self.get_representation_reshape(if_same_layer, network, "X")
        # align = torch.abs(torch.mm(H.t(), Z))
        # a = torch.sum(align, dim=1, keepdim=False)
        # self.pw_alignments[network][other_network] = a.cpu().numpy()
        # self.pw_corrs[network][other_network] = (s*a).cpu().numpy()
        # self.pw_similarities[network][other_network] = (torch.sum(s*a)/torch.sum(a)).item()

        # # For Y
        # H = torch.mm(Y, v)
        # Z = self.get_representation_reshape(if_same_layer, other_network, "X")
        # align = torch.abs(torch.mm(H.t(), Z))
        # a = torch.sum(align, dim=1, keepdim=False)
        # self.pw_alignments[other_network][network] = a.cpu().numpy()
        # self.pw_corrs[other_network][network] = (s*a).cpu().numpy()
        # self.pw_similarities[other_network][network] = (torch.sum(s*a)/torch.sum(a)).item()


    def write_correlations(self, output_file):
        if self.save_cca_transforms:
            output = {
                "transforms": self.transforms,
                "corrs": self.corrs,
                "sv_similarities": self.sv_similarities,
                "pw_alignments": self.pw_alignments,
                "pw_corrs": self.pw_corrs,
                "pw_similarities": self.pw_similarities,
            }
        else:
            output = {
                "corrs": self.corrs,
                "sv_similarities": self.sv_similarities,
                "pw_alignments": self.pw_alignments,
                "pw_corrs": self.pw_corrs,
                "pw_similarities": self.pw_similarities,
            }
        with open(output_file, "wb") as f:
            pickle.dump(output, f)

    def __str__(self):
        return "cca"


import numpy as np

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T, atol=1e-2):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)




