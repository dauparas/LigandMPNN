import sys

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
from model_utils import (
    DecLayer,
    DecLayerJ,
    EncLayer,
    PositionalEncodings,
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)

from openfold.data.data_transforms import atom37_to_torsion_angles, make_atom14_masks
from openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from openfold.utils import feats
from openfold.utils.rigid_utils import Rigid

torch_pi = torch.tensor(np.pi, device="cpu")


map_mpnn_to_af2_seq = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    device="cpu",
)


def pack_side_chains(
    feature_dict,
    model_sc,
    num_denoising_steps,
    num_samples=10,
    repack_everything=True,
    num_context_atoms=16,
):
    device = feature_dict["X"].device
    torsion_dict = make_torsion_features(feature_dict, repack_everything)
    feature_dict["X"] = torsion_dict["xyz14_noised"]
    feature_dict["X_m"] = torsion_dict["xyz14_m"]
    if "Y" not in list(feature_dict):
        feature_dict["Y"] = torch.zeros(
            [
                feature_dict["X"].shape[0],
                feature_dict["X"].shape[1],
                num_context_atoms,
                3,
            ],
            device=device,
        )
        feature_dict["Y_t"] = torch.zeros(
            [feature_dict["X"].shape[0], feature_dict["X"].shape[1], num_context_atoms],
            device=device,
        )
        feature_dict["Y_m"] = torch.zeros(
            [feature_dict["X"].shape[0], feature_dict["X"].shape[1], num_context_atoms],
            device=device,
        )
    h_V, h_E, E_idx = model_sc.encode(feature_dict)
    feature_dict["h_V"] = h_V
    feature_dict["h_E"] = h_E
    feature_dict["E_idx"] = E_idx
    for step in range(num_denoising_steps):
        mean, concentration, mix_logits = model_sc.decode(feature_dict)
        mix = D.Categorical(logits=mix_logits)
        comp = D.VonMises(mean, concentration)
        pred_dist = D.MixtureSameFamily(mix, comp)
        predicted_samples = pred_dist.sample([num_samples])
        log_probs_of_samples = pred_dist.log_prob(predicted_samples)
        sample = torch.gather(
            predicted_samples, dim=0, index=torch.argmax(log_probs_of_samples, 0)[None,]
        )[0,]
        torsions_pred_unit = torch.cat(
            [torch.sin(sample[:, :, :, None]), torch.cos(sample[:, :, :, None])], -1
        )
        torsion_dict["torsions_noised"][:, :, 3:] = torsions_pred_unit * torsion_dict[
            "mask_fix_sc"
        ] + torsion_dict["torsions_true"] * (1 - torsion_dict["mask_fix_sc"])
        pred_frames = feats.torsion_angles_to_frames(
            torsion_dict["rigids"],
            torsion_dict["torsions_noised"],
            torsion_dict["aatype"],
            torch.tensor(restype_rigid_group_default_frame, device=device),
        )
        xyz14_noised = feats.frames_and_literature_positions_to_atom14_pos(
            pred_frames,
            torsion_dict["aatype"],
            torch.tensor(restype_rigid_group_default_frame, device=device),
            torch.tensor(restype_atom14_to_rigid_group, device=device),
            torch.tensor(restype_atom14_mask, device=device),
            torch.tensor(restype_atom14_rigid_group_positions, device=device),
        )
        xyz14_noised = xyz14_noised * feature_dict["X_m"][:, :, :, None]
        feature_dict["X"] = xyz14_noised
        S_af2 = torsion_dict["S_af2"]

    feature_dict["X"] = xyz14_noised

    log_prob = pred_dist.log_prob(sample) * torsion_dict["mask_fix_sc"][
        ..., 0
    ] + 2.0 * (1 - torsion_dict["mask_fix_sc"][..., 0])

    tmp_types = torch.tensor(restype_atom14_to_rigid_group, device=device)[S_af2]
    tmp_types[tmp_types < 4] = 4
    tmp_types -= 4
    atom_types_for_b_factor = torch.nn.functional.one_hot(tmp_types, 4)  # [B, L, 14, 4]

    uncertainty = log_prob[:, :, None, :] * atom_types_for_b_factor  # [B,L,14,4]
    b_factor_pred = uncertainty.sum(-1)  # [B, L, 14]
    feature_dict["b_factors"] = b_factor_pred
    feature_dict["mean"] = mean
    feature_dict["concentration"] = concentration
    feature_dict["mix_logits"] = mix_logits
    feature_dict["log_prob"] = log_prob
    feature_dict["sample"] = sample
    feature_dict["true_torsion_sin_cos"] = torsion_dict["torsions_true"]
    return feature_dict


def make_torsion_features(feature_dict, repack_everything=True):
    device = feature_dict["mask"].device

    mask = feature_dict["mask"]
    B, L = mask.shape

    xyz37 = torch.zeros([B, L, 37, 3], device=device, dtype=torch.float32)
    xyz37[:, :, :3] = feature_dict["X"][:, :, :3]
    xyz37[:, :, 4] = feature_dict["X"][:, :, 3]

    S_af2 = torch.argmax(
        torch.nn.functional.one_hot(feature_dict["S"], 21).float()
        @ map_mpnn_to_af2_seq.to(device).float(),
        -1,
    )
    masks14_37 = make_atom14_masks({"aatype": S_af2})
    temp_dict = {
        "aatype": S_af2,
        "all_atom_positions": xyz37,
        "all_atom_mask": masks14_37["atom37_atom_exists"],
    }
    torsion_dict = atom37_to_torsion_angles("")(temp_dict)

    rigids = Rigid.make_transform_from_reference(
        n_xyz=xyz37[:, :, 0, :],
        ca_xyz=xyz37[:, :, 1, :],
        c_xyz=xyz37[:, :, 2, :],
        eps=1e-9,
    )

    if not repack_everything:
        xyz37_true = feature_dict["xyz_37"]
        temp_dict_true = {
            "aatype": S_af2,
            "all_atom_positions": xyz37_true,
            "all_atom_mask": masks14_37["atom37_atom_exists"],
        }
        torsion_dict_true = atom37_to_torsion_angles("")(temp_dict_true)
        torsions_true = torch.clone(torsion_dict_true["torsion_angles_sin_cos"])[
            :, :, 3:
        ]
        mask_fix_sc = feature_dict["chain_mask"][:, :, None, None]
    else:
        torsions_true = torch.zeros([B, L, 4, 2], device=device)
        mask_fix_sc = torch.ones([B, L, 1, 1], device=device)

    random_angle = (
        2 * torch_pi * torch.rand([S_af2.shape[0], S_af2.shape[1], 4], device=device)
    )
    random_sin_cos = torch.cat(
        [torch.sin(random_angle)[..., None], torch.cos(random_angle)[..., None]], -1
    )
    torsions_noised = torch.clone(torsion_dict["torsion_angles_sin_cos"])
    torsions_noised[:, :, 3:] = random_sin_cos * mask_fix_sc + torsions_true * (
        1 - mask_fix_sc
    )
    pred_frames = feats.torsion_angles_to_frames(
        rigids,
        torsions_noised,
        S_af2,
        torch.tensor(restype_rigid_group_default_frame, device=device),
    )
    
    xyz14_noised = feats.frames_and_literature_positions_to_atom14_pos(
        pred_frames,
        S_af2,
        torch.tensor(restype_rigid_group_default_frame, device=device),
        torch.tensor(restype_atom14_to_rigid_group, device=device).long(),
        torch.tensor(restype_atom14_mask, device=device),
        torch.tensor(restype_atom14_rigid_group_positions, device=device),
    )

    xyz14_m = masks14_37["atom14_atom_exists"] * mask[:, :, None]
    xyz14_noised = xyz14_noised * xyz14_m[:, :, :, None]
    torsion_dict["xyz14_m"] = xyz14_m
    torsion_dict["xyz14_noised"] = xyz14_noised
    torsion_dict["mask_for_loss"] = mask
    torsion_dict["rigids"] = rigids
    torsion_dict["torsions_noised"] = torsions_noised
    torsion_dict["mask_fix_sc"] = mask_fix_sc
    torsion_dict["torsions_true"] = torsions_true
    torsion_dict["S_af2"] = S_af2
    return torsion_dict


class Packer(nn.Module):
    def __init__(
        self,
        edge_features=128,
        node_features=128,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        atom37_order=False,
        device=None,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1,
        num_mix=3,
    ):
        super(Packer, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.num_positional_embeddings = num_positional_embeddings
        self.num_chain_embeddings = num_chain_embeddings
        self.num_rbf = num_rbf
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.atom37_order = atom37_order
        self.device = device
        self.atom_context_num = atom_context_num
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.softplus = nn.Softplus(beta=1, threshold=20)

        self.features = ProteinFeatures(
            edge_features=edge_features,
            node_features=node_features,
            num_positional_embeddings=num_positional_embeddings,
            num_chain_embeddings=num_chain_embeddings,
            num_rbf=num_rbf,
            top_k=top_k,
            augment_eps=augment_eps,
            atom37_order=atom37_order,
            device=device,
            atom_context_num=atom_context_num,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_v_sc = nn.Linear(node_features, hidden_dim, bias=True)
        self.linear_down = nn.Linear(2 * hidden_dim, hidden_dim, bias=True)
        self.W_torsions = nn.Linear(hidden_dim, 4 * 3 * num_mix, bias=True)
        self.num_mix = num_mix

        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.W_c = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e_context = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.W_nodes_y = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_edges_y = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.context_encoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(2)]
        )

        self.V_C = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_C_norm = nn.LayerNorm(hidden_dim)
        self.y_context_encoder_layers = nn.ModuleList(
            [DecLayerJ(hidden_dim, hidden_dim, dropout=dropout) for _ in range(2)]
        )

        self.h_V_C_dropout = nn.Dropout(dropout)

        # Decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, feature_dict):
        mask = feature_dict["mask"]
        V, E, E_idx, Y_nodes, Y_edges, E_context, Y_m = self.features.features_encode(
            feature_dict
        )

        h_E_context = self.W_e_context(E_context)
        h_V = self.W_v(V)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_V_C = self.W_c(h_V)
        Y_m_edges = Y_m[:, :, :, None] * Y_m[:, :, None, :]
        Y_nodes = self.W_nodes_y(Y_nodes)
        Y_edges = self.W_edges_y(Y_edges)
        for i in range(len(self.context_encoder_layers)):
            Y_nodes = self.y_context_encoder_layers[i](Y_nodes, Y_edges, Y_m, Y_m_edges)
            h_E_context_cat = torch.cat([h_E_context, Y_nodes], -1)
            h_V_C = self.context_encoder_layers[i](h_V_C, h_E_context_cat, mask, Y_m)

        h_V_C = self.V_C(h_V_C)
        h_V = h_V + self.V_C_norm(self.h_V_C_dropout(h_V_C))

        return h_V, h_E, E_idx

    def decode(self, feature_dict):
        h_V = feature_dict["h_V"]
        h_E = feature_dict["h_E"]
        E_idx = feature_dict["E_idx"]
        mask = feature_dict["mask"]
        device = h_V.device
        V, F = self.features.features_decode(feature_dict)

        h_F = self.W_f(F)
        h_EF = torch.cat([h_E, h_F], -1)

        h_V_sc = self.W_v_sc(V)
        h_V_combined = torch.cat([h_V, h_V_sc], -1)
        h_V = self.linear_down(h_V_combined)

        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_EF, E_idx)
            h_V = layer(h_V, h_EV, mask)

        torsions = self.W_torsions(h_V)
        torsions = torsions.reshape(h_V.shape[0], h_V.shape[1], 4, self.num_mix, 3)
        mean = torsions[:, :, :, :, 0].float()
        concentration = 0.1 + self.softplus(torsions[:, :, :, :, 1]).float()
        mix_logits = torsions[:, :, :, :, 2].float()
        return mean, concentration, mix_logits


class ProteinFeatures(nn.Module):
    def __init__(
        self,
        edge_features=128,
        node_features=128,
        num_positional_embeddings=16,
        num_chain_embeddings=16,
        num_rbf=16,
        top_k=30,
        augment_eps=0.0,
        atom37_order=False,
        device=None,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
    ):
        """Extract protein features"""
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.num_positional_embeddings = num_positional_embeddings
        self.num_chain_embeddings = num_chain_embeddings
        self.num_rbf = num_rbf
        self.top_k = top_k
        self.augment_eps = augment_eps
        self.atom37_order = atom37_order
        self.device = device
        self.atom_context_num = atom_context_num
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # deal with oxygen index
        # ------
        self.N_idx = 0
        self.CA_idx = 1
        self.C_idx = 2

        if atom37_order:
            self.O_idx = 4
        else:
            self.O_idx = 3
        # -------
        self.positional_embeddings = PositionalEncodings(num_positional_embeddings)

        # Features for the encoder
        enc_node_in = 21  # alphabet for the sequence
        enc_edge_in = (
            num_positional_embeddings + num_rbf * 25
        )  # positional + distance features

        self.enc_node_in = enc_node_in
        self.enc_edge_in = enc_edge_in

        self.enc_edge_embedding = nn.Linear(enc_edge_in, edge_features, bias=False)
        self.enc_norm_edges = nn.LayerNorm(edge_features)
        self.enc_node_embedding = nn.Linear(enc_node_in, node_features, bias=False)
        self.enc_norm_nodes = nn.LayerNorm(node_features)

        # Features for the decoder
        dec_node_in = 14 * atom_context_num * num_rbf
        dec_edge_in = num_rbf * 14 * 14 + 42

        self.dec_node_in = dec_node_in
        self.dec_edge_in = dec_edge_in

        self.W_XY_project_down1 = nn.Linear(num_rbf + 120, num_rbf, bias=True)
        self.dec_edge_embedding1 = nn.Linear(dec_edge_in, edge_features, bias=False)
        self.dec_norm_edges1 = nn.LayerNorm(edge_features)
        self.dec_node_embedding1 = nn.Linear(dec_node_in, node_features, bias=False)
        self.dec_norm_nodes1 = nn.LayerNorm(node_features)

        self.node_project_down = nn.Linear(
            5 * num_rbf + 64 + 4, node_features, bias=True
        )
        self.norm_nodes = nn.LayerNorm(node_features)

        self.type_linear = nn.Linear(147, 64)

        self.y_nodes = nn.Linear(147, node_features, bias=False)
        self.y_edges = nn.Linear(num_rbf, node_features, bias=False)

        self.norm_y_edges = nn.LayerNorm(node_features)
        self.norm_y_nodes = nn.LayerNorm(node_features)

        self.periodic_table_features = torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                    95,
                    96,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                [
                    0,
                    1,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    1,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                ],
                [
                    0,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    3,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    6,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                    7,
                ],
            ],
            dtype=torch.long,
            device=device,
        )

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _make_angle_features(self, A, B, C, Y):
        v1 = A - B
        v2 = C - B
        e1 = torch.nn.functional.normalize(v1, dim=-1)
        e1_v2_dot = torch.einsum("bli, bli -> bl", e1, v2)[..., None]
        u2 = v2 - e1 * e1_v2_dot
        e2 = torch.nn.functional.normalize(u2, dim=-1)
        e3 = torch.cross(e1, e2, dim=-1)
        R_residue = torch.cat(
            (e1[:, :, :, None], e2[:, :, :, None], e3[:, :, :, None]), dim=-1
        )

        local_vectors = torch.einsum(
            "blqp, blyq -> blyp", R_residue, Y - B[:, :, None, :]
        )

        rxy = torch.sqrt(local_vectors[..., 0] ** 2 + local_vectors[..., 1] ** 2 + 1e-8)
        f1 = local_vectors[..., 0] / rxy
        f2 = local_vectors[..., 1] / rxy
        rxyz = torch.norm(local_vectors, dim=-1) + 1e-8
        f3 = rxy / rxyz
        f4 = local_vectors[..., 2] / rxyz

        f = torch.cat([f1[..., None], f2[..., None], f3[..., None], f4[..., None]], -1)
        return f

    def _rbf(
        self,
        D,
        D_mu_shape=[1, 1, 1, -1],
        lower_bound=0.0,
        upper_bound=20.0,
        num_bins=16,
    ):
        device = D.device
        D_min, D_max, D_count = lower_bound, upper_bound, num_bins
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view(D_mu_shape)
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(
        self,
        A,
        B,
        E_idx,
        D_mu_shape=[1, 1, 1, -1],
        lower_bound=2.0,
        upper_bound=22.0,
        num_bins=16,
    ):
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[
            :, :, :, 0
        ]  # [B,L,K]
        RBF_A_B = self._rbf(
            D_A_B_neighbors,
            D_mu_shape=D_mu_shape,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            num_bins=num_bins,
        )
        return RBF_A_B

    def features_encode(self, features):
        """
        make protein graph and encode backbone
        """
        S = features["S"]
        X = features["X"]
        Y = features["Y"]
        Y_m = features["Y_m"]
        Y_t = features["Y_t"]
        mask = features["mask"]
        R_idx = features["R_idx"]
        chain_labels = features["chain_labels"]

        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)

        Ca = X[:, :, self.CA_idx, :]
        N = X[:, :, self.N_idx, :]
        C = X[:, :, self.C_idx, :]
        O = X[:, :, self.O_idx, :]

        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca  # shift from CA

        _, E_idx = self._dist(Ca, mask)

        backbone_coords_list = [N, Ca, C, O, Cb]

        RBF_all = []
        for atom_1 in backbone_coords_list:
            for atom_2 in backbone_coords_list:
                RBF_all.append(
                    self._get_rbf(
                        atom_1,
                        atom_2,
                        E_idx,
                        D_mu_shape=[1, 1, 1, -1],
                        lower_bound=self.lower_bound,
                        upper_bound=self.upper_bound,
                        num_bins=self.num_rbf,
                    )
                )
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = R_idx[:, :, None] - R_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        d_chains = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # find self vs non-self interaction
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.positional_embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.enc_edge_embedding(E)
        E = self.enc_norm_edges(E)

        V = torch.nn.functional.one_hot(S, self.enc_node_in).float()
        V = self.enc_node_embedding(V)
        V = self.enc_norm_nodes(V)

        Y_t = Y_t.long()
        Y_t_g = self.periodic_table_features[1][Y_t]  # group; 19 categories including 0
        Y_t_p = self.periodic_table_features[2][Y_t]  # period; 8 categories including 0

        Y_t_g_1hot_ = torch.nn.functional.one_hot(Y_t_g, 19)  # [B, L, M, 19]
        Y_t_p_1hot_ = torch.nn.functional.one_hot(Y_t_p, 8)  # [B, L, M, 8]
        Y_t_1hot_ = torch.nn.functional.one_hot(Y_t, 120)  # [B, L, M, 120]

        Y_t_1hot_ = torch.cat(
            [Y_t_1hot_, Y_t_g_1hot_, Y_t_p_1hot_], -1
        )  # [B, L, M, 147]
        Y_t_1hot = self.type_linear(Y_t_1hot_.float())

        D_N_Y = torch.sqrt(
            torch.sum((N[:, :, None, :] - Y) ** 2, -1) + 1e-6
        )  # [B, L, M, num_bins]
        D_N_Y = self._rbf(
            D_N_Y,
            D_mu_shape=[1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )

        D_Ca_Y = torch.sqrt(
            torch.sum((Ca[:, :, None, :] - Y) ** 2, -1) + 1e-6
        )  # [B, L, M, num_bins]
        D_Ca_Y = self._rbf(
            D_Ca_Y,
            D_mu_shape=[1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )

        D_C_Y = torch.sqrt(
            torch.sum((C[:, :, None, :] - Y) ** 2, -1) + 1e-6
        )  # [B, L, M, num_bins]
        D_C_Y = self._rbf(
            D_C_Y,
            D_mu_shape=[1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )

        D_O_Y = torch.sqrt(
            torch.sum((O[:, :, None, :] - Y) ** 2, -1) + 1e-6
        )  # [B, L, M, num_bins]
        D_O_Y = self._rbf(
            D_O_Y,
            D_mu_shape=[1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )

        D_Cb_Y = torch.sqrt(
            torch.sum((Cb[:, :, None, :] - Y) ** 2, -1) + 1e-6
        )  # [B, L, M, num_bins]
        D_Cb_Y = self._rbf(
            D_Cb_Y,
            D_mu_shape=[1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )

        f_angles = self._make_angle_features(N, Ca, C, Y)

        D_all = torch.cat(
            (D_N_Y, D_Ca_Y, D_C_Y, D_O_Y, D_Cb_Y, Y_t_1hot, f_angles), dim=-1
        )  # [B,L,M,5*num_bins+5]
        E_context = self.node_project_down(D_all)  # [B, L, M, node_features]
        E_context = self.norm_nodes(E_context)

        Y_edges = self._rbf(
            torch.sqrt(
                torch.sum((Y[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
            )
        )  # [B, L, M, M, num_bins]

        Y_edges = self.y_edges(Y_edges)
        Y_nodes = self.y_nodes(Y_t_1hot_.float())

        Y_edges = self.norm_y_edges(Y_edges)
        Y_nodes = self.norm_y_nodes(Y_nodes)

        return V, E, E_idx, Y_nodes, Y_edges, E_context, Y_m

    def features_decode(self, features):
        """
        Make features for decoding. Explicit side chain atom and other atom distances.
        """

        S = features["S"]
        X = features["X"]
        X_m = features["X_m"]
        mask = features["mask"]
        E_idx = features["E_idx"]

        Y = features["Y"][:, :, : self.atom_context_num]
        Y_m = features["Y_m"][:, :, : self.atom_context_num]
        Y_t = features["Y_t"][:, :, : self.atom_context_num]

        X_m = X_m * mask[:, :, None]
        device = S.device

        B, L, _, _ = X.shape

        RBF_sidechain = []
        X_m_gathered = gather_nodes(X_m, E_idx)  # [B, L, K, 14]

        for i in range(14):
            for j in range(14):
                rbf_features = self._get_rbf(
                    X[:, :, i, :],
                    X[:, :, j, :],
                    E_idx,
                    D_mu_shape=[1, 1, 1, -1],
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                    num_bins=self.num_rbf,
                )
                rbf_features = (
                    rbf_features
                    * X_m[:, :, i, None, None]
                    * X_m_gathered[:, :, :, j, None]
                )
                RBF_sidechain.append(rbf_features)

        D_XY = torch.sqrt(
            torch.sum((X[:, :, :, None, :] - Y[:, :, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, L, 14, atom_context_num]
        XY_features = self._rbf(
            D_XY,
            D_mu_shape=[1, 1, 1, 1, -1],
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            num_bins=self.num_rbf,
        )  # [B, L, 14, atom_context_num, num_rbf]
        XY_features = XY_features * X_m[:, :, :, None, None] * Y_m[:, :, None, :, None]

        Y_t_1hot = torch.nn.functional.one_hot(
            Y_t.long(), 120
        ).float()  # [B, L, atom_context_num, 120]
        XY_Y_t = torch.cat(
            [XY_features, Y_t_1hot[:, :, None, :, :].repeat(1, 1, 14, 1, 1)], -1
        )  # [B, L, 14, atom_context_num, num_rbf+120]
        XY_Y_t = self.W_XY_project_down1(
            XY_Y_t
        )  # [B, L, 14, atom_context_num, num_rbf]
        XY_features = XY_Y_t.view([B, L, -1])

        V = self.dec_node_embedding1(XY_features)
        V = self.dec_norm_nodes1(V)

        S_1h = torch.nn.functional.one_hot(S, self.enc_node_in).float()
        S_1h_gathered = gather_nodes(S_1h, E_idx)  # [B, L, K, 21]
        S_features = torch.cat(
            [S_1h[:, :, None, :].repeat(1, 1, E_idx.shape[2], 1), S_1h_gathered], -1
        )  # [B, L, K, 42]

        F = torch.cat(
            tuple(RBF_sidechain), dim=-1
        )  # [B,L,atom_context_num,14*14*num_rbf]
        F = torch.cat([F, S_features], -1)
        F = self.dec_edge_embedding1(F)
        F = self.dec_norm_edges1(F)
        return V, F
