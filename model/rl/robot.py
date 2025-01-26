import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import global_max_pool
from rl.net import NeuralNetworkModule
from model.resnet import ResnetBlockFC
from model.rl.encoder import RLEncoder
from model.rl.decoder import RLDecoder


class RobotEncoder(nn.Module):
    def __init__(self, normalize_radius: float, grid_resolution: int = 64):
        super(RobotEncoder, self).__init__()
        self.normalize_radius = normalize_radius
        self.voxel_encoder_net = RLEncoder(
            f_dim=4,
            c_dim=64,
            hidden_dim=64,
            scatter_type="max",
            grid_resolution=grid_resolution,
            padding=0.02,
            unet3d_kwargs={"layer_order": "cl", "num_levels": 2},
        )
        self.voxel_decoder_net = RLDecoder(
            f_dim=64,
            c_dim=64,
            hidden_dim=64,
            grid_resolution=grid_resolution,
            sample_mode="bilinear",
            padding=0.02,
        )
        self.kinematic_encoder_conv1 = TransformerConv(
            in_channels=78, out_channels=256, edge_dim=9
        )
        self.kinematic_encoder_conv2 = TransformerConv(
            in_channels=256, out_channels=512, edge_dim=9
        )
        self.kinematic_encoder_conv3 = TransformerConv(
            in_channels=512, out_channels=512, edge_dim=9
        )
        self.mask_voxel = False
        self.mask_node = False
        self.mask_edge = False

        self.voxel_noise_rate = 0
        self.node_noise_rate = 0
        self.edge_noise_rate = 0

    def set_mask(self, mask_voxel: bool, mask_node: bool, mask_edge: bool):
        self.mask_voxel = mask_voxel
        self.mask_node = mask_node
        self.mask_edge = mask_edge

    def set_noise_rate(
        self, voxel_noise_rate: float, node_noise_rate: float, edge_noise_rate: float
    ):
        self.voxel_noise_rate = voxel_noise_rate
        self.node_noise_rate = node_noise_rate
        self.edge_noise_rate = edge_noise_rate

    def forward(
        self,
        com,
        voxel_positions,
        voxel_features,
        node_positions,
        node_features,
        edges,
        edge_features,
    ):
        """
        Note:
            Node num N* corresponds to rigid body num, variable across batch
            Edge num E* corresponds to joint num, variable across batch
            There are E* * 2 edges for edges in both directions
        Args:
            com: List of length B, inner shape [1, 3]
            voxel_positions: List of length B, inner shape [voxel_num, 3],
            voxel_features: List of length B, inner shape [voxel_num, 4]
            node_positions: List of length B, inner shape [N*, 3]
            node_features: List of length B, inner shape [N*, 14]
            edges: List of length B, inner shape [2, E* * 2]
            edge_features: List of length B, inner shape [E* * 2, 9]
        Returns:
            out_node_features: Shape [N_0 + N_1 + ... + N_(B-1), 256]
            out_pooled_features: Shape [B, 512]
            out_node_batch: Index tensor, Shape [N_0 + N_1 + ... + N_(B-1)]
            out_edge_batch: Index tensor, Shape [(E_0 + E_1 + ... + E_(B-1)) * 2]
            out_edges: Shape [2, (E_0 + E_1 + ... + E_(B-1)) * 2]
            out_unique_edge_batch: Index tensor, Shape [E_0 + E_1 + ... + E_(B-1)]
            out_unique_edges: Shape [2, E_0 + E_1 + ... + E_(B-1)]
        """
        if self.voxel_noise_rate > 0:
            voxel_features = [
                t.where(t.rand_like(v) < self.edge_noise_rate, t.randn_like(v), v)
                for v in voxel_features
            ]
        if self.node_noise_rate > 0:
            node_features = [
                t.where(t.rand_like(n) < self.edge_noise_rate, t.randn_like(n), n)
                for n in node_features
            ]
        if self.edge_noise_rate > 0:
            edge_features = [
                t.where(t.rand_like(e) < self.edge_noise_rate, t.randn_like(e), e)
                for e in edge_features
            ]

        if self.mask_voxel:
            voxel_features = [t.zeros_like(v) for v in voxel_features]
        if self.mask_node:
            node_features = [t.zeros_like(n) for n in node_features]
        if self.mask_edge:
            edge_features = [t.zeros_like(e) for e in edge_features]

        com = t.concatenate(com, dim=0)
        voxel_positions = t.stack(voxel_positions, dim=0)
        voxel_features = t.stack(voxel_features, dim=0)
        device = com.device
        norm_voxel_positions = (
            (voxel_positions - com.unsqueeze(1)) / 2 / self.normalize_radius
        ) + 0.5
        latent = self.voxel_encoder_net(norm_voxel_positions, voxel_features)

        node_batch = []
        node_num = []
        max_node_num = 0
        for i in range(len(node_positions)):
            node_batch += [i] * len(node_positions[i])
            node_num.append(len(node_positions[i]))
            max_node_num = max(max_node_num, len(node_positions[i]))
        node_batch = t.tensor(node_batch, dtype=t.long, device=device)
        norm_node_positions_batch = t.full(
            [com.shape[0], max_node_num, 3],
            0.5,
            dtype=voxel_positions.dtype,
            device=device,
        )
        for i in range(len(node_positions)):
            norm_node_positions_batch[i, : len(node_positions[i])] += (
                (node_positions[i] - com[i]) / 2 / self.normalize_radius
            )
        # Add offsets to edge indices
        edge_indices = []
        unique_edge_indices = []
        edge_batch = []
        unique_edge_batch = []
        offset = 0
        for i, (e, n_num) in enumerate(zip(edges, node_num)):
            edge_batch += [i] * e.shape[1]
            edge_indices.append(e + offset)
            unique_edge_batch += [i] * (e.shape[1] // 2)
            unique_edge_indices.append((e + offset)[:, : e.shape[1] // 2])
            offset += n_num
        edge_batch = t.tensor(edge_batch, dtype=t.long, device=device)
        edge_indices = t.concatenate(edge_indices, dim=1)
        unique_edge_batch = t.tensor(unique_edge_batch, dtype=t.long, device=device)
        unique_edge_indices = t.concatenate(unique_edge_indices, dim=1)
        edge_features = t.concatenate(edge_features, dim=0)

        node_voxel_features_batch = self.voxel_decoder_net(
            norm_node_positions_batch, latent
        )
        node_voxel_features = t.concatenate(
            [node_voxel_features_batch[i, :num] for i, num in enumerate(node_num)],
            dim=0,
        )
        # Shape [N_0 + N_1 + ... N_(B-1), 64 + 14 = 78]
        node_features = t.concatenate(
            (
                node_voxel_features,
                t.concatenate(node_features, dim=0).to(device),
            ),
            dim=1,
        )
        out1 = self.kinematic_encoder_conv1(node_features, edge_indices, edge_features)
        out2 = F.leaky_relu(
            self.kinematic_encoder_conv2(
                F.leaky_relu(out1, negative_slope=0.2), edge_indices, edge_features
            ),
            negative_slope=0.2,
        )
        out2 = self.kinematic_encoder_conv3(out2, edge_indices, edge_features)
        pooled_out = global_max_pool(out2, node_batch, size=com.shape[0])
        return (
            out1,
            pooled_out,
            node_batch,
            edge_batch,
            edge_indices,
            unique_edge_batch,
            unique_edge_indices,
        )


class RobotActor(NeuralNetworkModule):
    def __init__(
        self,
        encoder: RobotEncoder,
        freeze_encoder: bool = False,
        resolution: int = 5,
    ):
        super().__init__()
        self.resolution = resolution
        self.encoder = encoder if not freeze_encoder else [encoder]
        self.action_resnet = nn.Sequential(
            ResnetBlockFC(1024, 256),
            ResnetBlockFC(256, 64),
            ResnetBlockFC(64, resolution),
        )
        self.freeze_encoder = freeze_encoder
        self.set_input_module(self.action_resnet)
        self.set_output_module(self.action_resnet)

    def forward(
        self,
        com,
        voxel_positions,
        voxel_features,
        node_positions,
        node_features,
        edges,
        edge_features,
        action=None,
    ):
        batch_size = len(com)
        if self.freeze_encoder:
            with t.no_grad():
                (
                    out_node_features,
                    out_pooled_features,
                    out_node_batch,
                    out_edge_batch,
                    out_edges,
                    out_unique_edge_batch,
                    out_unique_edges,
                ) = self.encoder[0](
                    com,
                    voxel_positions,
                    voxel_features,
                    node_positions,
                    node_features,
                    edges,
                    edge_features,
                )
        else:
            (
                out_node_features,
                out_pooled_features,
                out_node_batch,
                out_edge_batch,
                out_edges,
                out_unique_edge_batch,
                out_unique_edges,
            ) = self.encoder(
                com,
                voxel_positions,
                voxel_features,
                node_positions,
                node_features,
                edges,
                edge_features,
            )

        # Shape [E_0 + E_1 + ... + E_(B-1), 256]
        edge_total = out_unique_edges.shape[1]
        first_node_features = t.gather(
            out_node_features,
            dim=0,
            index=out_unique_edges[0].unsqueeze(1).expand(edge_total, 256),
        )
        second_node_features = t.gather(
            out_node_features,
            dim=0,
            index=out_unique_edges[1].unsqueeze(1).expand(edge_total, 256),
        )
        node_pooled_features = t.gather(
            out_pooled_features,
            dim=0,
            index=out_unique_edge_batch.unsqueeze(1).expand(
                out_unique_edge_batch.shape[0], 512
            ),
        )

        edge_features = t.cat(
            (first_node_features, second_node_features, node_pooled_features), dim=1
        )
        action_logits = self.action_resnet(edge_features)
        dist = Categorical(logits=action_logits)
        raw_action = dist.sample() if action is None else t.concatenate(action)
        raw_log_prob = dist.log_prob(raw_action)
        raw_entropy = dist.entropy()

        # Now convert from concatenated batch to list batch
        action = []
        log_prob = []
        entropy = []
        for idx in range(batch_size):
            action.append(raw_action[out_unique_edge_batch == idx])
            log_prob.append(t.sum(raw_log_prob[out_unique_edge_batch == idx]))
            entropy.append(t.sum(raw_entropy[out_unique_edge_batch == idx]))
        log_prob = t.stack(log_prob)
        entropy = t.stack(entropy)
        return action, log_prob, entropy


class RobotCritic(NeuralNetworkModule):
    def __init__(
        self,
        encoder: RobotEncoder,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder if not freeze_encoder else [encoder]
        self.value_resnet = nn.Sequential(
            ResnetBlockFC(512, 128), ResnetBlockFC(128, 32), ResnetBlockFC(32, 1)
        )
        self.freeze_encoder = freeze_encoder
        self.set_input_module(self.value_resnet)
        self.set_output_module(self.value_resnet)

    def forward(
        self,
        com,
        voxel_positions,
        voxel_features,
        node_positions,
        node_features,
        edges,
        edge_features,
    ):
        if self.freeze_encoder:
            with t.no_grad():
                _, out_pooled_features, *__ = self.encoder[0](
                    com,
                    voxel_positions,
                    voxel_features,
                    node_positions,
                    node_features,
                    edges,
                    edge_features,
                )
        else:
            _, out_pooled_features, *__ = self.encoder(
                com,
                voxel_positions,
                voxel_features,
                node_positions,
                node_features,
                edges,
                edge_features,
            )

        value = self.value_resnet(out_pooled_features)
        return value.view(-1)


class RobotContinuousActor(NeuralNetworkModule):
    def __init__(self, encoder: RobotEncoder, freeze_encoder: bool = False):
        super().__init__()
        self.encoder = encoder if not freeze_encoder else [encoder]
        self.action_resnet = nn.Sequential(
            ResnetBlockFC(1024, 256),
            ResnetBlockFC(256, 64),
            ResnetBlockFC(64, 2),
        )
        self.freeze_encoder = freeze_encoder
        self.set_input_module(self.action_resnet)
        self.set_output_module(self.action_resnet)

    def forward(
        self,
        com,
        voxel_positions,
        voxel_features,
        node_positions,
        node_features,
        edges,
        edge_features,
        action=None,
    ):
        batch_size = len(com)
        if self.freeze_encoder:
            with t.no_grad():
                (
                    out_node_features,
                    out_pooled_features,
                    out_node_batch,
                    out_edge_batch,
                    out_edges,
                    out_unique_edge_batch,
                    out_unique_edges,
                ) = self.encoder[0](
                    com,
                    voxel_positions,
                    voxel_features,
                    node_positions,
                    node_features,
                    edges,
                    edge_features,
                )
        else:
            (
                out_node_features,
                out_pooled_features,
                out_node_batch,
                out_edge_batch,
                out_edges,
                out_unique_edge_batch,
                out_unique_edges,
            ) = self.encoder(
                com,
                voxel_positions,
                voxel_features,
                node_positions,
                node_features,
                edges,
                edge_features,
            )

        # Shape [E_0 + E_1 + ... + E_(B-1), 256]
        edge_total = out_unique_edges.shape[1]
        first_node_features = t.gather(
            out_node_features,
            dim=0,
            index=out_unique_edges[0].unsqueeze(1).expand(edge_total, 256),
        )
        second_node_features = t.gather(
            out_node_features,
            dim=0,
            index=out_unique_edges[1].unsqueeze(1).expand(edge_total, 256),
        )
        node_pooled_features = t.gather(
            out_pooled_features,
            dim=0,
            index=out_unique_edge_batch.unsqueeze(1).expand(
                out_unique_edge_batch.shape[0], 512
            ),
        )

        edge_features = t.cat(
            (first_node_features, second_node_features, node_pooled_features), dim=1
        )

        mu, logvar = t.split(self.action_resnet(edge_features), 1, dim=1)
        mu = mu.squeeze(1)
        logvar = logvar.squeeze(1)
        dist = Normal(mu, t.exp(0.5 * logvar))
        raw_action = dist.sample() if action is None else t.concatenate(action)
        raw_log_prob = dist.log_prob(raw_action)
        raw_entropy = dist.entropy()

        # Now convert from concatenated batch to list batch
        action = []
        log_prob = []
        entropy = []
        for idx in range(batch_size):
            action.append(raw_action[out_unique_edge_batch == idx])
            log_prob.append(t.sum(raw_log_prob[out_unique_edge_batch == idx]))
            entropy.append(t.sum(raw_entropy[out_unique_edge_batch == idx]))
        log_prob = t.stack(log_prob)
        entropy = t.stack(entropy)
        return action, log_prob, entropy


RobotContinuousCritic = RobotCritic
