from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_agent import PPOActorCritic


class MultiTargetPPOActorCritic(PPOActorCritic):
    def __init__(
        self,
        node_num: int,
        edge_num: int,
        action_size: int,
        max_edges: int = 4,
        max_targets: int = 5,
    ):
        super().__init__(
            node_num=node_num,
            edge_num=edge_num,
            action_size=action_size,
            max_edges=max_edges,
        )
        self.max_targets = int(max_targets)

        node_embedding_dim = self.node_embedding.embedding_dim
        edge_embedding_dim = self.edge_embedding.embedding_dim
        fingerprint_dim = self.fingerprint_fc.out_features
        fusion_dim = self.fusion_fc.out_features
        hidden_dim = self.hidden_fc.out_features

        total_dim = node_embedding_dim + (self.max_edges * edge_embedding_dim) + fingerprint_dim
        self.fusion_fc = nn.Linear(total_dim, fusion_dim)
        self.hidden_fc = nn.Linear(fusion_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, self.action_size)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def forward(
        self,
        current_node: torch.Tensor,
        edge_history: torch.Tensor,
        signal_state: torch.Tensor,
        rsrp_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_node = current_node.long().view(-1)
        edge_history = edge_history.long()
        signal_state = signal_state.float()
        rsrp_state = rsrp_state.float()

        node_emb = self.node_embedding(current_node)
        edge_emb = self.edge_embedding(edge_history).reshape(edge_history.shape[0], -1)
        fingerprint_feature = self.encode_multi_target_fingerprint(signal_state, rsrp_state, edge_history)

        z = torch.cat([node_emb, edge_emb, fingerprint_feature], dim=1)
        z = F.relu(self.fusion_fc(z))
        z = F.relu(self.hidden_fc(z))

        logits = self.actor_head(z)
        value = self.critic_head(z).squeeze(-1)
        return logits, value

    @staticmethod
    def mask_logits(logits: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(legal_action_mask <= 0, -1e9)

    def act(
        self,
        current_node: torch.Tensor,
        edge_history: torch.Tensor,
        signal_state: torch.Tensor,
        rsrp_state: torch.Tensor,
        legal_action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(
            current_node,
            edge_history,
            signal_state,
            rsrp_state,
        )
        logits = self.mask_logits(logits, legal_action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(
        self,
        current_node: torch.Tensor,
        edge_history: torch.Tensor,
        signal_state: torch.Tensor,
        rsrp_state: torch.Tensor,
        legal_action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(
            current_node,
            edge_history,
            signal_state,
            rsrp_state,
        )
        logits = self.mask_logits(logits, legal_action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions.long())
        entropy = dist.entropy()
        return log_prob, entropy, value

    def encode_multi_target_fingerprint(
        self,
        signal_state: torch.Tensor,
        rsrp_state: torch.Tensor,
        edge_history: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, target_count, max_edges, segment_length, channels, sequence_length = signal_state.shape
        flat_signal = signal_state.reshape(batch_size * target_count, max_edges, segment_length, channels, sequence_length)
        flat_rsrp = rsrp_state.reshape(batch_size * target_count, max_edges, segment_length, 1)
        flat_edge_history = edge_history.unsqueeze(1).expand(-1, target_count, -1).reshape(batch_size * target_count, max_edges)

        feature = self.encode_fingerprint(flat_signal, flat_rsrp, flat_edge_history)
        feature = feature.reshape(batch_size, target_count, -1)
        return feature.mean(dim=1)
