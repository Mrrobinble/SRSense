from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActorCritic(nn.Module):
    def __init__(
        self,
        node_num: int,
        edge_num: int,
        action_size: int,
        max_edges: int = 4,
        node_embedding_dim: int = 16,
        edge_embedding_dim: int = 8,
        fingerprint_dim: int = 256,
        fusion_dim: int = 128,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.node_num = int(node_num)
        self.edge_num = int(edge_num)
        self.action_size = int(action_size)
        self.max_edges = int(max_edges)
        self.pad_edge_id = self.edge_num

        self.node_embedding = nn.Embedding(self.node_num, node_embedding_dim)
        self.edge_embedding = nn.Embedding(self.edge_num + 1, edge_embedding_dim, padding_idx=self.pad_edge_id)

        self.signal_encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.packet_fc = nn.Linear(128, 96)
        self.rsrp_fc = nn.Linear(1, 16)
        self.packet_gru = nn.GRU(112, 64, batch_first=True, bidirectional=True)
        self.packet_attn_fc = nn.Linear(128, 1)
        self.edge_gru = nn.GRU(256, 96, batch_first=True, bidirectional=True)
        self.edge_attn_fc = nn.Linear(192, 1)
        self.fingerprint_fc = nn.Linear(384, fingerprint_dim)

        total_dim = node_embedding_dim + self.max_edges * edge_embedding_dim + fingerprint_dim
        self.fusion_fc = nn.Linear(total_dim, fusion_dim)
        self.hidden_fc = nn.Linear(fusion_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, self.action_size)
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)

    def encode_fingerprint(
        self,
        signal_state: torch.Tensor,
        rsrp_state: torch.Tensor,
        edge_history: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_edges, segment_length, channels, sequence_length = signal_state.shape
        edge_mask = edge_history.ne(self.pad_edge_id).float()
        has_edge = edge_mask.sum(dim=1, keepdim=True).gt(0)
        x = signal_state.reshape(batch_size * max_edges * segment_length, channels, sequence_length)
        x = self.signal_encoder(x)
        avg = x.mean(dim=-1)
        mx = x.amax(dim=-1)
        x = torch.cat([avg, mx], dim=-1)
        x = F.relu(self.packet_fc(x))
        x = x.reshape(batch_size * max_edges, segment_length, 96)

        r = F.relu(self.rsrp_fc(rsrp_state.reshape(batch_size * max_edges, segment_length, 1)))
        packet_seq = torch.cat([x, r], dim=-1)
        packet_out, _ = self.packet_gru(packet_seq)
        packet_score = self.packet_attn_fc(packet_out).squeeze(-1)
        packet_weight = torch.softmax(packet_score, dim=1)
        packet_attended = (packet_out * packet_weight.unsqueeze(-1)).sum(dim=1)
        packet_mean = packet_out.mean(dim=1)
        edge_repr = torch.cat([packet_attended, packet_mean], dim=-1).reshape(batch_size, max_edges, 256)

        edge_repr = edge_repr * edge_mask.unsqueeze(-1)
        edge_out, _ = self.edge_gru(edge_repr)
        edge_out = edge_out * edge_mask.unsqueeze(-1)
        safe_edge_mask = edge_mask.clone()
        safe_edge_mask[:, 0] = torch.where(has_edge.squeeze(1), safe_edge_mask[:, 0], torch.ones_like(safe_edge_mask[:, 0]))
        edge_score = self.edge_attn_fc(edge_out).squeeze(-1).masked_fill(safe_edge_mask <= 0, -1e9)
        edge_weight = torch.softmax(edge_score, dim=1)
        edge_attended = (edge_out * edge_weight.unsqueeze(-1)).sum(dim=1)
        denom = edge_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        edge_mean = edge_out.sum(dim=1) / denom
        feature = F.relu(self.fingerprint_fc(torch.cat([edge_attended, edge_mean], dim=-1)))
        return torch.where(has_edge, feature, torch.zeros_like(feature))
