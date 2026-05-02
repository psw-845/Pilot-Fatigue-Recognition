import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict



class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, :x.size(2)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalAttentionPool(nn.Module):

    def __init__(self, D: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(D, max(D // 4, 1), kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(max(D // 4, 1), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(x).squeeze(1), dim=-1)
        vec  = (attn.unsqueeze(1) * x).sum(dim=-1)
        return vec, attn




class LearnableMissingToken(nn.Module):
    def __init__(self, D: int, modalities: List[str]):
        super().__init__()
        self.tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(D) * 0.02)
            for m in modalities
        })

    def get(self, name: str, W: int,
            device: torch.device) -> torch.Tensor:
        tok = self.tokens[name].to(device)
        return tok.unsqueeze(0).expand(W, -1).clone()




class EEGBranch(nn.Module):

    def __init__(self, C_eeg: int = 42, D: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(C_eeg, C_eeg, kernel_size=7,
                      padding=3, groups=C_eeg),
            nn.Conv1d(C_eeg, 64, kernel_size=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=5,
                      padding=2, groups=64),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
            nn.Conv1d(128, D, kernel_size=3, padding=1),
            nn.BatchNorm1d(D), nn.GELU(),
        )
        self.pool = TemporalAttentionPool(D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.pool(self.cnn(x))
        return self.norm(h)


class EyeBranch(nn.Module):
    def __init__(self, C_eye: int = 11, D: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(C_eye, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),
            nn.Conv1d(32, D, kernel_size=3, padding=1),
            nn.BatchNorm1d(D), nn.GELU(),
        )
        self.pool = TemporalAttentionPool(D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.pool(self.cnn(x))
        return self.norm(h)


class FNIRSBranch(nn.Module):

    def __init__(self, C_fnirs: int = 22, D: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            CausalConv1d(C_fnirs, 32, 3), nn.BatchNorm1d(32), nn.GELU(),
            CausalConv1d(32,      64, 3), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(0.2),
            CausalConv1d(64,       D, 3), nn.BatchNorm1d(D),  nn.GELU(),
        )
        self.hrf_pool = TemporalAttentionPool(D)
        self.norm     = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        h, _ = self.hrf_pool(feat)
        return self.norm(h)



class AFDModule(nn.Module):
    def __init__(self, D: int = 64, n_modalities: int = 3):
        super().__init__()
        self.D     = D
        self.n_mod = n_modalities
        self.gate_mlp = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, D // 2),
                nn.GELU(),
                nn.Linear(D // 2, 1),
            )
            for _ in range(n_modalities)
        ])
        self.proj = nn.Sequential(
            nn.Linear(D * n_modalities, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = torch.cat(
            [self.gate_mlp[i](feats[i]) for i in range(self.n_mod)],
            dim=-1)
        modal_attn = torch.softmax(gates, dim=-1)
        weighted   = [modal_attn[:, i:i+1] * feats[i]
                      for i in range(self.n_mod)]
        fused = self.proj(torch.cat(weighted, dim=-1))
        return fused, modal_attn


class WindowAttentionPool(nn.Module):
    def __init__(self, D: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(x).squeeze(-1), dim=0)
        vec  = (attn.unsqueeze(-1) * x).sum(0)
        return vec, attn



class DisentangleEncoder(nn.Module):
    def __init__(self, D: int = 64):
        super().__init__()
        Dh = D // 2
        self.fatigue_mu    = nn.Linear(D, Dh)
        self.fatigue_logv  = nn.Linear(D, Dh)
        self.workload_mu   = nn.Linear(D, Dh)
        self.workload_logv = nn.Linear(D, Dh)
        self.workload_reg  = nn.Sequential(
            nn.Linear(Dh, Dh // 2),
            nn.GELU(),
            nn.Linear(Dh // 2, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def _kl(self, mu: torch.Tensor,
            logv: torch.Tensor) -> torch.Tensor:
        # mean 而非 sum，不随维度数缩放
        return -0.5 * (1 + logv - mu.pow(2) - logv.exp()).mean()

    def forward(
        self, stage_vec: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        f_mu   = self.fatigue_mu(stage_vec)
        f_logv = self.fatigue_logv(stage_vec)
        z_f    = self.reparameterize(f_mu, f_logv)

        w_mu   = self.workload_mu(stage_vec)
        w_logv = self.workload_logv(stage_vec)
        z_w    = self.reparameterize(w_mu, w_logv)

        return {
            'z_fatigue':         z_f,
            'z_workload':        z_w,
            'kl_fatigue':        self._kl(f_mu, f_logv),
            'kl_workload':       self._kl(w_mu, w_logv),
            'workload_progress': self.workload_reg(z_w).squeeze(-1),
        }


class OrthogonalConstraint(nn.Module):
    def forward(self, z_f: torch.Tensor,
                z_w: torch.Tensor) -> torch.Tensor:
        N  = z_f.size(0)
        Dh = z_f.size(1)
        cross = (z_f.T @ z_w) / N
        return cross.pow(2).sum() / (Dh * Dh)



class CausalStageTransformer(nn.Module):
    def __init__(self, D: int = 32, nhead: int = 4,
                 num_layers: int = 2):
        super().__init__()
        self.pos_enc = PositionalEncoding(D, max_len=16)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=nhead,
            dim_feedforward=D * 4,
            dropout=0.1, batch_first=True,
            activation='gelu', norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers)
        self.encoder.enable_nested_tensor = False

    def forward(self, z_seq: torch.Tensor,
                use_causal_mask: bool = True) -> torch.Tensor:
        S = z_seq.size(0)
        x = self.pos_enc(z_seq.unsqueeze(0))
        mask = None
        if use_causal_mask and S > 1:
            mask = torch.triu(
                torch.full((S, S), float('-inf'),
                           device=x.device), diagonal=1)
        return self.encoder(x, mask=mask).squeeze(0)


class FatigueProcessPrototype(nn.Module):
    def __init__(self, D: int = 32, n_stages: int = 9,
                 tau: float = 0.1):
        super().__init__()
        self.n_stages  = n_stages
        self.tau       = tau
        self.prototype = nn.Parameter(
            torch.randn(2, n_stages, D) * 0.02)
        self.proj      = nn.Linear(D, D)

    def _soft_align_sim(self, z_seq: torch.Tensor,
                        proto: torch.Tensor) -> torch.Tensor:
        z_n     = F.normalize(z_seq,  dim=-1)
        p_n     = F.normalize(proto,  dim=-1)
        sim_mat = z_n @ p_n.T
        soft_max = (sim_mat *
                    torch.softmax(sim_mat / self.tau,
                                  dim=-1)).sum(dim=-1)
        return soft_max.mean()

    def forward(
        self, z_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q     = self.proj(z_seq)
        S     = q.size(0)
        proto = self.prototype[:, :S, :]

        logits = torch.stack([
            self._soft_align_sim(q, proto[0]),
            self._soft_align_sim(q, proto[1]),
        ])

        attns = []
        for i in range(2):
            p_n  = F.normalize(proto[i], dim=-1)
            q_n  = F.normalize(q, dim=-1)
            attn = torch.softmax(
                (q_n @ p_n.T).mean(0), dim=0)
            attns.append(attn)
        proto_attn = torch.stack(attns)

        return logits, proto_attn



class TPFN(nn.Module):
    def __init__(
        self,
        C_eeg:    int   = 42,   T_eeg:   int   = 2000,
        C_eye:    int   = 11,   T_eye:   int   = 240,
        C_fnirs:  int   = 22,   T_fnirs: int   = 122,
        D:        int   = 64,
        D_z:      int   = 32,
        n_stages: int   = 9,
        dropout:  float = 0.4,
    ):
        super().__init__()
        assert D_z == D // 2,
        self.D        = D
        self.D_z      = D_z
        self.n_stages = n_stages

        self.eeg_branch   = EEGBranch(C_eeg,    D)
        self.eye_branch   = EyeBranch(C_eye,    D)
        self.fnirs_branch = FNIRSBranch(C_fnirs, D)

        self.missing_token = LearnableMissingToken(
            D, ['eeg', 'eye', 'fnirs'])
        self.afd           = AFDModule(D, n_modalities=3)
        self.win_pool      = WindowAttentionPool(D)
        self.disentangle   = DisentangleEncoder(D)
        self.ortho_loss    = OrthogonalConstraint()
        self.causal_tf     = CausalStageTransformer(
            D_z, nhead=4, num_layers=2)
        self.prototype_cls = FatigueProcessPrototype(
            D_z, n_stages)

        self.stage_score_head = nn.Sequential(
            nn.LayerNorm(D_z),
            nn.Linear(D_z, 1),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std  = x.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (x - mean) / std

    def encode_stage(
        self,
        eeg:    Optional[torch.Tensor],
        eye:    Optional[torch.Tensor],
        fnirs:  Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[Dict]:
        W = None
        for x in [eeg, eye, fnirs]:
            if x is not None:
                W = x.shape[0]
                break
        if W is None:
            return None
        encoded = {}
        if eeg is not None:
            encoded['eeg'] = self.eeg_branch(
                 eeg.to(device))
        if eye is not None:
            encoded['eye'] = self.eye_branch(
                 eye.to(device))
        if fnirs is not None:
            encoded['fnirs'] = self.fnirs_branch(
                fnirs.to(device))

        if not encoded:
            return None

        W = min(f.size(0) for f in encoded.values())

        feat_eeg = (encoded['eeg'][:W]
                    if 'eeg' in encoded
                    else self.missing_token.get('eeg', W, device))
        feat_eye = (encoded['eye'][:W]
                    if 'eye' in encoded
                    else self.missing_token.get('eye', W, device))
        feat_fnirs = (encoded['fnirs'][:W]
                      if 'fnirs' in encoded
                      else self.missing_token.get('fnirs', W, device))

        fused, modal_attn   = self.afd([feat_eeg, feat_eye, feat_fnirs])
        fused               = self.dropout(fused)
        stage_vec, win_attn = self.win_pool(fused)
        dis                 = self.disentangle(stage_vec)

        return {
            'stage_vec':         stage_vec,
            'z_fatigue':         dis['z_fatigue'],
            'z_workload':        dis['z_workload'],
            'kl_fatigue':        dis['kl_fatigue'],
            'kl_workload':       dis['kl_workload'],
            'workload_progress': dis['workload_progress'],
            'modal_attn':        modal_attn,
            'win_attn':          win_attn,
        }

    def forward(
        self, session, device: torch.device
    ) -> Optional[Dict]:
        def _t(x):
            return torch.as_tensor(x, dtype=torch.float32) \
                   if x is not None else None

        stage_results  = []
        z_fatigue_seq  = []
        z_workload_seq = []
        stage_progress = []

        for stage in session.stages:
            res = self.encode_stage(
                _t(stage.eeg), _t(stage.eye),
                _t(stage.fnirs), device)
            if res is None:
                continue
            stage_results.append(res)
            z_fatigue_seq.append(res['z_fatigue'])
            z_workload_seq.append(res['z_workload'])
            prog = stage.stage_index / max(
                len(session.stages) - 1, 1)
            stage_progress.append(prog)

        if not z_fatigue_seq:
            return None

        N          = len(z_fatigue_seq)
        z_f_tensor = torch.stack(z_fatigue_seq)
        z_ctx      = self.causal_tf(z_f_tensor)

        logits, proto_attn = self.prototype_cls(z_ctx)
        stage_scores = torch.sigmoid(
            self.stage_score_head(z_ctx).squeeze(-1))

        Z_f   = torch.stack(z_fatigue_seq)
        Z_w   = torch.stack(z_workload_seq)
        ortho = self.ortho_loss(Z_f, Z_w)

        workload_pred = torch.stack(
            [r['workload_progress'] for r in stage_results])
        kl_fatigue  = sum(r['kl_fatigue']
                          for r in stage_results)
        kl_workload = sum(r['kl_workload']
                          for r in stage_results)
        modal_attn  = torch.stack(
            [r['modal_attn'].mean(0) for r in stage_results])

        return {
            'logits': logits,
            'stage_scores': stage_scores,
            'workload_pred': workload_pred,
            'workload_target': torch.tensor(
                stage_progress, dtype=torch.float32,
                device=device),

            'kl_fatigue': kl_fatigue,
            'kl_workload': kl_workload,
            'ortho_loss': ortho,
            'modal_attn': modal_attn,
            'proto_attn': proto_attn,

            'z_f': Z_f,

            'z_w': Z_w,

            'z_fatigue_seq': z_ctx,

            'z_fatigue_raw_seq': Z_f,
            'z_workload_seq': Z_w,

            'label': torch.tensor(
                session.label, dtype=torch.long,
                device=device),
            'session_id': session.session_id,
            'n_stages': N,
        }
    @torch.no_grad()
    def realtime_predict(self, partial_stages,
                         device: torch.device) -> Dict:
        self.eval()

        def _t(x):
            return torch.as_tensor(x, dtype=torch.float32) \
                   if x is not None else None

        z_seq = []
        for stage in partial_stages:
            res = self.encode_stage(
                _t(stage.eeg), _t(stage.eye),
                _t(stage.fnirs), device)
            if res is not None:
                z_seq.append(res['z_fatigue'])

        if not z_seq:
            return {'fatigue_prob': 0.5, 'confidence': 0.0,
                    'n_stages_seen': 0, 'stage_scores': []}

        z_tensor     = torch.stack(z_seq)
        z_ctx        = self.causal_tf(z_tensor)
        logits, _    = self.prototype_cls(z_ctx)
        prob         = torch.softmax(logits, dim=0)[1].item()
        stage_scores = torch.sigmoid(
            self.stage_score_head(z_ctx).squeeze(-1)
        ).cpu().numpy().tolist()

        return {
            'fatigue_prob':  prob,
            'confidence':    len(z_seq) / self.n_stages,
            'n_stages_seen': len(z_seq),
            'stage_scores':  stage_scores,
        }


class TPFNLoss(nn.Module):
    def __init__(
        self,
        lambda_workload: float = 0.1,
        lambda_mono:     float = 0.05,
        lambda_smooth:   float = 0.02,
        lambda_kl:       float = 0.01,
        lambda_ortho:    float = 0.1,
        pos_weight:      float = 1.0,
    ):
        super().__init__()
        self.lw  = lambda_workload
        self.lm  = lambda_mono
        self.ls  = lambda_smooth
        self.lkl = lambda_kl
        self.lo  = lambda_ortho
        self.register_buffer('pw',
            torch.tensor([1.0, pos_weight]))
        self.mse = nn.MSELoss()

    def forward(self, out: Dict,
                label_soft: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        device = out['logits'].device
        logits = out['logits'].unsqueeze(0)

        # 支持 soft label（Mixup）
        if label_soft is not None:
            log_p = F.log_softmax(logits, dim=-1)
            L_cls = -(label_soft.to(device) * log_p).sum()
        else:
            ce    = nn.CrossEntropyLoss(
                weight=self.pw.to(device))
            label = out['label'].unsqueeze(0)
            L_cls = ce(logits, label)

        L_workload = self.mse(
            out['workload_pred'],
            out['workload_target'].to(device))

        scores = out['stage_scores']
        S      = scores.size(0)

        L_mono = torch.tensor(0.0, device=device)
        if label_soft is None and \
                out['label'].item() == 1 and S >= 4:
            early  = scores[:S // 3].mean()
            late   = scores[S * 2 // 3:].mean()
            L_mono = F.relu(early - late + 0.1)

        L_smooth = torch.tensor(0.0, device=device)
        if S >= 2:
            L_smooth = ((scores[1:] - scores[:-1]) ** 2).mean()

        L_kl    = (out['kl_fatigue'] + out['kl_workload']) / S
        L_ortho = out['ortho_loss']

        total = (L_cls
                 + self.lw  * L_workload
                 + self.lm  * L_mono
                 + self.ls  * L_smooth
                 + self.lkl * L_kl
                 + self.lo  * L_ortho)

        return {
            'loss':     total,
            'cls':      L_cls.detach(),
            'workload': L_workload.detach(),
            'monotone': L_mono.detach(),
            'smooth':   L_smooth.detach(),
            'kl':       L_kl.detach(),
            'ortho':    L_ortho.detach(),
        }