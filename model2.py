import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution.

    The output at time t only depends on current and historical samples.
    This is used in the fNIRS branch to model delayed hemodynamic responses
    through a causal receptive field, rather than by imposing a fixed
    physiological delay or manual temporal shift.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=self.pad,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, :x.size(2)]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalAttentionPool(nn.Module):
    """
    Attention pooling over temporal samples inside one local window.

    Input:
        x: [B, D, L]

    Output:
        vec:  [B, D]
        attn: [B, L]
    """
    def __init__(self, D: int):
        super().__init__()

        self.score = nn.Sequential(
            nn.Conv1d(D, max(D // 4, 1), kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(max(D // 4, 1), 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(x).squeeze(1), dim=-1)
        vec = (attn.unsqueeze(1) * x).sum(dim=-1)
        return vec, attn


class LearnableMissingToken(nn.Module):
    """
    Learnable modality-specific missing tokens.
    """
    def __init__(self, D: int, modalities: List[str]):
        super().__init__()

        self.tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(D) * 0.02)
            for m in modalities
        })

    def get(
        self,
        name: str,
        W: int,
        device: torch.device
    ) -> torch.Tensor:
        tok = self.tokens[name].to(device)
        return tok.unsqueeze(0).expand(W, -1).clone()


class EEGBranch(nn.Module):
    """
    EEG branch.

    EEG is encoded using a non-causal 1D-CNN.
    """
    def __init__(self, C_eeg: int = 42, D: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                C_eeg,
                C_eeg,
                kernel_size=7,
                padding=3,
                groups=C_eeg
            ),
            nn.Conv1d(C_eeg, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            nn.Conv1d(
                64,
                64,
                kernel_size=5,
                padding=2,
                groups=64
            ),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            nn.Conv1d(128, D, kernel_size=3, padding=1),
            nn.BatchNorm1d(D),
            nn.GELU(),
        )

        self.pool = TemporalAttentionPool(D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.pool(self.cnn(x))
        return self.norm(h)


class EyeBranch(nn.Module):
    """
    Eye-tracking branch.

    Eye-tracking signals are encoded using a non-causal 1D-CNN.
    """
    def __init__(self, C_eye: int = 11, D: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(C_eye, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.2),

            nn.Conv1d(32, D, kernel_size=3, padding=1),
            nn.BatchNorm1d(D),
            nn.GELU(),
        )

        self.pool = TemporalAttentionPool(D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.pool(self.cnn(x))
        return self.norm(h)


class FNIRSBranch(nn.Module):
    """
    fNIRS branch.

    fNIRS responses are modeled using a 12-s causal receptive field ending
    at the current 4-s reference window, without imposing a fixed
    physiological delay.

    Expected input:
        x: [W, C_fnirs, T_context]

    W:
        Number of reference steps/windows in the current flight phase.

    T_context:
        fNIRS temporal context ending at each reference step. In the
        manuscript setting, this corresponds to a 12-s causal receptive field.
    """
    def __init__(self, C_fnirs: int = 22, D: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            CausalConv1d(C_fnirs, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            CausalConv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Dropout(0.2),

            CausalConv1d(64, D, kernel_size=3),
            nn.BatchNorm1d(D),
            nn.GELU(),
        )

        self.hrf_pool = TemporalAttentionPool(D)
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        h, _ = self.hrf_pool(feat)
        return self.norm(h)


class SAFModule(nn.Module):
    """
    Semantically Asynchronous Fusion module.

    This follows the manuscript formula:

        beta_m = softmax(phi_m(u_m))
        v      = sum_m beta_m * u_m

    Fusion is performed at the window/reference-step representation level,
    without enforcing sample-level physiological synchrony.
    """
    def __init__(self, D: int = 128, n_modalities: int = 3):
        super().__init__()

        self.D = D
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

    def forward(
        self,
        feats: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = torch.cat(
            [self.gate_mlp[i](feats[i]) for i in range(self.n_mod)],
            dim=-1
        )

        modal_attn = torch.softmax(gates, dim=-1)

        fused = sum(
            modal_attn[:, i:i + 1] * feats[i]
            for i in range(self.n_mod)
        )

        return fused, modal_attn


class WindowAttentionPool(nn.Module):
    """
    Window-attention pooling within each flight phase.

        alpha^(s,t) = softmax(psi(v^(s,t)))
        h^(s)       = sum_t alpha^(s,t) v^(s,t)
    """
    def __init__(self, D: int = 128):
        super().__init__()

        self.score = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.Tanh(),
            nn.Linear(D // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(x).squeeze(-1), dim=0)
        vec = (attn.unsqueeze(-1) * x).sum(0)
        return vec, attn


class DisentangleEncoder(nn.Module):
    """
    Variational disentanglement module.

    The phase-level multimodal representation h^(s) is mapped into:
        z_f^(s): fatigue-related latent variable
        z_w^(s): task-progress/workload-related latent variable
    """
    def __init__(self, D: int = 128):
        super().__init__()

        Dh = D // 2

        self.fatigue_mu = nn.Linear(D, Dh)
        self.fatigue_logv = nn.Linear(D, Dh)

        self.workload_mu = nn.Linear(D, Dh)
        self.workload_logv = nn.Linear(D, Dh)

        self.workload_reg = nn.Sequential(
            nn.Linear(Dh, Dh // 2),
            nn.GELU(),
            nn.Linear(Dh // 2, 1),
            nn.Sigmoid(),
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps

        return mu

    def _kl(
        self,
        mu: torch.Tensor,
        logv: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard Gaussian KL divergence:

            D_KL(q(z|h) || p(z))
            = -1/2 sum_i (1 + log sigma_i^2 - mu_i^2 - sigma_i^2)
        """
        return -0.5 * (1 + logv - mu.pow(2) - logv.exp()).sum()

    def forward(
        self,
        stage_vec: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        f_mu = self.fatigue_mu(stage_vec)
        f_logv = self.fatigue_logv(stage_vec)
        z_f = self.reparameterize(f_mu, f_logv)

        w_mu = self.workload_mu(stage_vec)
        w_logv = self.workload_logv(stage_vec)
        z_w = self.reparameterize(w_mu, w_logv)

        workload_progress = self.workload_reg(z_w).squeeze(-1)

        return {
            "z_fatigue": z_f,
            "z_workload": z_w,
            "kl_fatigue": self._kl(f_mu, f_logv),
            "kl_workload": self._kl(w_mu, w_logv),
            "workload_progress": workload_progress,
        }


class OrthogonalConstraint(nn.Module):
    """
    Orthogonality loss:

        L_orth = || (1/T) Z_f^T Z_w ||_F^2
    """
    def forward(
        self,
        z_f: torch.Tensor,
        z_w: torch.Tensor
    ) -> torch.Tensor:
        T = z_f.size(0)
        cross = (z_f.T @ z_w) / max(T, 1)
        return cross.pow(2).sum()


class CausalStageTransformer(nn.Module):
    """
    Causal stage Transformer.

    The representation of phase s can only attend to phases 1,...,s.
    """
    def __init__(
        self,
        D: int = 64,
        nhead: int = 4,
        num_layers: int = 2
    ):
        super().__init__()

        self.pos_enc = PositionalEncoding(D, max_len=32)

        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=nhead,
            dim_feedforward=D * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        self.encoder.enable_nested_tensor = False

    def forward(
        self,
        z_seq: torch.Tensor,
        use_causal_mask: bool = True
    ) -> torch.Tensor:
        S = z_seq.size(0)
        x = self.pos_enc(z_seq.unsqueeze(0))

        mask = None
        if use_causal_mask and S > 1:
            mask = torch.triu(
                torch.full(
                    (S, S),
                    float("-inf"),
                    device=x.device
                ),
                diagonal=1
            )

        return self.encoder(x, mask=mask).squeeze(0)


class FatigueProcessPrototype(nn.Module):
    """
    Fatigue-process prototype matching.

    For each class c in {0, 1}, a learnable prototype sequence P_c is used.
    Soft matching is performed between the input fatigue-process sequence and
    each class prototype.
    """
    def __init__(
        self,
        D: int = 64,
        n_stages: int = 9,
        tau: float = 0.1
    ):
        super().__init__()

        self.n_stages = n_stages
        self.tau = tau

        self.prototype = nn.Parameter(
            torch.randn(2, n_stages, D) * 0.02
        )

    def _soft_align_sim(
        self,
        z_seq: torch.Tensor,
        proto: torch.Tensor
    ) -> torch.Tensor:
        z_n = F.normalize(z_seq, dim=-1)
        p_n = F.normalize(proto, dim=-1)

        sim_mat = z_n @ p_n.T

        align_weight = torch.softmax(sim_mat / self.tau, dim=-1)
        soft_sim = (sim_mat * align_weight).sum(dim=-1)

        return soft_sim.mean()

    def forward(
        self,
        z_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S = z_seq.size(0)
        proto = self.prototype[:, :S, :]

        logits = torch.stack([
            self._soft_align_sim(z_seq, proto[0]),
            self._soft_align_sim(z_seq, proto[1]),
        ])

        proto_attn = []

        z_n = F.normalize(z_seq, dim=-1)

        for i in range(2):
            p_n = F.normalize(proto[i], dim=-1)
            sim_mat = z_n @ p_n.T

            attn = torch.softmax(sim_mat.mean(dim=0), dim=0)
            proto_attn.append(attn)

        proto_attn = torch.stack(proto_attn)

        return logits, proto_attn


class StagewiseFatigueIndicator(nn.Module):
    """
    Stage-wise Fatigue Indicator module.

    Structure:
        Linear -> GELU -> Linear -> Sigmoid
    """
    def __init__(self, D_z: int = 64):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(D_z, D_z // 2),
            nn.GELU(),
            nn.Linear(D_z // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        return self.head(z_ctx).squeeze(-1)


class SADFPM(nn.Module):
    """
    Semantic Asynchronous Disentangled Fatigue Process Modeling.

    Main components:
        1. Modality-specific EEG, eye-tracking, and fNIRS encoders.
        2. Semantically Asynchronous Fusion.
        3. Task-progress-constrained variational disentanglement.
        4. Causal stage Transformer.
        5. Fatigue-process prototype matching.
        6. Stage-wise Fatigue Indicator.

    fNIRS setting:
        All modalities are organized by the same 4-s reference step for
        window-level correspondence. fNIRS responses are encoded using a
        12-s causal receptive field ending at the current 4-s reference
        window, without imposing a fixed physiological delay.
    """
    def __init__(
        self,
        C_eeg: int = 42,
        T_eeg: int = 2000,
        C_eye: int = 11,
        T_eye: int = 240,
        C_fnirs: int = 22,
        T_fnirs: int = 122,
        D: int = 128,
        D_z: int = 64,
        n_stages: int = 9,
        dropout: float = 0.4,
        fnirs_context_windows: int = 3,
        fnirs_context_mode: str = "auto",
    ):
        """
        Args:
            C_eeg:
                Number of EEG channels.
            T_eeg:
                Number of EEG samples in the 4-s reference window.
            C_eye:
                Number of eye-tracking channels/features.
            T_eye:
                Number of eye-tracking samples in the 4-s reference window.
            C_fnirs:
                Number of fNIRS channels.
            T_fnirs:
                Number of fNIRS samples in the 12-s causal context.
            D:
                Hidden feature dimension. The manuscript setting is D=128.
            D_z:
                Latent dimension for z_f and z_w. The manuscript setting is
                D_z = D / 2.
            n_stages:
                Number of flight phases. This is kept configurable because it
                must match the preprocessing output.
            dropout:
                Dropout probability.
            fnirs_context_windows:
                Number of 4-s fNIRS reference windows used to build the causal
                fNIRS context when the input contains only 4-s fNIRS windows.
                The default value 3 corresponds to a 12-s causal receptive
                field.
            fnirs_context_mode:
                "auto":
                    Treat fNIRS input as pre-built 12-s context if its temporal
                    length is close to T_fnirs; otherwise build the context from
                    current and historical reference windows.
                "prebuilt":
                    The fNIRS input is already the 12-s causal context ending
                    at each reference window.
                "build":
                    Always build the 12-s causal context from current and
                    historical fNIRS reference windows.

        Note:
            Building a 12-s causal context is not equivalent to applying a
            fixed physiological delay. The context only defines the maximum
            historical receptive field available to the causal fNIRS encoder.
        """
        super().__init__()

        if D_z != D // 2:
            raise ValueError(
                f"D_z must be D // 2 to match the disentanglement design. "
                f"Got D={D}, D_z={D_z}."
            )

        if fnirs_context_mode not in {"auto", "prebuilt", "build"}:
            raise ValueError(
                "fnirs_context_mode must be one of "
                "{'auto', 'prebuilt', 'build'}."
            )

        self.C_eeg = C_eeg
        self.T_eeg = T_eeg

        self.C_eye = C_eye
        self.T_eye = T_eye

        self.C_fnirs = C_fnirs
        self.T_fnirs = T_fnirs

        self.D = D
        self.D_z = D_z
        self.n_stages = n_stages

        self.fnirs_context_windows = fnirs_context_windows
        self.fnirs_context_mode = fnirs_context_mode

        self.eeg_branch = EEGBranch(C_eeg, D)
        self.eye_branch = EyeBranch(C_eye, D)
        self.fnirs_branch = FNIRSBranch(C_fnirs, D)

        self.missing_token = LearnableMissingToken(
            D,
            ["eeg", "eye", "fnirs"]
        )

        self.saf = SAFModule(D, n_modalities=3)
        self.win_pool = WindowAttentionPool(D)

        self.disentangle = DisentangleEncoder(D)
        self.ortho_loss = OrthogonalConstraint()

        self.causal_tf = CausalStageTransformer(
            D_z,
            nhead=4,
            num_layers=2
        )

        self.prototype_cls = FatigueProcessPrototype(
            D_z,
            n_stages=n_stages
        )

        self.sfi = StagewiseFatigueIndicator(D_z)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize(
        x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None

        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True) + 1e-6

        return (x - mean) / std

    def _build_fnirs_causal_context(
        self,
        fnirs: torch.Tensor
    ) -> torch.Tensor:
        """
        Build the fNIRS causal context for each reference step.

        Input:
            fnirs: [W, C_fnirs, L_ref]

        Output:
            context: [W, C_fnirs, K * L_ref]

        For each reference step t, the context is:

            [t-K+1, ..., t-1, t]

        where K = fnirs_context_windows.

        Missing historical windows at the beginning of a phase are padded with
        zeros. This operation does not shift fNIRS relative to other modalities;
        it only gives the causal fNIRS encoder access to historical fNIRS
        samples ending at the current reference step.
        """
        if fnirs.dim() != 3:
            raise ValueError(
                "fNIRS input must have shape [W, C_fnirs, T]."
            )

        W, C, L = fnirs.shape
        K = self.fnirs_context_windows

        contexts = []

        for t in range(W):
            chunks = []

            for offset in range(K - 1, -1, -1):
                idx = t - offset

                if idx < 0:
                    chunks.append(torch.zeros_like(fnirs[0]))
                else:
                    chunks.append(fnirs[idx])

            contexts.append(torch.cat(chunks, dim=-1))

        return torch.stack(contexts, dim=0)

    def _prepare_fnirs_input(
        self,
        fnirs: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Prepare fNIRS input for the causal fNIRS branch.

        The model supports both preprocessing strategies:
            1. The dataset already provides 12-s causal fNIRS context.
            2. The dataset provides 4-s fNIRS reference windows and the model
               builds the 12-s causal context internally.

        In both cases, the resulting fNIRS representation is computed from a
        causal receptive field ending at the current 4-s reference window.
        """
        if fnirs is None:
            return None

        if fnirs.dim() != 3:
            raise ValueError(
                "fNIRS tensor should have shape [W, C_fnirs, T]."
            )

        if self.fnirs_context_mode == "prebuilt":
            return fnirs

        if self.fnirs_context_mode == "build":
            return self._build_fnirs_causal_context(fnirs)

        temporal_len = fnirs.size(-1)

        if temporal_len >= int(0.75 * self.T_fnirs):
            return fnirs

        return self._build_fnirs_causal_context(fnirs)

    def encode_stage(
        self,
        eeg: Optional[torch.Tensor],
        eye: Optional[torch.Tensor],
        fnirs: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        W = None

        for x in [eeg, eye, fnirs]:
            if x is not None:
                W = x.shape[0]
                break

        if W is None:
            return None

        encoded = {}

        if eeg is not None:
            eeg = self._normalize(eeg.to(device))
            encoded["eeg"] = self.eeg_branch(eeg)

        if eye is not None:
            eye = self._normalize(eye.to(device))
            encoded["eye"] = self.eye_branch(eye)

        if fnirs is not None:
            fnirs = self._prepare_fnirs_input(fnirs.to(device))
            fnirs = self._normalize(fnirs)
            encoded["fnirs"] = self.fnirs_branch(fnirs)

        if not encoded:
            return None

        W = min(f.size(0) for f in encoded.values())

        feat_eeg = (
            encoded["eeg"][:W]
            if "eeg" in encoded
            else self.missing_token.get("eeg", W, device)
        )

        feat_eye = (
            encoded["eye"][:W]
            if "eye" in encoded
            else self.missing_token.get("eye", W, device)
        )

        feat_fnirs = (
            encoded["fnirs"][:W]
            if "fnirs" in encoded
            else self.missing_token.get("fnirs", W, device)
        )

        fused, modal_attn = self.saf([
            feat_eeg,
            feat_eye,
            feat_fnirs
        ])

        fused = self.dropout(fused)

        stage_vec, win_attn = self.win_pool(fused)

        dis = self.disentangle(stage_vec)

        return {
            "stage_vec": stage_vec,
            "z_fatigue": dis["z_fatigue"],
            "z_workload": dis["z_workload"],
            "kl_fatigue": dis["kl_fatigue"],
            "kl_workload": dis["kl_workload"],
            "workload_progress": dis["workload_progress"],
            "modal_attn": modal_attn,
            "win_attn": win_attn,
        }

    def forward(
        self,
        session,
        device: torch.device
    ) -> Optional[Dict[str, torch.Tensor]]:
        def _to_tensor(x):
            if x is None:
                return None
            return torch.as_tensor(x, dtype=torch.float32)

        stage_results = []
        z_fatigue_seq = []
        z_workload_seq = []
        stage_progress = []

        for stage in session.stages:
            res = self.encode_stage(
                _to_tensor(stage.eeg),
                _to_tensor(stage.eye),
                _to_tensor(stage.fnirs),
                device
            )

            if res is None:
                continue

            stage_results.append(res)
            z_fatigue_seq.append(res["z_fatigue"])
            z_workload_seq.append(res["z_workload"])

            progress = stage.stage_index / max(
                len(session.stages) - 1,
                1
            )
            stage_progress.append(progress)

        if not z_fatigue_seq:
            return None

        S = len(z_fatigue_seq)

        Z_f = torch.stack(z_fatigue_seq)
        Z_w = torch.stack(z_workload_seq)

        z_ctx = self.causal_tf(Z_f)

        logits, proto_attn = self.prototype_cls(z_ctx)

        stage_scores = self.sfi(z_ctx)

        workload_pred = torch.stack([
            r["workload_progress"]
            for r in stage_results
        ])

        workload_target = torch.tensor(
            stage_progress,
            dtype=torch.float32,
            device=device
        )

        kl_fatigue = sum(
            r["kl_fatigue"]
            for r in stage_results
        )

        kl_workload = sum(
            r["kl_workload"]
            for r in stage_results
        )

        modal_attn = torch.stack([
            r["modal_attn"].mean(0)
            for r in stage_results
        ])

        ortho = self.ortho_loss(Z_f, Z_w)

        label = torch.tensor(
            session.label,
            dtype=torch.long,
            device=device
        )

        return {
            "logits": logits,
            "stage_scores": stage_scores,

            "workload_pred": workload_pred,
            "workload_target": workload_target,

            "kl_fatigue": kl_fatigue,
            "kl_workload": kl_workload,
            "ortho_loss": ortho,

            "modal_attn": modal_attn,
            "proto_attn": proto_attn,

            "z_f": Z_f,
            "z_w": Z_w,

            "z_fatigue_seq": z_ctx,
            "z_fatigue_raw_seq": Z_f,
            "z_workload_seq": Z_w,

            "label": label,
            "session_id": session.session_id,
            "n_stages": S,
        }

    @torch.no_grad()
    def realtime_predict(
        self,
        partial_stages,
        device: torch.device
    ) -> Dict:
        self.eval()

        def _to_tensor(x):
            if x is None:
                return None
            return torch.as_tensor(x, dtype=torch.float32)

        z_seq = []

        for stage in partial_stages:
            res = self.encode_stage(
                _to_tensor(stage.eeg),
                _to_tensor(stage.eye),
                _to_tensor(stage.fnirs),
                device
            )

            if res is not None:
                z_seq.append(res["z_fatigue"])

        if not z_seq:
            return {
                "fatigue_prob": 0.5,
                "confidence": 0.0,
                "n_stages_seen": 0,
                "stage_scores": [],
            }

        z_tensor = torch.stack(z_seq)
        z_ctx = self.causal_tf(z_tensor)

        logits, _ = self.prototype_cls(z_ctx)
        prob = torch.softmax(logits, dim=0)[1].item()

        stage_scores = self.sfi(z_ctx).cpu().numpy().tolist()

        return {
            "fatigue_prob": prob,
            "confidence": len(z_seq) / self.n_stages,
            "n_stages_seen": len(z_seq),
            "stage_scores": stage_scores,
        }


class SADFPMLoss(nn.Module):
    """
    Joint SAD-FPM objective:

        L = L_cls
            + lambda_prog   L_prog
            + lambda_mono   L_mono
            + lambda_smooth L_smooth
            + lambda_KL     L_KL
            + lambda_orth   L_orth

    L_mono is label-gated and mainly applied to fatigue sessions.
    L_smooth is applied to the latent stage-wise fatigue scores.
    """
    def __init__(
        self,
        lambda_prog: float = 0.1,
        lambda_mono: float = 0.05,
        lambda_smooth: float = 0.02,
        lambda_kl: float = 0.01,
        lambda_orth: float = 0.1,
        pos_weight: float = 1.0,
        monotonic_margin: float = 0.1,
    ):
        super().__init__()

        self.lambda_prog = lambda_prog
        self.lambda_mono = lambda_mono
        self.lambda_smooth = lambda_smooth
        self.lambda_kl = lambda_kl
        self.lambda_orth = lambda_orth

        self.monotonic_margin = monotonic_margin

        self.register_buffer(
            "class_weight",
            torch.tensor([1.0, pos_weight])
        )

        self.mse = nn.MSELoss()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        label_soft: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        device = out["logits"].device
        logits = out["logits"].unsqueeze(0)

        if label_soft is not None:
            label_soft = label_soft.to(device)

            if label_soft.dim() == 1:
                label_soft = label_soft.unsqueeze(0)

            log_prob = F.log_softmax(logits, dim=-1)
            L_cls = -(label_soft * log_prob).sum(dim=-1).mean()

            fatigue_gate = label_soft[:, 1].mean()
        else:
            ce = nn.CrossEntropyLoss(
                weight=self.class_weight.to(device)
            )

            label = out["label"].unsqueeze(0)
            L_cls = ce(logits, label)

            fatigue_gate = out["label"].float()

        L_prog = self.mse(
            out["workload_pred"],
            out["workload_target"].to(device)
        )

        scores = out["stage_scores"]
        S = scores.size(0)

        L_mono = torch.tensor(0.0, device=device)

        if S >= 4:
            early = scores[:S // 3].mean()
            late = scores[S * 2 // 3:].mean()

            L_mono = fatigue_gate * F.relu(
                early - late + self.monotonic_margin
            )

        L_smooth = torch.tensor(0.0, device=device)

        if S >= 2:
            L_smooth = ((scores[1:] - scores[:-1]) ** 2).mean()

        L_kl = (
            out["kl_fatigue"]
            + out["kl_workload"]
        ) / max(S, 1)

        L_orth = out["ortho_loss"]

        total = (
            L_cls
            + self.lambda_prog * L_prog
            + self.lambda_mono * L_mono
            + self.lambda_smooth * L_smooth
            + self.lambda_kl * L_kl
            + self.lambda_orth * L_orth
        )

        return {
            "loss": total,
            "cls": L_cls.detach(),
            "prog": L_prog.detach(),
            "monotone": L_mono.detach(),
            "smooth": L_smooth.detach(),
            "kl": L_kl.detach(),
            "ortho": L_orth.detach(),
        }
