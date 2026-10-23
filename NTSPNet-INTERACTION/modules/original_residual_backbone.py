from typing import Dict, Iterable, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch


class OriginalResidualBackbone(nn.Module):
    """Inject module-only residuals into an otherwise exact v22 trunk.

    The extended branch is a feature provider.  Its public trajectory outputs
    are discarded; only mode-invariant Style, Reachability and Priority
    contexts are adapted and added to the proposal/refinement representations
    of the original branch.  Three bounded gates start at exact zero.
    """

    MODULE_NAMES = ('style', 'reachability', 'priority')
    STAGE_NAMES = ('proposal', 'refinement')
    MODULE_SELECTIONS = {
        'all': MODULE_NAMES,
        'style_reachability': ('style', 'reachability'),
        'style_priority': ('style', 'priority'),
        'reachability_priority': ('reachability', 'priority'),
    }

    def __init__(self,
                 base: nn.Module,
                 residual: nn.Module,
                 hidden_dim: int,
                 num_modes: int,
                 max_scale: float = 0.1,
                 training_scope: str = 'fusion_only',
                 active_modules: Union[str, Iterable[str]] = 'all') -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')
        if num_modes <= 0:
            raise ValueError('num_modes must be positive')
        if not 0.0 < max_scale <= 1.0:
            raise ValueError(
                'baseline_residual_max_scale must be in (0, 1]'
            )
        if training_scope not in ('fusion_only', 'residual_branch'):
            raise ValueError(
                "baseline_residual_training_scope must be 'fusion_only' "
                "or 'residual_branch'"
            )
        active_modules = self._normalize_active_modules(active_modules)
        if (
            training_scope == 'residual_branch'
            and active_modules != self.MODULE_NAMES
        ):
            raise ValueError(
                'module-specific residual ablations require '
                "baseline_residual_training_scope='fusion_only'"
            )
        self.base = base
        self.residual = residual
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.max_scale = max_scale
        self.training_scope = training_scope
        self.active_modules = active_modules

        self.gates = nn.ParameterDict({
            module_name: nn.Parameter(torch.zeros(()))
            for module_name in self.MODULE_NAMES
        })
        self.adapters = nn.ModuleDict({
            stage: nn.ModuleDict({
                module_name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for module_name in self.MODULE_NAMES
            })
            for stage in self.STAGE_NAMES
        })
        self._configure_trainable_parameters()

    @classmethod
    def _normalize_active_modules(
            cls,
            active_modules: Union[str, Iterable[str]]) -> Tuple[str, ...]:
        if isinstance(active_modules, str):
            requested = cls.MODULE_SELECTIONS.get(
                active_modules,
                (active_modules,),
            )
        else:
            requested = tuple(active_modules)
        if not requested:
            raise ValueError('at least one baseline residual module is required')
        if len(set(requested)) != len(requested):
            raise ValueError('baseline residual modules must be unique')
        unknown = tuple(
            module_name
            for module_name in requested
            if module_name not in cls.MODULE_NAMES
        )
        if unknown:
            raise ValueError(
                f'unknown baseline residual modules: {unknown}'
            )
        return tuple(
            module_name
            for module_name in cls.MODULE_NAMES
            if module_name in requested
        )

    def _configure_trainable_parameters(self) -> None:
        self.base.requires_grad_(False)
        self.residual.requires_grad_(
            self.training_scope == 'residual_branch'
        )
        self.adapters.requires_grad_(False)
        self.gates.requires_grad_(False)
        for module_name in self.active_modules:
            self.gates[module_name].requires_grad_(True)
            for stage in self.STAGE_NAMES:
                self.adapters[stage][module_name].requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The v22 path is always frozen.  During the low-cost phase the
        # extended feature provider is frozen as well, including dropout state.
        self.base.eval()
        if self.training_scope == 'fusion_only':
            self.residual.eval()
        return self

    def require_style_statistics(self) -> None:
        self.residual.require_style_statistics()

    def fusion_scales(self) -> Dict[str, torch.Tensor]:
        return {
            module_name: (
                self.max_scale * torch.tanh(
                    self.gates[module_name] / self.max_scale,
                )
                if module_name in self.active_modules
                else torch.zeros_like(self.gates[module_name])
            )
            for module_name in self.MODULE_NAMES
        }

    def _validate_contexts(self, contexts) -> None:
        if not isinstance(contexts, dict):
            raise TypeError('extended branch must return module contexts')
        for stage in self.STAGE_NAMES:
            if stage not in contexts or not isinstance(contexts[stage], dict):
                raise ValueError(f'missing module context stage {stage!r}')
            expected_shape = None
            for module_name in self.active_modules:
                if module_name not in contexts[stage]:
                    raise ValueError(
                        f'missing {stage}/{module_name} module context'
                    )
                context = contexts[stage][module_name]
                if context.ndim != 3 or context.size(-1) != self.hidden_dim:
                    raise ValueError(
                        f'{stage}/{module_name} context must have shape '
                        f'[N,H,{self.hidden_dim}], received '
                        f'{tuple(context.shape)}'
                    )
                if expected_shape is None:
                    expected_shape = context.shape
                elif context.shape != expected_shape:
                    raise ValueError(
                        f'all {stage} module contexts must share one shape'
                    )
                if not torch.isfinite(context).all():
                    raise ValueError(
                        f'{stage}/{module_name} context contains non-finite '
                        'values'
                    )

    def _adapt_contexts(self, contexts):
        self._validate_contexts(contexts)
        scales = self.fusion_scales()
        representation_residuals = {}
        for stage in self.STAGE_NAMES:
            stage_residual = None
            for module_name in self.active_modules:
                context = contexts[stage][module_name]
                adapted = self.adapters[stage][module_name](context)
                contribution = scales[module_name] * adapted
                stage_residual = (
                    contribution
                    if stage_residual is None
                    else stage_residual + contribution
                )
            representation_residuals[stage] = (
                stage_residual.unsqueeze(2)
                .expand(-1, -1, self.num_modes, -1)
                .reshape(-1, self.hidden_dim)
            )
        return representation_residuals

    def forward(self, data: Batch, l_embs: torch.Tensor):
        residual_grad_enabled = (
            self.training and self.training_scope == 'residual_branch'
        )
        with torch.set_grad_enabled(residual_grad_enabled):
            _, module_contexts = self.residual(
                data=data,
                l_embs=l_embs,
                return_module_contexts=True,
            )
        representation_residuals = self._adapt_contexts(module_contexts)
        return self.base(
            data=data,
            l_embs=l_embs,
            representation_residuals=representation_residuals,
        )
