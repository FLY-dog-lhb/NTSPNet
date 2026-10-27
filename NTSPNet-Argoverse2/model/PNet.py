"""PNet for AV2, with scratch training and explicit three-stage residual training."""
import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from modules import Backbone, OriginalBackbone, OriginalResidualBackbone, MapEncoder
from metrics.forecasting import ForecastMetrics, trajectory_metrics, scene_probabilities
from utils import generate_target

DATASET_SCHEMA = 'av2-h50-f60-raw-centerline-v1'


class PNet(pl.LightningModule):
    def __init__(self, hidden_dim=128, num_modes=6, num_attn_layers=3, num_heads=8,
                 num_hops=4, duration=10, a2a_radius=80., l2a_radius=80., dropout=.1,
                 lr=5e-4, weight_decay=1e-4, warmup_epochs=4, max_epochs=64,
                 trunk='extended', protocol='single-agent', lambda_hist_aux=.3,
                 lambda_style=.05, lambda_priority=.3, freeze_map=False,
                 dataset_schema=DATASET_SCHEMA):
        super().__init__()
        if dataset_schema != DATASET_SCHEMA:
            raise ValueError('Checkpoint data semantics do not match AV2')
        if trunk not in ('baseline', 'extended', 'fusion'):
            raise ValueError('Unknown trunk')
        if protocol not in ('single-agent', 'multi-world'):
            raise ValueError('Unknown evaluation protocol')
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError('hidden_dim must be positive and divisible by num_heads')
        if any(x < 0 for x in (lambda_hist_aux, lambda_style, lambda_priority, weight_decay)) or lr <= 0:
            raise ValueError('Loss weights and weight decay must be nonnegative; learning rate must be positive')
        if num_modes < 1 or num_attn_layers < 1 or max_epochs < 1:
            raise ValueError('modes, layers and epochs must be positive')
        self.save_hyperparameters()
        self.num_historical_steps = 50; self.num_future_steps = 60
        self.MapEncoder = MapEncoder(hidden_dim, num_hops, num_heads, dropout)
        arguments = dict(hidden_dim=hidden_dim, num_historical_steps=50,
                         num_future_steps=60, duration=duration, a2a_radius=a2a_radius,
                         l2a_radius=l2a_radius, num_attn_layers=num_attn_layers,
                         num_modes=num_modes, num_heads=num_heads, dropout=dropout, motion_source='csv')
        if trunk == 'baseline':
            self.Backbone = OriginalBackbone(**arguments)
        else:
            provider = Backbone(**arguments, style_film_mode='bounded_identity',
                                conflict_update_scope='participants_only')
            provider.lane_reachability_builder.prediction_window = 6.
            provider.lane_reachability_builder.tau_max = 6.
            self.Backbone = (provider if trunk == 'extended' else
                             OriginalResidualBackbone(OriginalBackbone(**arguments), provider,
                                                      hidden_dim, num_modes, training_scope='fusion_only'))
        if freeze_map or trunk == 'fusion':
            self.MapEncoder.requires_grad_(False)
        self.validation_metrics = ForecastMetrics()
        self.register_buffer('fusion_initialized', torch.tensor(False))

    def train(self, mode=True):
        super().train(mode)
        if self.hparams.freeze_map or self.hparams.trunk == 'fusion':
            self.MapEncoder.eval()
        return self

    def forward(self, data):
        if self.hparams.trunk == 'fusion' and not bool(self.fusion_initialized):
            raise RuntimeError('Fusion requires trained AV2 baseline and provider checkpoints')
        return self.Backbone(data=data, l_embs=self.MapEncoder(data))

    @classmethod
    def restore(cls, path, **overrides):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        hp = checkpoint.get('hyper_parameters', {})
        if hp.get('dataset_schema') != DATASET_SCHEMA:
            raise ValueError('Expected a PNet-Argoverse2 checkpoint; INTERACTION/AV1 weights are incompatible')
        model = cls.load_from_checkpoint(path, map_location='cpu', strict=True, **overrides)
        model._trained_steps = int(checkpoint.get('global_step', 0))
        return model

    def load_style_statistics(self, path):
        if self.hparams.trunk != 'extended':
            raise ValueError('Style calibration applies to the extended trunk')
        metadata = torch.load(path, map_location='cpu', weights_only=True)
        if metadata.get('source_split') != 'train' or metadata.get('dataset_schema') != DATASET_SCHEMA:
            raise ValueError('Style statistics must come from AV2 training histories using this schema')
        self.Backbone.load_style_statistics(path)

    def initialize_provider(self, baseline_path):
        if self.hparams.trunk != 'extended' or not self.hparams.freeze_map:
            raise ValueError('Provider initialization requires extended trunk and --freeze-map')
        base = self.restore(baseline_path)
        self._check_architecture(base, 'baseline')
        self.MapEncoder.load_state_dict(base.MapEncoder.state_dict(), strict=True)

    def initialize_fusion(self, baseline_path, provider_path):
        if self.hparams.trunk != 'fusion':
            raise ValueError('Expected fusion trunk')
        base, provider = self.restore(baseline_path), self.restore(provider_path)
        self._check_architecture(base, 'baseline'); self._check_architecture(provider, 'extended')
        if not provider.hparams.freeze_map:
            raise ValueError('Train provider using --base-checkpoint and --freeze-map')
        base_map, provider_map = base.MapEncoder.state_dict(), provider.MapEncoder.state_dict()
        if any(not torch.equal(base_map[k], provider_map[k]) for k in base_map):
            raise ValueError('Provider must use exactly the frozen baseline MapEncoder')
        self.MapEncoder.load_state_dict(base_map, strict=True)
        self.Backbone.base.load_state_dict(base.Backbone.state_dict(), strict=True)
        self.Backbone.residual.load_state_dict(provider.Backbone.state_dict(), strict=True)
        self.fusion_initialized.fill_(True)

    def _check_architecture(self, model, trunk):
        if getattr(model, '_trained_steps', 0) <= 0:
            raise ValueError('Stage initialization requires a checkpoint with completed optimizer steps')
        if model.hparams.trunk != trunk:
            raise ValueError('Expected %s checkpoint' % trunk)
        for name in ('hidden_dim', 'num_modes', 'num_attn_layers', 'num_heads', 'num_hops',
                     'duration', 'a2a_radius', 'l2a_radius', 'protocol', 'dataset_schema'):
            if self.hparams[name] != model.hparams[name]:
                raise ValueError('Checkpoint configuration mismatch: ' + name)

    def compute_loss(self, data, outputs):
        proposal, refinement, probability = outputs
        target, valid = generate_target(data['agent'].position, data['agent'].visible_mask, 50, 60)
        historical_valid = data['agent'].visible_mask[:, :50]
        eligible = historical_valid & data['agent'].train_mask[:, None] & valid.any(dim=-1)
        # Fragment/static tracks remain context but never become regression targets.
        errors = (torch.linalg.vector_norm(proposal.detach() - target[:, :, None], dim=-1)
                  * valid[:, :, None]).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)[:, :, None]
        best = errors.argmin(dim=-1)
        if self.hparams.protocol == 'multi-world':
            # Shared best modes for scored actors, independently at each causal anchor.
            eligible &= data['agent'].scored_mask[:, None]
            for scene in range(data.num_graphs):
                actors = data['agent'].batch == scene
                weights = eligible[actors].float()
                world_error = (errors[actors] * weights[:, :, None]).sum(dim=0) / weights.sum(dim=0).clamp_min(1)[:, None]
                best[actors] = world_error.argmin(dim=-1)[None]
        rows = torch.arange(len(best), device=best.device)[:, None]
        steps = torch.arange(50, device=best.device)[None]
        chosen_proposal = proposal[rows, steps, best]
        chosen_refinement = refinement[rows, steps, best]
        per_step = (F.smooth_l1_loss(chosen_proposal, target, reduction='none').sum(dim=-1)
                    + F.smooth_l1_loss(chosen_refinement, target, reduction='none').sum(dim=-1))
        regression = (per_step * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        classification = -probability.gather(-1, best[..., None]).squeeze(-1).clamp_min(1e-8).log()
        weights = probability.new_full((1, 50), self.hparams.lambda_hist_aux / 49.)
        weights[:, -1] = 1.
        weights = weights * eligible
        denom = weights.sum().clamp_min(1e-8)
        trajectory_loss = ((regression + classification) * weights).sum() / denom
        losses = {'trajectory': trajectory_loss}
        loss = trajectory_loss
        if self.hparams.trunk == 'extended':
            if self.hparams.lambda_style:
                self.Backbone.require_style_statistics()
                mask = data['agent'].style_prefix_reliable_mask
                logits = data['agent'].style_prefix_logits
                style = (F.cross_entropy(logits[mask], data['agent'].style_prefix_pseudo_label[mask])
                         if mask.any() else logits.sum() * 0.)
                losses['style'] = style; loss = loss + self.hparams.lambda_style * style
            if self.hparams.lambda_priority:
                mask = data['agent'].priority_pseudo_label_valid_mask
                logits = data['agent'].priority_instance_logits
                if mask.any():
                    priority = F.cross_entropy(logits[mask], data['agent'].priority_pseudo_label[mask])
                    p = data['agent'].priority_instance_probability
                    priority = priority + .2 * F.smooth_l1_loss(
                        (p[:, 0] - p[:, 1])[mask], data['agent'].priority_order_target[mask])
                else:
                    priority = logits.sum() * 0.
                losses['priority'] = priority; loss = loss + self.hparams.lambda_priority * priority
        losses['loss'] = loss
        return losses

    def on_fit_start(self):
        if self.hparams.trunk == 'extended' and self.hparams.lambda_style:
            self.Backbone.require_style_statistics()
        if self.hparams.trunk == 'fusion' and not bool(self.fusion_initialized):
            raise RuntimeError('Refusing to freeze randomly initialized fusion branches')

    def training_step(self, data, batch_idx):
        losses = self.compute_loss(data, self(data))
        if not torch.isfinite(losses['loss']):
            raise FloatingPointError('Non-finite training loss for ' + str(data.scenario_id))
        for name, value in losses.items():
            self.log('train_' + name, value, on_step=True, on_epoch=True,
                     batch_size=data.num_graphs, prog_bar=name == 'loss', sync_dist=True)
        return losses['loss']

    def selected_targets(self, data, scene, require_future=False):
        mask = (data['agent'].focal_mask if self.hparams.protocol == 'single-agent'
                else data['agent'].scored_mask)
        mask = mask & (data['agent'].batch == scene)
        if not data['agent'].visible_mask[mask, 49].all():
            raise ValueError('Scored target missing final observed frame')
        if require_future and not data['agent'].visible_mask[mask, 50:110].all():
            raise ValueError('Scored target missing future annotations; do not evaluate blind test')
        if not mask.any():
            raise ValueError('Scene contains no selected targets')
        return mask

    def on_validation_epoch_start(self):
        self.validation_metrics.reset()

    def validation_step(self, data, batch_idx):
        _, output, probability = self(data)
        for scene in range(data.num_graphs):
            selected = self.selected_targets(data, scene, require_future=True)
            values = trajectory_metrics(output[selected, -1], data['agent'].position[selected, 50:110],
                                        probability[selected, -1], self.hparams.protocol == 'multi-world')
            if not torch.isfinite(values).all():
                raise FloatingPointError('Non-finite validation metrics for ' + str(data.scenario_id[scene]))
            self.validation_metrics.update(values)

    def on_validation_epoch_end(self):
        result = self.validation_metrics.compute()
        names = (('minADE', 'minFDE', 'MR', 'brier_minFDE') if self.hparams.protocol == 'single-agent'
                 else ('avgMinADE', 'avgMinFDE', 'actorMR_at_best_world', 'brier_avgMinFDE'))
        for name, value in zip(names, result):
            self.log('val_' + name, value, prog_bar=True, sync_dist=False)
        self.log('val_score', result[1], sync_dist=False)

    def predict_step(self, data, batch_idx, dataloader_idx=0):
        _, output, probability = self(data)
        scenes = []
        for scene in range(data.num_graphs):
            selected = self.selected_targets(data, scene)
            indices = torch.where(selected)[0]
            local_indices = indices - data['agent'].ptr[scene]
            p = probability[selected, -1]
            p = p[0] if self.hparams.protocol == 'single-agent' else scene_probabilities(p)
            trajectory = output[selected, -1].double() + data.origin[scene]
            scenes.append(dict(scenario_id=data.scenario_id[scene],
                               track_ids=[data.track_ids[scene][i] for i in local_indices.cpu().tolist()],
                               probability=p.detach().cpu(), trajectory=trajectory.detach().cpu()))
        return scenes

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        def schedule(epoch):
            warmup = self.hparams.warmup_epochs
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(self.hparams.max_epochs - warmup, 1)
            return .5 * (1 + math.cos(math.pi * min(progress, 1.)))
        return {'optimizer': optimizer, 'lr_scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)}
