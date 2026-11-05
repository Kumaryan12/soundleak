"""Train a small recorded-CSI point-cloud model from scratch, offline.

The supervised mode requires explicit validation and test environments. A separate
--sanity-check mode can verify optimization on cached frames; it makes no test
accuracy claim and its checkpoint is permanently tagged diagnostic-only.
"""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from soundleak.reconstruction_train import (SmallCSICloud, chamfer_loss, load_training_data,
    partition_examples, model_batch, digest)
from soundleak.reconstruction import symmetric_nn_distance


def score(model, examples, device):
    model.eval()
    with torch.inference_mode():
        predicted = model_batch(model, examples, device)
        loss = float(chamfer_loss(predicted, [e['target'].to(device) for e in examples]).cpu())
        distance = float(np.mean([symmetric_nn_distance(p.cpu().numpy(), e['target'].numpy())
                                  for p, e in zip(predicted, examples)]))
    return loss, distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/real/wifi3d')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--validation-environments', nargs='+')
    parser.add_argument('--test-environments', nargs='+')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--points', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a fresh directory to preserve the previous run.')
    if not 1 <= args.steps <= 100_000 or not 1 <= args.batch_size <= 32 or not 0 < args.learning_rate <= .1:
        parser.error('Invalid training budget or learning rate.')
    if not 16 <= args.points <= 1200:
        parser.error('Output point count must be between 16 and 1200.')
    if args.sanity_check and (args.validation_environments or args.test_environments):
        parser.error('Sanity checks do not create a held-out evaluation.')
    if not args.sanity_check and (not args.validation_environments or not args.test_environments):
        parser.error('Training requires --validation-environments and --test-environments. Use --sanity-check only for an explicitly labeled optimization diagnostic.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable; choose CPU.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; choose CPU.')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest_path = args.manifest or args.data/('manifest.json' if args.sanity_check else 'evaluation-manifest.json')
    examples = load_training_data(args.data, manifest_path)
    try:
        training, validation, test = (examples, [], []) if args.sanity_check else partition_examples(
            examples, args.validation_environments, args.test_environments)
    except ValueError as exc:
        parser.error(str(exc))
    model = SmallCSICloud(args.points).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    initial_loss, initial_distance = score(model, training, args.device)
    history = [{'step': 0, 'training_loss': initial_loss, 'validation_loss': None}]
    best_validation = float('inf')
    selected_step = 0
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        model.train()
        selected = rng.choice(len(training), min(len(training), args.batch_size), replace=False)
        batch = [training[i] for i in selected]
        optimizer.zero_grad(set_to_none=True)
        predicted = model_batch(model, batch, args.device)
        loss = chamfer_loss(predicted, [e['target'].to(args.device) for e in batch])
        if not torch.isfinite(loss):
            raise ValueError('Training loss is nonfinite; refusing to save this model.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            train_loss, _ = score(model, training, args.device)
            val_loss, _ = score(model, validation, args.device) if validation else (None, None)
            if not np.isfinite(train_loss) or (val_loss is not None and not np.isfinite(val_loss)):
                raise ValueError('Nonfinite scored loss; refusing to save this model.')
            history.append({'step': step, 'training_loss': train_loss, 'validation_loss': val_loss})
            print('Step', step, 'training squared Chamfer:', round(train_loss, 6), 'validation:', val_loss, flush=True)
            if not validation or val_loss < best_validation:
                selected_step = step
                if validation:
                    best_validation = val_loss
                torch.save({'schema_version': 1, 'architecture': 'small_csi_cloud_v1', 'points': args.points,
                            'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'diagnostic_only': bool(args.sanity_check), 'trained_from_scratch': True,
                            'seed': args.seed, 'selected_step': step,
                            'training_ids': [e['id'] for e in training], 'validation_ids': [e['id'] for e in validation],
                            'test_ids': [e['id'] for e in test],
                            'manifest_sha256': digest(manifest_path)}, args.output/'model.pt')
    checkpoint = torch.load(args.output/'model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    final_loss, final_distance = score(model, training, args.device)
    val_loss, val_distance = score(model, validation, args.device) if validation else (None, None)
    # Final test scoring happens only after validation-based model selection.
    test_loss, test_distance = score(model, test, args.device) if test else (None, None)
    # This only measures training-set sensitivity in sanity mode. No target has
    # been held out, and these numbers must never be relabeled test accuracy.
    model.eval()
    with torch.inference_mode():
        predictions = model_batch(model, training, args.device).cpu().numpy()
        zero = model(torch.zeros((1, 3, 114, 2, 10), device=args.device))[0].cpu().numpy()
    fixed = float(np.mean([symmetric_nn_distance(predictions[0], e['target'].numpy()) for e in training]))
    zero_distance = float(np.mean([symmetric_nn_distance(zero, e['target'].numpy()) for e in training]))
    report = {'schema_version': 1, 'architecture': 'small_csi_cloud_v1', 'trained_from_scratch': True,
              'diagnostic_only': bool(args.sanity_check), 'source_kind': 'recorded_wifi_csi',
              'coordinate_units': 'standardized_model_coordinates', 'dataset_license': 'CC BY-NC 4.0',
              'mode': 'training-set optimization diagnostic' if args.sanity_check else 'supervised train/validation/test environment-split training',
              'parameters': sum(p.numel() for p in model.parameters()), 'output_points': args.points,
              'device': args.device, 'seed': args.seed, 'steps_run': args.steps, 'selected_step': selected_step,
              'duration_seconds': time.perf_counter()-start, 'learning_rate': args.learning_rate,
              'batch_size': args.batch_size, 'manifest_sha256': digest(manifest_path),
              'model_sha256': digest(args.output/'model.pt'),
              'torch_version': importlib.metadata.version('torch'),
              'training_ids': [e['id'] for e in training], 'validation_ids': [e['id'] for e in validation],
              'test_ids': [e['id'] for e in test],
              'initial_training_loss': initial_loss, 'final_training_loss': final_loss,
              'initial_training_nn_distance': initial_distance, 'final_training_nn_distance': final_distance,
              'fixed_prediction_training_nn_distance': fixed, 'zero_input_training_nn_distance': zero_distance,
              'validation_loss': val_loss, 'validation_nn_distance': val_distance,
              'test_loss': test_loss, 'test_nn_distance': test_distance,
              'history': history,
              'limitations': ['A training-set fit does not demonstrate reconstruction accuracy on unseen data.',
                             'Sanity-check checkpoints are diagnostic-only and must not replace the hosted model.',
                             'Targets use per-frame standardized coordinates, not verified physical room dimensions.',
                             'This compact model outputs fewer points than the original replay; its score is not a direct benchmark comparison.',
                             'A held-out environment used for checkpoint selection is validation, not an untouched final test set.',
                             'Any final test result must be assessed against full negative controls, more participants/recordings, and an unchanged test protocol.']}
    (args.output/'training.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (args.output/'loss.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['step', 'training_loss', 'validation_loss'], lineterminator='\n')
        writer.writeheader();writer.writerows(history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    ax.plot([h['step'] for h in history], [h['training_loss'] for h in history], label='Training set')
    if validation:
        ax.plot([h['step'] for h in history], [h['validation_loss'] for h in history], label='Validation environment')
    ax.set(xlabel='Optimizer step', ylabel='Squared Chamfer loss',
           title='Optimization diagnostic · no held-out test' if args.sanity_check else 'Recorded CSI training · environment-level validation')
    ax.legend();fig.savefig(args.output/'loss.png', dpi=150);plt.close(fig)
    print('Saved', args.output, 'diagnostic_only:', args.sanity_check, 'parameters:', report['parameters'], flush=True)
    print('Training loss:', initial_loss, '->', final_loss, flush=True)


if __name__ == '__main__':
    main()
