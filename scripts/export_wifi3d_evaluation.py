"""Export a standalone scientific figure from a recorded Wi-Fi evaluation index."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from soundleak.reconstruction_eval import read_evaluation, METHODS


def export(folder):
    folder = Path(folder)
    report, _ = read_evaluation(folder/'evaluation.json')
    methods = report['summary']['available_methods']
    record = report['records'][0]
    labels = [METHODS[k].replace(' frames', '\nframes').replace(' first ', '\nfirst ') for k in methods]
    colors = ['#008b89', '#8355ad', '#677986', '#cf823b', '#a44359'][:len(methods)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    fig.suptitle('Recorded Wi-Fi reconstruction: signal-dependence checks\n'
                 f"{report['summary']['frames']} frames · {report['summary']['environments']} environment(s) · Training overlap unverified", fontsize=14)
    for ax, metric, title in [(axes[0, 0], 'mean_distance', 'Shape error · lower is better'),
                              (axes[0, 1], 'mean_change_field_mae', 'Temporal change-field MAE · lower is better')]:
        values = [report['summary'][metric][k] for k in methods]
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=9)
        ax.set_ylim(0, max(values)*1.25)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Standardized coordinate units')
        ax.tick_params(axis='x', labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    for key, color in zip(methods, colors):
        axes[1, 0].plot([f['frame_number'] for f in record['frames']],
                         [f['scores'][key] for f in record['frames']], marker='.', label=METHODS[key], color=color)
    axes[1, 0].set(title=record['recording']+' · sampled frame errors', xlabel='Frame index', ylabel='Shape error')
    axes[1, 0].legend(fontsize=8)
    transitions = record['transitions']
    x = [t['to_frame'] for t in transitions]
    axes[1, 1].plot(x, [t['reference_change'] for t in transitions], marker='o', color='#b36b19', label='Measured reference')
    axes[1, 1].plot(x, [t['methods']['correct']['predicted_change'] for t in transitions], marker='o', color='#008b89', label='Correct CSI prediction')
    axes[1, 1].plot(x, [t['methods']['fixed']['predicted_change'] for t in transitions], '--', color='#677986', label='Fixed prediction')
    axes[1, 1].set(title='Changes at shared surface-distance probes', xlabel='Transition ending at frame', ylabel='Mean absolute field change')
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.spines[['top', 'right']].set_visible(False)
    if report['summary']['missing_methods']:
        note = 'Different-environment control unavailable: official archive downloads blocked. Only cached data is scored.'
    else:
        note = 'Exploratory fixed sample, not a held-out benchmark. Correlated frames; recordings receive equal weight.'
    fig.supxlabel(note+'\nStandardized coordinates, no fitted alignment; temporal scores are not displacement in meters. MM-Fi: CC BY-NC 4.0.', fontsize=9)
    path = folder/'evaluation.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT/'outputs/wifi3d/evaluation')
    export(parser.parse_args().folder)
