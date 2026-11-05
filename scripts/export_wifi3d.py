"""Export a standalone orbit viewer and a scientific comparison figure."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from soundleak.reconstruction import read_reconstruction


def export(folder):
    folder = Path(folder)
    report = read_reconstruction(folder/'reconstruction.json')
    assets = ROOT/'soundleak/reconstruction_frontend'
    fragment = (assets/'viewer.html').read_text()
    javascript = (assets/'viewer.js').read_text().replace('export default function', 'function render', 1)
    css = (assets/'viewer.css').read_text()
    serialized = json.dumps(report, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Soundleak · Recorded Wi-Fi 3D reconstruction</title>
<style>body{margin:0;background:#0b121c;color:#dbe5ef;font:16px system-ui}main{max-width:1200px;margin:auto;padding:24px}h1{letter-spacing:-.04em}p{line-height:1.55}a{color:#51e0d2}select{padding:10px;background:#162638;color:white;border:1px solid #365267;border-radius:8px;margin:12px}aside{background:#202334;border:1px solid #67557a;border-radius:12px;padding:16px;margin:18px 0}small{color:#a4b6c7}</style>
<style>''' + css + '''</style><main><h1>Recorded Wi-Fi → predicted 3D point cloud</h1>
<p>Six real MM-Fi CSI frames, passed through the pretrained CSI2PointCloud model. Amber points are independently measured LiDAR reference geometry. Coordinates are standardized model units, not meters.</p>
<aside>Validation result: reusing the first prediction performed slightly better than updating the prediction for each Wi-Fi frame. This is a recorded-data inference experiment, not validated room mapping.</aside>
<label for="frame">Recorded frame</label><select id="frame"></select><div id="viewer-root"></div>
<p id="scores"></p><p><a href="https://github.com/Arritmic/csi2pointcloud">CSI2PointCloud</a> · <a href="https://ntu-aiot-lab.github.io/mm-fi/">MM-Fi dataset</a></p>
<small>MM-Fi: Jianfei Yang and colleagues (NeurIPS 2023), CC BY-NC 4.0, noncommercial research. Model code: Määttä, Sharifipour, Bordallo López and Álvarez Casado (2025), MIT. Checkpoint source and input hashes are in reconstruction.json.</small></main>
<script type="module">const report=''' + serialized + ';\n' + javascript + '\nconst fragment=' + json.dumps(fragment) + ''';
const selector=document.getElementById('frame');let cleanup;
report.frames.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.id;selector.append(o);});
function mount(){if(cleanup)cleanup();const frame=report.frames[Number(selector.value)];const root=document.getElementById('viewer-root');root.innerHTML=fragment;cleanup=render({parentElement:root,data:{...frame,zero_input_control:report.zero_input_control}});document.getElementById('scores').textContent=`Mean nearest-point distance: Wi-Fi ${report.mean_nn_distance.toFixed(4)} · Zero input ${report.mean_zero_input_nn_distance.toFixed(4)} · Fixed first frame ${report.mean_fixed_frame_nn_distance.toFixed(4)}. Lower is better, standardized units, no ICP alignment.`;}
selector.onchange=mount;mount();</script></html>'''
    (folder/'viewer.html').write_text(document)
    chosen = [report['frames'][i] for i in (0, 3, 5)]
    combined = np.concatenate([np.asarray(f[k]) for f in chosen for k in ('predicted', 'reference')])
    low, high = combined.min(axis=0), combined.max(axis=0)
    pad = (high - low) * .06
    figure = plt.figure(figsize=(14, 9), facecolor='white')
    figure.suptitle('Recorded MM-Fi Wi-Fi → CSI2PointCloud predictions\nStandardized coordinates · Measured reference comparison', fontsize=14)
    for column, frame in enumerate(chosen):
        for row, key, color, label in [(0, 'predicted', '#007f83', 'Wi-Fi model prediction'),
                                       (1, 'reference', '#a75508', 'Measured LiDAR reference')]:
            ax = figure.add_subplot(2, 3, row * 3 + column + 1, projection='3d')
            points = np.asarray(frame[key])
            ax.scatter(*points.T, c=color, s=2, alpha=.55, rasterized=True)
            ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]), ylim=(low[1]-pad[1], high[1]+pad[1]),
                   zlim=(low[2]-pad[2], high[2]+pad[2]), xlabel='X', ylabel='Y',
                   zlabel='Z', title=f"{label}\nFrame {frame['frame_number']:03d}")
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(4))
            ax.tick_params(labelsize=8, pad=1)
            ax.set_box_aspect(np.maximum(high-low, .1))
            ax.view_init(elev=18, azim=-65)
    figure.text(.5, .02, 'Noncommercial research sample: one recording, six selected frames. Fixed first-frame control slightly outperforms changing-input predictions.',
                ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, .92))
    figure.savefig(folder/'comparison.png', dpi=150)
    plt.close(figure)
    print('Exported:', folder/'viewer.html', 'and', folder/'comparison.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=ROOT/'outputs/wifi3d/mmfi-reference')
    export(parser.parse_args().folder)
