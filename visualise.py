"""
Intelligence Kernel — Results Visualisation
Generates publication-quality plots from experiment results.
Run after experiments.py:  python visualise.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results.json')
OUT_DIR      = os.path.dirname(__file__)

# ── Style ────────────────────────────────────────────────────────
COLORS = {
    'coupled':   '#2c4a7c',
    'baseline':  '#b4b2a9',
    'causal':    '#1D9E75',
    'corr':      '#D85A30',
    'confirmed': '#1a5c38',
    'trend':     '#854F0B',
    'rejected':  '#A32D2D',
    'accent':    '#7c3c1a',
}

def style():
    plt.rcParams.update({
        'font.family':       'serif',
        'font.size':         11,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linewidth':    0.5,
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
        'xtick.direction':   'out',
        'ytick.direction':   'out',
    })

# ══════════════════════════════════════════════════════════════════

def plot_experiment_2(ax, data):
    """The KEY METRIC: compression ratio over iterations."""
    coupled  = data.get('coupled_ratios', [])
    baseline = data.get('baseline_ratios', [])
    n = min(len(coupled), len(baseline))

    iters = np.arange(n)

    # Raw values (light)
    ax.plot(iters, coupled[:n],  alpha=0.25, color=COLORS['coupled'],  linewidth=0.8)
    ax.plot(iters, baseline[:n], alpha=0.25, color=COLORS['baseline'], linewidth=0.8)

    # Smoothed rolling mean (window=8)
    w = 8
    if n >= w:
        def smooth(arr):
            return np.convolve(arr, np.ones(w)/w, mode='valid')
        smooth_iters   = iters[w-1:]
        smooth_coupled  = smooth(coupled[:n])
        smooth_baseline = smooth(baseline[:n])
        ax.plot(smooth_iters, smooth_coupled,  color=COLORS['coupled'],
                linewidth=2.2, label='Coupled (L0↔L1 feedback)')
        ax.plot(smooth_iters, smooth_baseline, color=COLORS['baseline'],
                linewidth=2.2, label='Baseline (no feedback)',
                linestyle='--')

    # Trend line for coupled
    slope = data.get('slope_coupled', 0)
    intercept = np.mean(coupled[:n]) - slope * (n/2)
    trend_y = slope * iters + intercept
    ax.plot(iters, trend_y, color=COLORS['coupled'], linewidth=1.0,
            linestyle=':', alpha=0.7)

    # Annotation
    p = data.get('p_coupled', 1.0)
    verdict = data.get('verdict', '')
    color = (COLORS['confirmed'] if 'CONFIRMED' in verdict
             else COLORS['trend'] if 'TREND' in verdict
             else COLORS['rejected'])

    improvement = 0
    if len(coupled) >= 40:
        improvement = (np.mean(coupled[:20]) - np.mean(coupled[-20:])) / np.mean(coupled[:20]) * 100
    ax.text(0.98, 0.96, f"Δ = {improvement:.1f}%  p={p:.3f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Compression ratio (lower = better)')
    ax.set_title('Experiment 2: Self-Accelerating Compression\n'
                 'The Key Metric', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)

    verdict_short = ("✓ CONFIRMED" if 'CONFIRMED' in verdict
                     else "~ TREND" if 'TREND' in verdict
                     else "✗ NOT CONFIRMED")
    ax.text(0.02, 0.04, verdict_short, transform=ax.transAxes,
            fontsize=9, color=color, fontweight='bold')


def plot_experiment_3(ax, data):
    """Distribution robustness: causal vs baseline MSE by shift."""
    results = data.get('results_by_shift', {})
    if not results:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        return

    shifts = sorted(float(k) for k in results.keys())
    causal_mse  = [results[str(round(s,2)) if isinstance(list(results.keys())[0], str)
                           else round(s,2)].get('causal', 0) for s in shifts]
    base_mse    = [results[str(round(s,2)) if isinstance(list(results.keys())[0], str)
                           else round(s,2)].get('baseline', 0) for s in shifts]

    # Try both string and float keys
    causal_mse, base_mse = [], []
    for s in shifts:
        key_options = [s, round(s, 2), str(s), str(round(s, 2))]
        for k in key_options:
            if k in results:
                causal_mse.append(results[k].get('causal', 0))
                base_mse.append(results[k].get('baseline', 0))
                break
        else:
            causal_mse.append(0)
            base_mse.append(0)

    x = np.arange(len(shifts))
    w = 0.35
    bars_c = ax.bar(x - w/2, causal_mse, w, color=COLORS['causal'],
                    label='Causally grounded L2', alpha=0.85)
    bars_b = ax.bar(x + w/2, base_mse,   w, color=COLORS['corr'],
                    label='Correlation baseline', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"×{s:.2f}" for s in shifts], fontsize=9)
    ax.set_xlabel('Distribution shift (weight multiplier)')
    ax.set_ylabel('Mean squared prediction error')
    ax.set_title('Experiment 3: Distribution Robustness\nCausal Grounding vs Correlation Baseline',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    p = data.get('p', 1.0)
    verdict = data.get('verdict', '')
    color = (COLORS['confirmed'] if 'CONFIRMED' in verdict else COLORS['rejected'])
    verdict_short = "✓ CONFIRMED" if 'CONFIRMED' in verdict else "✗ NOT CONFIRMED"
    ax.text(0.98, 0.96, f"p={p:.4f}", transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color=color, fontweight='bold')
    ax.text(0.02, 0.96, verdict_short, transform=ax.transAxes,
            fontsize=9, color=color, fontweight='bold')


def plot_experiment_4(ax, data):
    """Source attribution confusion matrix."""
    confusion = data.get('confusion', {})
    classes   = ['compression', 'causal', 'simulation']
    labels    = ['L0\ncompress', 'L1\ncausal', 'L2\nsimulate']

    matrix = np.zeros((3, 3))
    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            val = confusion.get(true_c, {})
            if isinstance(val, dict):
                matrix[i][j] = val.get(pred_c, 0)

    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Predicted source', fontsize=10)
    ax.set_ylabel('True source',      fontsize=10)
    ax.set_title('Experiment 4: Source Attribution\nConfusion Matrix',
                 fontsize=11, fontweight='bold')

    for i in range(3):
        for j in range(3):
            val = int(matrix[i][j])
            color = 'white' if matrix[i][j] > matrix.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    color=color, fontweight='bold', fontsize=13)

    acc = data.get('accuracy', 0)
    p   = data.get('p', 1.0)
    verdict = data.get('verdict', '')
    color = (COLORS['confirmed'] if 'CONFIRMED' in verdict else COLORS['rejected'])
    verdict_short = "✓ CONFIRMED" if 'CONFIRMED' in verdict else "~ TREND" if 'TREND' in verdict else "✗"

    ax.text(0.98, 0.04, f"Accuracy: {acc:.0%}  p={p:.4f}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=color, fontweight='bold')
    ax.text(0.02, 0.04, verdict_short, transform=ax.transAxes,
            fontsize=9, color=color, fontweight='bold')


def plot_summary(ax, results):
    """Summary scorecard."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    propositions = [
        ("P1", "Causal disambiguation",     results.get('exp1', {}).get('verdict', 'NOT RUN')),
        ("P2", "Self-accelerating compress", results.get('exp2', {}).get('verdict', 'NOT RUN')),
        ("P3", "Distribution robustness",    results.get('exp3', {}).get('verdict', 'NOT RUN')),
        ("P4", "Source attribution",         results.get('exp4', {}).get('verdict', 'NOT RUN')),
    ]

    ax.text(5, 9.3, "Proposition Summary", ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1a1812')

    confirmed = sum(1 for _, _, v in propositions if 'CONFIRMED' in v)

    for i, (label, desc, verdict) in enumerate(propositions):
        y = 7.5 - i * 1.8
        is_confirmed = 'CONFIRMED' in verdict
        is_trend     = 'TREND' in verdict and not is_confirmed
        color = (COLORS['confirmed'] if is_confirmed
                 else COLORS['trend']    if is_trend
                 else COLORS['rejected'])
        icon = "✓" if is_confirmed else "~" if is_trend else "✗"

        rect = FancyBboxPatch((0.3, y - 0.55), 9.4, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=color + '18',
                              edgecolor=color, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(0.9, y + 0.05, f"{icon} {label}", ha='left', va='center',
                fontsize=12, fontweight='bold', color=color)
        ax.text(2.5, y + 0.05, desc, ha='left', va='center',
                fontsize=10, color='#3d3d3a')
        short = ('Confirmed' if is_confirmed
                 else 'Trend' if is_trend else 'Not confirmed')
        ax.text(9.5, y + 0.05, short, ha='right', va='center',
                fontsize=9, color=color, style='italic')

    # Bottom score
    score_color = (COLORS['confirmed'] if confirmed >= 3
                   else COLORS['trend'] if confirmed >= 2
                   else COLORS['rejected'])
    ax.text(5, 0.6, f"{confirmed}/4 propositions confirmed",
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=score_color)

    rec = ("→ Submit: NeurIPS/ICLR main track" if confirmed >= 3
           else "→ Submit: Workshop track" if confirmed >= 2
           else "→ Revise and retest")
    ax.text(5, 0.1, rec, ha='center', va='center',
            fontsize=9, color=score_color, style='italic')


# ══════════════════════════════════════════════════════════════════

def main():
    style()

    if not os.path.exists(RESULTS_PATH):
        print(f"No results.json found. Run experiments.py first.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        'Intelligence Kernel — Experimental Validation Results\n'
        'Testing Four Propositions from the Coupling Architecture',
        fontsize=14, fontweight='bold', y=0.98, color='#1a1812')

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97,
                           top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Experiment 2 top-left (most important — gets prime position)
    if 'exp2' in results:
        plot_experiment_2(ax1, results['exp2'])
    else:
        ax1.text(0.5, 0.5, 'Exp 2 not run', ha='center', va='center',
                 transform=ax1.transAxes, color='gray')
        ax1.set_title('Experiment 2 (not run)')

    # Summary scorecard top-right
    plot_summary(ax2, results)

    # Experiment 3 bottom-left
    if 'exp3' in results:
        plot_experiment_3(ax3, results['exp3'])
    else:
        ax3.text(0.5, 0.5, 'Exp 3 not run', ha='center', va='center',
                 transform=ax3.transAxes, color='gray')
        ax3.set_title('Experiment 3 (not run)')

    # Experiment 4 bottom-right
    if 'exp4' in results:
        plot_experiment_4(ax4, results['exp4'])
    else:
        ax4.text(0.5, 0.5, 'Exp 4 not run', ha='center', va='center',
                 transform=ax4.transAxes, color='gray')
        ax4.set_title('Experiment 4 (not run)')

    out_path = os.path.join(OUT_DIR, 'results_figure.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Figure saved: {out_path}")
    return out_path

if __name__ == '__main__':
    main()
