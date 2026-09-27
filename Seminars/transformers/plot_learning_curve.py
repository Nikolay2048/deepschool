"""Render lesson 09 loss in a separate process, without importing PyTorch."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    history = json.load(sys.stdin)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    steps = [row['step'] for row in history]
    ax.plot(steps, [row['train'] for row in history], color='#225ea8',
            label='Train', linewidth=2)
    ax.plot(steps, [row['val'] for row in history], color='#b35806',
            label='Validation', linestyle='--', linewidth=2)
    ax.set(xlabel='Шаг обновления', ylabel='Cross-entropy, нат/символ',
           title='Улучшенный профиль: фиксированные окна по 128 символов')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
