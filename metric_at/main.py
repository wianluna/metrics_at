import argparse
from pathlib import Path

from metric_at.trainer import AdversarialTrainer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Adversarial Training')
    parser.add_argument('--config', type=Path, help='Path to training/testing configuration')
    return parser.parse_args()


def main():
    args = parse_arguments()
    AdversarialTrainer.run(args.config)


if __name__ == '__main__':
    main()
