"""Fine-tune one RDN model. Training only starts when this script is executed."""
import argparse
import json
from lab_code.experiment import get_config, run_experiment, DEFAULT_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=(*DEFAULT_MODELS, 'cnn'), required=True)
    parser.add_argument('--data-root', help='Existing RDN package root; or set RDN_DATA_ROOT')
    parser.add_argument('--run-name')
    parser.add_argument('--checkpoint', help='Existing trained model for checkpoint continuation')
    parser.add_argument('--initialization', choices=['pretrained', 'checkpoint'])
    parser.add_argument('--weights-dir', help='Local/cached VMamba ImageNet checkpoint directory')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--no-test', action='store_true', help='Train/validate only, without final test evaluation')
    parser.add_argument('--allow-cpu', action='store_true', help='Explicit slow CPU mode for non-VMamba models')
    parser.add_argument('--print-config', action='store_true', help='Print effective config, without data/weights/training')
    args = parser.parse_args()
    overrides = {k: v for k, v in {'initialization': args.initialization, 'max_epochs': args.epochs,
                 'batch_size': args.batch_size, 'accumulation_steps': args.accumulation_steps,
                 'seed': args.seed, 'precision': args.precision}.items() if v is not None}
    if args.epochs is not None:
        overrides['warmup_epochs'] = min(get_config(args.model)['warmup_epochs'], args.epochs - 1)
    if args.print_config:
        print(json.dumps(get_config(args.model, overrides), indent=2))
        return
    run_experiment(args.model, data_root=args.data_root, overrides=overrides, run_name=args.run_name,
                   checkpoint=args.checkpoint, weights_dir=args.weights_dir,
                   run_training=not args.eval_only, run_test=not args.no_test,
                   require_cuda=not args.allow_cpu)


if __name__ == '__main__':
    main()
