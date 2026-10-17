"""Collect existing results; does not perform inference or training."""
import argparse
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'summary.csv')
    args = parser.parse_args()
    rows = []
    for info_path in sorted(args.results.glob('*/*/model_info.json')):
        folder = info_path.parent
        info = json.loads(info_path.read_text())
        summary_path = folder / 'training_validation_metrics.json'
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        test_path = folder / 'test_metrics.json'
        test = json.loads(test_path.read_text()) if test_path.exists() else {}
        rows.append({'model': info['model'], 'initialization': 'pretrained' if info['pretrained'] else 'checkpoint',
                     'total_parameters': info['total_parameters'], 'trainable_parameters': info['trainable_parameters'],
                     'epochs': summary['epochs_completed'], 'selected_epoch': summary['selected_epoch'],
                     'train_accuracy': summary['train']['accuracy'],
                     'validation_accuracy': summary['validation']['accuracy'],
                     'test_accuracy': test.get('accuracy'), 'test_balanced_accuracy': test.get('balanced_accuracy'),
                     'test_macro_f1': test.get('macro_f1'), 'test_predictions_class0': test.get('prediction_counts', [None, None])[0],
                     'test_predictions_class1': test.get('prediction_counts', [None, None])[1], 'run': str(folder)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'{len(rows)} completed runs found; ' + (f'saved {args.output}' if rows else 'no summary written'))
