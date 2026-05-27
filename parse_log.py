import re
import csv

def parse_log(log_path, output_csv=None):
    epoch_pattern = re.compile(
        r'Epoch (\d+):.*?'
        r'am_loss=([\d.]+).*?'
        r'am_loss_syn=([\d.]+).*?'
        r'acc=([\d.]+).*?'
        r'g_loss=([\d.]+).*?'
        r'total_loss=([\d.]+).*?'
        r'd_loss=([\d.]+)'
    )
    eer_pattern = re.compile(
        r'cosine EER:\s*([\d.]+)%'
    )
    mindcf2_pattern = re.compile(
        r'cosine minDCF\(10-2\):\s*([\d.]+)'
    )
    mindcf3_pattern = re.compile(
        r'cosine minDCF\(10-3\):\s*([\d.]+)'
    )

    results = []
    current_metrics = None  # None = no open epoch record
    pending = {'eer': None, 'mindcf_2': None, 'mindcf_3': None}

    with open(log_path, 'r') as f:
        content = f.read()

    lines = re.split(r'[\r\n]+', content)

    for line in lines:
        m_eer = eer_pattern.search(line)
        if m_eer:
            val = float(m_eer.group(1))
            if current_metrics is not None: current_metrics['eer'] = val
            else: pending['eer'] = val

        m_dcf2 = mindcf2_pattern.search(line)
        if m_dcf2:
            val = float(m_dcf2.group(1))
            if current_metrics is not None: current_metrics['mindcf_2'] = val
            else: pending['mindcf_2'] = val

        m_dcf3 = mindcf3_pattern.search(line)
        if m_dcf3:
            val = float(m_dcf3.group(1))
            if current_metrics is not None: current_metrics['mindcf_3'] = val
            else: pending['mindcf_3'] = val

        m = epoch_pattern.search(line)
        if m:
            epoch = int(m.group(1)) + 1
            train_vals = {
                'am_loss': float(m.group(2)),
                'am_loss_syn': float(m.group(3)),
                'acc': float(m.group(4)),
                'g_loss': float(m.group(5)),
                'total_loss': float(m.group(6)),
                'd_loss': float(m.group(7)),
            }
            if current_metrics is not None and current_metrics['epoch'] == epoch:
                # Same epoch: later tqdm update — keep eval metrics, update train metrics
                current_metrics.update(train_vals)
            else:
                # New epoch — flush previous record first
                if current_metrics is not None:
                    results.append(dict(current_metrics))
                current_metrics = {
                    'epoch': epoch,
                    **train_vals,
                    'eer': pending['eer'],
                    'mindcf_2': pending['mindcf_2'],
                    'mindcf_3': pending['mindcf_3'],
                }
                pending = {'eer': None, 'mindcf_2': None, 'mindcf_3': None}

    # Flush trailing record
    if current_metrics is not None:
        for k in ('eer', 'mindcf_2', 'mindcf_3'):
            if current_metrics[k] is None and pending[k] is not None:
                current_metrics[k] = pending[k]
        results.append(dict(current_metrics))

    # Deduplicate: keep last entry per epoch
    seen = {}
    for r in results:
        seen[r['epoch']] = r
    results = [seen[k] for k in sorted(seen)]

    # Print table
    header = f"{'Epoch':>6} {'am_loss':>9} {'am_loss_syn':>12} {'acc':>7} {'g_loss':>8} {'d_loss':>8} {'total_loss':>11} {'EER(%)':>8} {'mDCF-2':>8} {'mDCF-3':>8}"
    print(header)
    print('-' * len(header))
    for r in results:
        eer_str   = f"{r['eer']:.2f}"    if r['eer']      is not None else '  -'
        dcf2_str  = f"{r['mindcf_2']:.4f}" if r['mindcf_2'] is not None else '     -'
        dcf3_str  = f"{r['mindcf_3']:.4f}" if r['mindcf_3'] is not None else '     -'
        print(f"{r['epoch']:>6} {r['am_loss']:>9.3f} {r['am_loss_syn']:>12.3f} {r['acc']:>7.2f} "
              f"{r['g_loss']:>8.3f} {r['d_loss']:>8.3f} {r['total_loss']:>11.3f} "
              f"{eer_str:>8} {dcf2_str:>8} {dcf3_str:>8}")

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            fieldnames = ['epoch', 'am_loss', 'am_loss_syn', 'acc', 'g_loss', 'd_loss', 'total_loss', 'eer', 'mindcf_2', 'mindcf_3']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved to {output_csv}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', nargs='?', default='train_log.txt')
    parser.add_argument('--csv', default=None, metavar='OUTPUT.csv')
    args = parser.parse_args()
    parse_log(args.log_file, args.csv)
