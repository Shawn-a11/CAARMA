"""Parse a 4-GPU DDP training log (default: train_ddp4.log).

Behaviour mirrors parse_log_algo2.py — same epoch / EER / minDCF regexes —
but additionally:
  * defaults to the DDP log filename and CSV name
  * detects and reports the number of DDP ranks
  * silently dedups the duplicated progress-bar lines tqdm prints under DDP
"""
import re
import csv
import os


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
    eer_pattern     = re.compile(r'cosine EER:\s*([\d.]+)%')
    mindcf2_pattern = re.compile(r'cosine minDCF\(10-2\):\s*([\d.]+)')
    mindcf3_pattern = re.compile(r'cosine minDCF\(10-3\):\s*([\d.]+)')
    rank_pattern    = re.compile(r'GLOBAL_RANK:\s*(\d+),\s*MEMBER:\s*\d+/(\d+)')

    results = []
    current_metrics = None
    pending = {'eer': None, 'mindcf_2': None, 'mindcf_3': None}
    ddp_world_size = None

    with open(log_path, 'r') as f:
        content = f.read()

    lines = re.split(r'[\r\n]+', content)

    for line in lines:
        m_rank = rank_pattern.search(line)
        if m_rank:
            ddp_world_size = int(m_rank.group(2))

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
                'am_loss':     float(m.group(2)),
                'am_loss_syn': float(m.group(3)),
                'acc':         float(m.group(4)),
                'g_loss':      float(m.group(5)),
                'total_loss':  float(m.group(6)),
                'd_loss':      float(m.group(7)),
            }
            if current_metrics is not None and current_metrics['epoch'] == epoch:
                current_metrics.update(train_vals)
            else:
                if current_metrics is not None:
                    results.append(dict(current_metrics))
                current_metrics = {
                    'epoch': epoch,
                    **train_vals,
                    'eer':      pending['eer'],
                    'mindcf_2': pending['mindcf_2'],
                    'mindcf_3': pending['mindcf_3'],
                }
                pending = {'eer': None, 'mindcf_2': None, 'mindcf_3': None}

    if current_metrics is not None:
        for k in ('eer', 'mindcf_2', 'mindcf_3'):
            if current_metrics[k] is None and pending[k] is not None:
                current_metrics[k] = pending[k]
        results.append(dict(current_metrics))

    # Dedup: tqdm refreshes + multi-rank prints produce duplicate epoch rows.
    seen = {}
    for r in results:
        seen[r['epoch']] = r
    results = [seen[k] for k in sorted(seen)]

    # ── Print table ──────────────────────────────────────────────────────
    if ddp_world_size is not None:
        print(f"[DDP] detected world_size = {ddp_world_size}")
    header = (f"{'Epoch':>6} {'am_loss':>9} {'am_loss_syn':>12} {'acc':>7} "
              f"{'g_loss':>8} {'d_loss':>8} {'total_loss':>11} "
              f"{'EER(%)':>8} {'mDCF-2':>8} {'mDCF-3':>8}")
    print(header)
    print('-' * len(header))
    for r in results:
        eer_s  = f"{r['eer']:.2f}"      if r['eer']      is not None else '-'
        dcf2_s = f"{r['mindcf_2']:.4f}" if r['mindcf_2'] is not None else '-'
        dcf3_s = f"{r['mindcf_3']:.4f}" if r['mindcf_3'] is not None else '-'
        print(f"{r['epoch']:>6} {r['am_loss']:>9.3f} {r['am_loss_syn']:>12.3f} "
              f"{r['acc']:>7.2f} {r['g_loss']:>8.3f} {r['d_loss']:>8.3f} "
              f"{r['total_loss']:>11.3f} {eer_s:>8} {dcf2_s:>8} {dcf3_s:>8}")

    # ── Append-safe CSV write ────────────────────────────────────────────
    if output_csv:
        fieldnames = ['epoch', 'am_loss', 'am_loss_syn', 'acc', 'g_loss',
                      'd_loss', 'total_loss', 'eer', 'mindcf_2', 'mindcf_3']

        existing_epochs = set()
        if os.path.exists(output_csv):
            with open(output_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_epochs.add(int(row['epoch']))

        new_rows = [r for r in results if r['epoch'] not in existing_epochs]

        if new_rows:
            write_header = not os.path.exists(output_csv)
            with open(output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(new_rows)
            print(f"\nAppended {len(new_rows)} new epoch(s) to {output_csv}")
        else:
            print(f"\nNo new epochs to append (all already in {output_csv})")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse 4-GPU DDP training log.')
    parser.add_argument('log_file', nargs='?', default='train_ddp4.log')
    parser.add_argument('--csv', default=None, metavar='OUTPUT.csv',
                        help='append-safe CSV output (default: none)')
    args = parser.parse_args()
    parse_log(args.log_file, args.csv)
