"""Scan MUSAN + RIRS_NOISES directories and produce noise.csv / rir.csv
that ``functions/augmentation.py`` reads at training-time.

Run ONCE on the server after extracting the datasets:

    cd /root/autodl-tmp
    tar -xzf musan.tar.gz       # → /root/autodl-tmp/musan/
    unzip rirs_noises.zip       # → /root/autodl-tmp/RIRS_NOISES/

    cd /root/autodl-tmp/CAARMA
    python gen_aug_csv.py \\
        --musan_dir /root/autodl-tmp/musan \\
        --rir_dir   /root/autodl-tmp/RIRS_NOISES \\
        --out_dir   /root/autodl-tmp/CAARMA

Outputs (one column 'wav' with absolute paths):
    noise.csv  — every .wav under MUSAN (noise + music + speech sub-dirs)
    rir.csv    — every .wav under RIRS_NOISES (real + simulated rirs)

The Augmentation class samples uniformly at random from these lists each
__getitem__ call, so the CSV order does not matter.
"""
import argparse
import csv
import os
import sys


def find_wavs(root: str) -> list[str]:
    """Recursively collect every .wav file under root."""
    wavs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.wav'):
                wavs.append(os.path.join(dirpath, fn))
    return wavs


def write_csv(out_path: str, wav_paths: list[str]) -> None:
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['wav'])
        for p in wav_paths:
            writer.writerow([p])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--musan_dir', required=True,
                        help='Path to extracted MUSAN root (e.g. /root/autodl-tmp/musan)')
    parser.add_argument('--rir_dir', required=True,
                        help='Path to extracted RIRS_NOISES root')
    parser.add_argument('--out_dir', default='.',
                        help='Where to write noise.csv / rir.csv (default: cwd)')
    args = parser.parse_args()

    for path, label in [(args.musan_dir, 'MUSAN'), (args.rir_dir, 'RIRS_NOISES')]:
        if not os.path.isdir(path):
            print(f'ERROR: {label} directory not found: {path}', file=sys.stderr)
            sys.exit(1)

    print(f'Scanning MUSAN at {args.musan_dir} ...')
    noise_wavs = find_wavs(args.musan_dir)
    if not noise_wavs:
        print('ERROR: no .wav files found under MUSAN', file=sys.stderr); sys.exit(1)
    print(f'  found {len(noise_wavs)} .wav files')

    print(f'Scanning RIRS_NOISES at {args.rir_dir} ...')
    rir_wavs = find_wavs(args.rir_dir)
    if not rir_wavs:
        print('ERROR: no .wav files found under RIRS_NOISES', file=sys.stderr); sys.exit(1)
    print(f'  found {len(rir_wavs)} .wav files')

    noise_csv = os.path.join(args.out_dir, 'noise.csv')
    rir_csv = os.path.join(args.out_dir, 'rir.csv')
    write_csv(noise_csv, noise_wavs)
    write_csv(rir_csv, rir_wavs)
    print(f'\nWrote {noise_csv}  ({len(noise_wavs)} rows)')
    print(f'Wrote {rir_csv}    ({len(rir_wavs)} rows)')
    print('\nNext: add these paths to config.yaml:')
    print(f'  noise_csv: "{os.path.abspath(noise_csv)}"')
    print(f'  reverb_csv: "{os.path.abspath(rir_csv)}"')


if __name__ == '__main__':
    main()
