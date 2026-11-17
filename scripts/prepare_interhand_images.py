#!/usr/bin/env python3
"""Restore the released InterHand camera/frame subsets from a local download."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()


def prepare(args):
    manifest = json.loads(args.manifest.read_text())
    selected = args.scene or list(manifest['scenes'])
    unknown = set(selected) - set(manifest['scenes'])
    if unknown:
        raise ValueError(f'Unknown scenes: {sorted(unknown)}')
    pending = []
    existing = 0
    for name in dict.fromkeys(selected):
        spec = manifest['scenes'][name]
        scene = args.scenes_root / name
        gt = json.loads((scene / 'gt_keypoints_3d.json').read_text())
        if sorted(gt, key=int) != spec['frame_ids']:
            raise ValueError(f'{name}: ground-truth frames differ from the release manifest')
        calibration = json.loads((scene / spec['calibration_file']).read_text())
        calibrated = {str(c['camera']) for c in calibration['cameras']}
        if not set(spec['camera_ids']) <= calibrated:
            raise ValueError(f'{name}: missing camera calibration')
        expected_names = {f'image{frame}.jpg' for frame in spec['frame_ids']}
        for folder in scene.glob('camera_*'):
            if folder.is_dir():
                if folder.name.removeprefix('camera_') not in spec['camera_ids']:
                    raise ValueError(f'Unexpected camera directory: {folder}')
                extra = {p.name for p in folder.glob('*.jpg')} - expected_names
                if extra:
                    raise ValueError(f'Unexpected images in {folder}: {sorted(extra)[:3]}')
        for camera in spec['camera_ids']:
            for frame in spec['frame_ids']:
                filename = f'image{frame}.jpg'
                source = (args.images_root / spec['source_sequence'] / f'cam{camera}' / filename).resolve()
                target = scene / f'camera_{camera}' / filename
                if not source.is_file():
                    raise FileNotFoundError(f'Missing source image: {source}')
                if target.exists() or target.is_symlink():
                    if not target.is_file() or not (target.samefile(source) or digest(target) == digest(source)):
                        raise ValueError(f'Refusing to replace a different file: {target}')
                    existing += 1
                else:
                    pending.append((source, target))
    # Validate every source and destination before creating any directories/files.
    if not args.dry_run:
        for source, target in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                # Exclusive creation also prevents an overwrite if a file appeared after preflight.
                with source.open('rb') as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
            else:
                target.symlink_to(source)
    action = 'Would copy' if args.copy else 'Would link'
    if not args.dry_run:
        action = 'Copied' if args.copy else 'Linked'
    print(f'{action} {len(pending)} images; {existing} matching images already present.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-root', type=Path, required=True,
                        help='Original InterHand images directory containing train/ and test/.')
    parser.add_argument('--scenes-root', type=Path, required=True,
                        help='Released scenes directory containing the two scene folders.')
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/interhand-scenes.json')
    parser.add_argument('--scene', action='append', help='Restore only this scene (repeatable); default: both.')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of using absolute symlinks.')
    parser.add_argument('--dry-run', action='store_true', help='Validate all paths without writing files.')
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
