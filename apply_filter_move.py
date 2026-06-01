import os
import json
import shutil

def apply_moves(report_path='datasets/filtered/report.json', pseudo_root='datasets/pseudo_web'):
    with open(report_path, 'r') as f:
        report = json.load(f)

    imgs_root = os.path.join(pseudo_root, 'images')
    labels_root = os.path.join(pseudo_root, 'labels')
    annotated_root = os.path.join(pseudo_root, 'annotated')

    removed_base = os.path.join('datasets', 'removed_non_tiles')
    uncertain_base = os.path.join('datasets', 'uncertain')
    os.makedirs(removed_base, exist_ok=True)
    os.makedirs(uncertain_base, exist_ok=True)

    # Move non_tiles
    for item in report.get('non_tiles', []):
        fn = item['file']
        # find in train/val
        moved = False
        for split in ['train', 'val']:
            src_img = os.path.join(imgs_root, split, fn)
            src_lbl = os.path.join(labels_root, split, os.path.splitext(fn)[0] + '.txt')
            src_ann = os.path.join(annotated_root, fn)
            dest_dir = os.path.join(removed_base, split)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isfile(src_img):
                shutil.move(src_img, os.path.join(dest_dir, fn))
                moved = True
            if os.path.isfile(src_lbl):
                shutil.move(src_lbl, os.path.join(dest_dir, os.path.basename(src_lbl)))
            if os.path.isfile(src_ann):
                shutil.move(src_ann, os.path.join(dest_dir, os.path.basename(src_ann)))
        if not moved:
            print('Warning: non-tile file not found in expected locations:', fn)

    # Move uncertain -> datasets/uncertain
    for fn in report.get('uncertain', []):
        moved = False
        for split in ['train', 'val']:
            src_img = os.path.join(imgs_root, split, fn)
            src_lbl = os.path.join(labels_root, split, os.path.splitext(fn)[0] + '.txt')
            src_ann = os.path.join(annotated_root, fn)
            dest_dir = os.path.join(uncertain_base, split)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.isfile(src_img):
                shutil.move(src_img, os.path.join(dest_dir, fn))
                moved = True
            if os.path.isfile(src_lbl):
                shutil.move(src_lbl, os.path.join(dest_dir, os.path.basename(src_lbl)))
            if os.path.isfile(src_ann):
                shutil.move(src_ann, os.path.join(dest_dir, os.path.basename(src_ann)))
        if not moved:
            print('Warning: uncertain file not found in expected locations:', fn)

    print('Apply moves complete.')


if __name__ == '__main__':
    apply_moves()
