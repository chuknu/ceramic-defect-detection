import os
import shutil
import json

# Standard COCO names (80 classes)
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Denylist of COCO classes that indicate non-tile images
DENY = set([
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'dog','cat','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','pizza','donut','cake',
    'chair','couch','dining table','toilet','tv','laptop','cell phone','microwave','oven','refrigerator'
])


def categorize(pseudo_root='datasets/pseudo_web'):
    imgs_root = os.path.join(pseudo_root, 'images')
    labels_root = os.path.join(pseudo_root, 'labels')
    annotated_root = os.path.join(pseudo_root, 'annotated')

    out_base = os.path.join('datasets', 'filtered')
    os.makedirs(out_base, exist_ok=True)

    report = {'tiles': [], 'non_tiles': [], 'uncertain': []}

    for split in ['train', 'val']:
        imgs_dir = os.path.join(imgs_root, split)
        labels_dir = os.path.join(labels_root, split)
        if not os.path.isdir(imgs_dir):
            continue

        dest_tiles = os.path.join(out_base, 'tiles', split)
        dest_non = os.path.join(out_base, 'non_tiles', split)
        dest_uncertain = os.path.join(out_base, 'uncertain', split)
        for d in [dest_tiles, dest_non, dest_uncertain]:
            os.makedirs(d, exist_ok=True)

        for fn in sorted(os.listdir(imgs_dir)):
            img_path = os.path.join(imgs_dir, fn)
            name, _ = os.path.splitext(fn)
            label_path = os.path.join(labels_dir, f"{name}.txt")

            if not os.path.isfile(label_path):
                # No labels -> uncertain
                dest = dest_uncertain
                report['uncertain'].append(fn)
            else:
                try:
                    with open(label_path, 'r') as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                except Exception:
                    lines = []

                if not lines:
                    dest = dest_uncertain
                    report['uncertain'].append(fn)
                else:
                    classes = set()
                    for l in lines:
                        parts = l.split()
                        try:
                            cid = int(parts[0])
                            cname = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else str(cid)
                            classes.add(cname)
                        except Exception:
                            continue

                    if classes & DENY:
                        dest = dest_non
                        report['non_tiles'].append({'file': fn, 'classes': list(classes)})
                    else:
                        dest = dest_tiles
                        report['tiles'].append({'file': fn, 'classes': list(classes)})

            # Move image, label, and annotated (if present)
            shutil.copy2(img_path, os.path.join(dest, fn))
            if os.path.isfile(label_path):
                shutil.copy2(label_path, os.path.join(dest, os.path.basename(label_path)))
            ann = os.path.join(annotated_root, fn)
            if os.path.isfile(ann):
                shutil.copy2(ann, os.path.join(dest, os.path.basename(ann)))

    # Write report
    report_path = os.path.join(out_base, 'report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('Filtering complete.')
    print('Tiles:', len(report['tiles']))
    print('Non-tiles:', len(report['non_tiles']))
    print('Uncertain:', len(report['uncertain']))
    print('Report saved to', report_path)


if __name__ == '__main__':
    categorize()
