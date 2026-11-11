# Equipment Pilot Data

No images are bundled or downloaded. Collect images only with permission and keep the raw pilot data local. Aim for at least 5 distinct physical objects in each class; the code requires 3 per class to create object-disjoint splits and warns below 5. More objects and varied collection sessions are much more useful than many near-identical frames of one object.

Use one main object per image. Do not photograph identifying labels, people, or unrelated personal information. Vary lighting, background, angle, and distance. Assign a stable physical object ID (for example `kbd-001`) and a collection session ID (for example `session-2026-01`). Do not reuse one object's ID for a different object.

Place files under `data/raw/equipment/images/` and create `data/raw/equipment/manifest.csv` with these columns:

```csv
image_path,label,physical_object_id,collection_session
images/keyboard/kbd-001/session-2026-01/front.jpg,keyboard,kbd-001,session-2026-01
images/mouse/mou-001/session-2026-01/front.jpg,mouse,mou-001,session-2026-01
images/headphones/hph-001/session-2026-01/front.jpg,headphones,hph-001,session-2026-01
```

The examples above illustrate schema only; they are not included as image data. `python -m ml.vision.cli prepare` validates paths, image integrity, classes, exact duplicate contents, and object-disjoint splits. It refuses to train with insufficient distinct objects. The final test split remains local and organiser-only.

The starter freezes ImageNet-pretrained ResNet-18 except its new classifier. AI Lead fine-tunes `layer4` and the classifier using training-only augmentation. Training fetches the torchvision ResNet-18 ImageNet weights only when explicitly invoked. Both inference pipelines use deterministic resize/crop/normalization. A validation-selected confidence threshold is only a low-score caution; it is not unknown-object detection or a reliability guarantee.
