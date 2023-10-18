# Adversarial training for NR-IQA models

### Install Requirements
1. Install [PDM](https://pdm.fming.dev/latest/)
2. Run
```bash
pdm install 
```

### Download Dataset and Model
Download the [KonIQ-10k](http://database.mmsp-kn.de/koniq-10k-database.html) 1024x768. Then run the following ln commands in the root of the repo.

```bash
cat your_downloaded_path/KonIQ-10k.tar.gz* | tar -xzf - # your_downloaded_path is your path to the downloaded files for KonIQ-10k dataset
ln -s koniq10k/images/ KonIQ-10k
```
### Usage
1. Change data and log paths in training configuration file (`presets/train.yaml`) 
2. Run training with
```bash
train --config presets/train.yaml
```

(Do not use more than one gpu, I'm not sure parallel training works correctly)
