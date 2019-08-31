# The GANdam project

A Generative Adversarial Network (GAN) applied to a dataset of images of mechas from the anime Mobile Suit Gundam. Read the full blog post [here](https://gazoche.xyz/the-gandam-project.html).

## Instructions

### Step 1 : data scraping

```
cd Code/scraping/
python scrape_images_urls.py
python download_images.py
```
Intermediate steps are saved to the `Generated/` folder.

### Step 2 : preprocessing

```
cd Code/preprocessing/
python flatten_dataset.py
```
You must then create the folder `Generated/preprocessing/manually_selected/` and manually copy there the images you want from `Generated/preprocessing/flattened_images/`.

Then :
```
python resize_images.py
python augment_images.py
```

### Step 3 : training

```
cd Code/training/
python train.py
```
Checkpoints and final model are saved at `Generated/training/models/`, and generator samples at `Generated/training/training_samples/`.

### Step 4 : sampling the generator

```
cd Code/testing/
python sample_generator.py
```
Samples will be saved at `Generated/testing/samples/`.
