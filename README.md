# ENLIGHT: Interpretable Multimodal AI for Grounded Cancer Pathology Diagnosis and Molecular Profiling

![teaser](teaser.png)

Artificial intelligence (AI)-powered computational pathology has emerged as an increasingly useful tool in cancer evaluation, augmenting clinical interpretation and uncovering previously unrecognized relationships between tissue morphology and underlying molecular alterations. Recent advances in pathology foundation models have enabled diagnostic workflows with unprecedented scalability and efficiency. However, standard AI models remain black boxes, offering limited interpretability and insufficient pathological grounding to justify their assessments. Here we establish **E**xplainable **N**eoplasm **L**earning **i**n **G**rounded **H**istology **T**erms (ENLIGHT), a large multimodal model (LMM) designed to systematically identify cancer diagnosis, subtypes, and genetic alterations across 28 organs. We trained ENLIGHT on 38.36 million pathology image-text pairs and evaluated it on 5.68 million independent validation samples across 40 patient cohorts from 44 institutions worldwide, covering six categories of core diagnostic tasks. ENLIGHT demonstrates strong generalizability in zero-shot classification, cancer subtyping, cross-modal retrieval, visual question answering, report generation, and molecular profile prediction. Across these tasks, it consistently outperforms state-of-the-art (SOTA) pathology models tailored to predictive objectives by up to 12\%, and generative tasks by up to 281\% (mean ROUGE-L for six open-ended visual question answer benchmarks) on independent, unseen cohorts. Importantly, ENLIGHT explains its diagnostic decisions using interpretable pathological concepts that align with established medical knowledge, while uncovering novel links between tissue morphology and molecular alterations. By integrating the reasoning capabilities of LMMs with interpretable pathology grounding, ENLIGHT provides a versatile, scalable, and transparent platform to advance biomedical research, education, and clinical decision support in pathology.

Gong X et al. (under review).

## Install environment

See [environment.md](environment.md) for setup instructions.

## Download checkpoints

Download from [GoogleDrive](https://drive.google.com/drive/folders/1vp7jRgdy-SWwXx_4kuRqd3ZjIaLrW6c1?usp=drive_link) and set the path to $CKPTDIR

Or download via command line:

```bash
wget -r -nd "https://drive.google.com/drive/folders/1vp7jRgdy-SWwXx_4kuRqd3ZjIaLrW6c1?usp=drive_link" -P $CKPTDIR
```

## Download example data

Download from [GoogleDrive](https://drive.google.com/drive/folders/1xCp2AOyz_euA0W0jvHjtxSFJqKaLsD29?usp=drive_link) and set the path to $DATADIR

Or download via command line:

```bash
wget -r -nd "https://drive.google.com/drive/folders/1xCp2AOyz_euA0W0jvHjtxSFJqKaLsD29?usp=drive_link" -P $DATADIR
```

## Zero-shot discrimination tasks

See [eval-zeroshot/dataset.md](eval-zeroshot/dataset.md) for the full list of datasets and download links.
Set the data path to $BASE

##### Evaluate cancer grading

```
python eval-zeroshot/zeroshot_classification.py --database $BASE --data AGGC22 --pretrained_path $CKPTDIR/enlight-fm/enlight-visual-encoder.pt --task grading
```

##### Evaluate microenvironment classification

```
python eval/zeroshot_classification.py --database $BASE --data SPIDER_colon --pretrained_path $CKPTDIR/enlight-fm/enlight-visual-encoder.pt 
```

##### Evaluate image-to-text and text-to-image retrieval

```
python eval/zeroshot_retrieval.py --database $BASE --data ut-0 --pretrained_path $CKPTDIR/enlight-fm/enlight-visual-encoder.pt
```

## Generation tasks

### Patch-level VQA

##### Download dataset

PathMMU: https://huggingface.co/datasets/jamessyx/PathMMU

PathVQA: https://huggingface.co/datasets/dz-osamu/PathVQA

Set the path to $IMG_DIR

##### Preprocess to inference format

```
python eval-generation/format_vqa_batch.py $IMG_DIR pathmmu
```

##### Batch infer to answer

```
BASE=$IMG_DIR CKPTDIR=$CKPTDIR bash eval-generation/vqa_pathmmu.sh
```

### Slide-level VQA

##### From explicitly cropped H5 file
```
CKPTDIR=$CKPTDIR QUESTION=$YourQuery SLIDE_CROPPED=1 SLIDE_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2.h5 bash eval-generation/vqa_slide.sh
```

##### Or from raw SVS
```
CKPTDIR=$CKPTDIR QUESTION=$YourQuery SLIDE_CROPPED=0 SLIDE_PATH=$SVS_PATH bash eval-generation/vqa_slide.sh
```

### Slide-level report generation

##### From explicitly cropped H5 file
```
CKPTDIR=$CKPTDIR SLIDE_CROPPED=1 SLIDE_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2.h5 bash eval-generation/report_generate_slide.sh
```

##### Or from raw SVS
```
CKPTDIR=$CKPTDIR SLIDE_CROPPED=0 SLIDE_PATH=$SVS_PATH bash eval-generation/vqa_slide.sh
```

## Explainable Classification tasks

### Feature extraction for slides 

To get started quickly, this step can be skipped — the pre-extracted `*_Feat8.h5` files in $DATADIR are ready to use for the steps below.

(optional) Extract ENLIGHT features from cropped slide patch

```
CKPTDIR=$CKPTDIR SLIDE_CROPPED=1 SLIDE_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2.h5 SLIDE_FEAT_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2_Feat8.h5 bash preprocess/slide_visualenc.sh
```

(optional) Extract ENLIGHT features from raw SVS

```
CKPTDIR=$CKPTDIR SLIDE_CROPPED=0 SLIDE_PATH=$SVS_PATH SLIDE_FEAT_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2_Feat8.h5 bash preprocess/slide_visualenc.sh
```

(optional) Extract features from additional backbones: GIGA, CONCH, CHIEF, UNI, LUNIT, VIRCHOW, HOPT


### Classify and Explain Subtyping

```
CKPTDIR=$CKPTDIR SLIDE_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2.h5 SLIDE_FEAT_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2_Feat8.h5 bash eval-xclassify/explain_subtype.sh
```

### Classify and Explain Molecular Alteration

```
CKPTDIR=$CKPTDIR SLIDE_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2.h5 SLIDE_FEAT_PATH=$DATADIR/TCGA-06-0122-01Z-00-DX2_Feat8.h5 bash eval-xclassify/explain_molecular.sh
```


## Acknowledgements

We thank the following open-source repositories:

[DSMIL](https://github.com/binli123/dsmil-wsi)

[open_clip](https://github.com/mlfoundations/open_clip)

[LLaVA](https://github.com/haotian-liu/LLaVA)

[Quilt-1M](https://github.com/wisdomikezogwo/quilt1m)

[PathAssist](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology)

## Issues

Please open new threads or address all questions to [xuan_gong@hms.harvard.edu](mailto:xuan_gong@hms.harvard.edu) or [Kun-Hsing_Yu@hms.harvard.edu](mailto:Kun-Hsing_Yu@hms.harvard.edu)
