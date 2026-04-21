#!/bin/bash

HFTOKEN=your_huggingface_token_here
SLIDE_CROPPED=${SLIDE_CROPPED:-1}

python preprocess/slide_visualenc.py --backbone ENLIGHT \
                                --ckpt-path $CKPTDIR/enlight-fm/enlight-visual-encoder.pt \
                                --batch-size 16 \
                                --slide-path $SLIDE_PATH \
                                ${SLIDE_CROPPED:+--slide-cropped} \
                                --output-path $SLIDE_FEAT_PATH \
                                --hf-token $HFTOKEN