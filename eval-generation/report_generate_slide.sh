SLIDE_CROPPED=${SLIDE_CROPPED:-1}

python eval-generation/infer_slide.py \
                    --model-gen $CKPTDIR/enlight-fm \
                    ${SLIDE_CROPPED:+--slide-cropped} \
                    --slide-path $SLIDE_PATH