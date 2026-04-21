python eval-xclassify/explain_classify.py \
            --ckpt-dir $CKPTDIR \
            --config eval-xclassify/config/GBM_DEL_SMARCA4.yml \
            --slide-path $SLIDE_PATH \
            --slide-cropped \
            --slide-multifeat-path $SLIDE_FEAT_PATH
