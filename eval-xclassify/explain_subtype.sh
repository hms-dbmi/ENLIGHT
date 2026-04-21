
python eval-xclassify/explain_classify.py \
            --ckpt-dir $CKPTDIR \
            --config eval-xclassify/config/GBM-LGG.yml \
            --slide-path $SLIDE_PATH \
            --slide-cropped \
            --slide-multifeat-path $SLIDE_FEAT_PATH


# CKPTDIR=./ckpts DATADIR=./data bash eval-xclassify/explain_subtype.sh