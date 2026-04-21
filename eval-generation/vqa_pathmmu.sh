QA_DIR=$(grep QA_DIR utils/constants.py | sed 's/.*= *"\(.*\)"/\1/')
# BASE=/n/data2/hms/dbmi/kyu/lab/xug751/datavl/pathmmu

QSFILE=$QA_DIR/ques_pathmmu.jsonl
ASFILE=$QA_DIR/ans_pathmmu.jsonl 
echo $QSFILE $ASFILE

python eval-generation/vqa_batch_infer.py \
                        --model-gen $CKPTDIR/enlight-fm \
                        --image-folder $BASE \
                        --question-file $QSFILE \
                        --answers-file $ASFILE \
                        --temperature 0 \
                        --batch-size 8 \
                        --num-workers 1 \
                        --max_new_tokens 2048