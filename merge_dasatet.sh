DATASET_DIR="RoadDamageDataset"
MERGED_DIR="All"

imageset_textfiles="train_train.txt train_val.txt train_trainval.txt
    D00_train.txt D01_train.txt D10_train.txt
    D11_train.txt D20_train.txt D40_train.txt
    D43_train.txt D44_train.txt train.txt
    D00_val.txt D01_val.txt D10_val.txt
    D11_val.txt D20_val.txt D40_val.txt
    D43_val.txt D44_val.txt val.txt
    D00_trainval.txt D01_trainval.txt D10_trainval.txt
    D11_trainval.txt D20_trainval.txt D40_trainval.txt
    D43_trainval.txt D44_trainval.txt trainval.txt"


for dir in Annotations JPEGImages labels ImageSets/Main
do
    mkdir -p $DATASET_DIR/$MERGED_DIR/$dir
done

for filename in $imageset_textfiles
do
    cat $DATASET_DIR/*/ImageSets/Main/$filename > $DATASET_DIR/$MERGED_DIR/ImageSets/Main/$filename
done


for dir in Annotations JPEGImages labels
do
    cp $DATASET_DIR/*/$dir/* $DATASET_DIR/$MERGED_DIR/$dir/
done
