corpusdir=/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus/
newcorpusdir=/IUS/homes4/rohytg/work/data/002_ExtendedPAL/corpus_mods/esvm/

array=()
# Read the file in parameter and fill the array named "array"
getArray() {
    i=0
    while read line # Read a line
    do
        array[i]=$line # Put it into the array
        i=$(($i + 1))
    done < $1
}
getArray "/IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/Images.txt"

mkdir -p $newcorpusdir/trainNneg
while read line; do
  ln -s $corpusdir/${array[$(($line-1))]} $newcorpusdir/trainNneg/$line.jpg
done < /IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesTrain.txt

# fix the 10571

mkdir -p $newcorpusdir/test/
while read line; do
  ln -s $corpusdir/${array[$(($line-1))]} $newcorpusdir/test/$line.jpg
done < /IUS/homes4/rohytg/work/data/002_ExtendedPAL/lists/NdxesTest.txt


