for category in SMS Science Text_Analysis Video_Images Weather; do
    echo "Starting SFT for $category"
    bash scripts/sft.sh $category
    echo "SFT for $category completed"
done