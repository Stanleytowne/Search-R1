for category in Mapping Media Movies Search; do
    echo "Starting SFT for $category"
    bash scripts/sft.sh $category
    echo "SFT for $category completed"
done