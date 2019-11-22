#!/bin/bash
echo "downloading glove word2vec from s3..."
wget https://brazilian-headlines-generator.s3.amazonaws.com/glove_s300.zip

mv glove_s300.zip ../../data/external

unzip -o ../../data/external/glove_s300.zip -d ../../data/interim/

echo "download complete!"
