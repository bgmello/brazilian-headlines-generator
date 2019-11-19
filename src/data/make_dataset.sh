#!/bin/bash
echo "downloading news-of-the-site-folhauol from s3..."
wget https://brazilian-headlines-generator.s3.amazonaws.com/news-of-the-site-folhauol.zip

mv news-of-the-site-folhauol.zip ../../data/raw

unzip -o ../../data/raw/news-of-the-site-folhauol.zip -d ../../data/interim/news-of-the-site-folhauol

echo "download complete!"

