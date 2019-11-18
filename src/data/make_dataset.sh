#!/bin/bash

kaggle datasets download -d marlesson/news-of-the-site-folhauol

mv news-of-the-site-folhauol.zip ../../data/raw

unzip ../../data/raw/news-of-the-site-folhauol.zip ../../data/interim -y

