#!/bin/bash
mkdir -p data
pushd data
#curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
#unzip wikitext-103-v1.zip
curl -O https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip
popd
