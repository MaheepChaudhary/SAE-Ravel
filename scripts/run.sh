#!/bin/bash


echo "running neuron masking script"
python3.11 src/main.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5 -d "mps"
# python3.11 src/main.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5 -d "mps"

#
#echo "Running das masking script"
#python3.11 main.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#python3.11 main.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#
#echo "Running sae masking neel's script"
#python3.11 main.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#python3.11 main.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#
# echo "Running sae masking openai script"
# python3.11 main.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
# python3.11 main.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-5" -lr "0.001" -lid 5


# echo "Running sae masking apollo script"
# python3.11 main.py -a country -tla 61 -method "sae masking apollo" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
# python3.11 main.py -a continent -tla 61 -method "sae masking apollo" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-5" -lr "0.001" -lid 5

