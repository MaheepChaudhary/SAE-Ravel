#!/bin/bash


echo "running neuron masking script"
python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5 -d "mps"
python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5 -d "mps"

#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#
#echo "Running sae masking neel's script"
#python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
#
# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-5" -lr "0.001" -lid 5


# echo "Running sae masking apollo script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking apollo" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-5" -lr "0.001" -lid 5
# python3.11 main_train.py -a continent -tla 61 -method "sae masking apollo" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-5" -lr "0.001" -lid 5


#echo "running neuron masking script"
#python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
#python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
#
#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
#
#echo "Running sae masking neel's script"
#python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
#python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
#
# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-6" -lr "0.001" -lid 6
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-6" -lr "0.001" -lid 6
#
#
#
#echo "running neuron masking script"
#python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
#python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
#
#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
#
#echo "Running sae masking neel's script"
#python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
#python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
#
# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-7" -lr "0.001" -lid 7
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-7" -lr "0.001" -lid 7



#echo "running neuron masking script"
#python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8
#python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8
#
#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8
#
#echo "Running sae masking neel's script"
#python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8
#python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8

# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-8" -lr "0.001" -lid 8
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-8" -lr "0.001" -lid 8
#
#
#echo "running neuron masking script"
#python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
#python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
#
#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
#
#echo "Running sae masking neel's script"
#python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
#python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
#
# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-9" -lr "0.001" -lid 9
#
#
#echo "running neuron masking script"
#python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10
#python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10
#
#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10 
#
#echo "Running sae masking neel's script"
#python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10
#python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10
#
# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-10" -lr "0.001" -lid 10
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-10" -lr "0.001" -lid 10
#
#
#echo "running neuron masking script"
#python3.11 main_train.py -a country -tla 61 -method "neuron masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
#python3.11 main_train.py -a continent -tla 61 -method "neuron masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
#
#
#echo "Running das masking script"
#python3.11 main_train.py -a country -tla 61 -method "das masking" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
#python3.11 main_train.py -a continent -tla 61 -method "das masking" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
#
# echo "Running sae masking neel's script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking neel" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
# python3.11 main_train.py -a continent -tla 61 -method "sae masking neel" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11

# echo "Running sae masking openai script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking openai" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-11" -lr "0.001" -lid 11
# python3.11 main_train.py -a continent -tla 61 -method "sae masking openai" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-11" -lr "0.001" -lid 11
#
#
# echo "Running sae masking apollo script"
# python3.11 main_train.py -a country -tla 61 -method "sae masking apollo" -e 20 -t train -idd "country" -bs 16 -wb "True" -n "t(10-0.1)_lr0.001_layer-9" -lr "0.001" -lid 9
# python3.11 main_train.py -a continent -tla 61 -method "sae masking apollo" -e 20 -t train -idd "continent" -bs 16 -wb "True" -n "t(10-0.1)_lr00.01_layer-9" -lr "0.001" -lid 9


