# echo "Running eval for Bloom SAE"
# python3.11 eval_sae_main.py -d "cuda:1" -met "sae masking neel" -bs 16  

# echo "Running eval for OpenAI SAE"
# python3.11 eval_sae_main.py -d "cuda:1" -met "sae masking openai" -bs 16   


# echo "Running eval for Apollo SAE"
# python3.11 eval_sae_main.py -d "cuda:1" -met "sae masking apollo" -bs 16   

echo "Running eval for Bloom SAE"
python3.11 eval_sae_main.py -d "cuda:1" -met "acc sae masking neel" -bs 16  

echo "Running eval for OpenAI SAE"
python3.11 eval_sae_main.py -d "cuda:1" -met "acc sae masking openai" -bs 16   

echo "Running eval for Apollo SAE"
python3.11 eval_sae_main.py -d "cuda:1" -met "acc sae masking apollo" -bs 16   
