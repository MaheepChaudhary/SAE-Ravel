pip install -r requirements.txt

echo Installing the different SAE for OpenAI

mkdir openai_saes/downloaded_saes
for layer_index in {0..11}; do wget "https://openaipublic.blob.core.windows.net/sparse-autoencoder/gpt2-small/resid_delta_mlp/autoencoders/${layer_index}.pt" -P ./openai_sae/downloaded_saes/; done
echo Installing the different SAE for Bloom

git lfs install
git clone https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted

