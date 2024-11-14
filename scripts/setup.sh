sudo apt remove python3.10 python3.12 python3.9 python3.x
sudo apt autoremove

sudo apt install python3.11 python3.11-venv python3.11-dev

pip install -r requirements.txt
pip install sae-lens

echo Installing the different SAE for OpenAI

mkdir openai_saes/downloaded_saes
for layer_index in {0..11}; do wget "https://openaipublic.blob.core.windows.net/sparse-autoencoder/gpt2-small/resid_post_mlp_v5_32k/autoencoders/${layer_index}.pt" -P ./openai_sae/downloaded_saes/; done

echo Installing the different SAE for Bloom

git lfs install
git clone https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted

