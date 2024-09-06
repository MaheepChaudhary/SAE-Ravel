pip install -r requirements.txt

echo Installing the different SAE for OpenAI

for layer_index in {0..11}; do wget "sparse-autoencoder/gpt2-small/your_location/autoencoders/${layer_index}.pt" -P ./autoencoders/; done

echo Installing the different SAE for Bloom

git lfs install
git clone https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted

