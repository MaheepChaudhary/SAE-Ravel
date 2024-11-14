import os
import sys
import json
sys.path.append("../old_src")
sys.path.append("../utils")
sys.path.append("../models")
sys.path.append("../configs")

from utils.utils import *  


def main():

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/das_dl.json")
    args = parser.parse_args()
    
    with open(args.config_path, "r") as file:
        config = json.load(file)

    model_name = config["model"]  # Other options include "gpt2-medium", "gpt2-large", "gpt2-xl"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model = model.to(config["device"])
    rotate_layer = RotateLayer(config["rotate_layer_size"])
    # So as the orthogonal layer does not change even during training. 
    rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
    
    data_obj = data(config)
    continent_data, country_data = data_obj.load_data()
    print(continent_data) 
    #TODO: Find the accuracy of the whole data on continent and country
    

if __name__ == "__main__":    
    main()    

    


