from eval_gpt2 import *
from imports import *
from models import *
from ravel_data_prep import *

random.seed(2)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def config(DEVICE):
    model = LanguageModel("openai-community/gpt2", device_map=DEVICE)
    intervened_token_idx = -8
    return model, intervened_token_idx


def create_latex_table(data, headers):
    # Begin the LaTeX table environment
    latex_code = "\\begin{table}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{|" + " | ".join(["c"] * len(headers)) + "|}\n"
    latex_code += "\\hline\n"

    # Add headers
    latex_code += " & ".join(headers) + " \\\\\n"
    latex_code += "\\hline\n"

    # Add table data
    for row in data:
        latex_code += " & ".join(map(str, row)) + " \\\\\n"
        latex_code += "\\hline\n"

    # End the LaTeX table environment
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{Your caption here}\n"
    latex_code += "\\label{table:your_label}\n"
    latex_code += "\\end{table}"

    return latex_code


def loss(sent, model, intervened_token_idx, indices):
    (
        loss0_arr,
        loss1_arr,
        loss2_arr,
        loss3_arr,
        loss4_arr,
        loss5_arr,
        loss6_arr,
        loss7_arr,
        loss8_arr,
        loss9_arr,
        loss10_arr,
        loss11_arr,
    ) = ([], [], [], [], [], [], [], [], [], [], [], [])

    with torch.no_grad():
        for i in tqdm(range(indices)):

            samples = sent["input_ids"][i * 16 : (i + 1) * 16]

            (
                loss0,
                loss1,
                loss2,
                loss3,
                loss4,
                loss5,
                loss6,
                loss7,
                loss8,
                loss9,
                loss10,
                loss11,
            ) = model(samples)

            loss0_arr.append(loss0.mean(0).item())
            loss1_arr.append(loss1.mean(0).item())
            loss2_arr.append(loss2.mean(0).item())
            loss3_arr.append(loss3.mean(0).item())
            loss4_arr.append(loss4.mean(0).item())
            loss5_arr.append(loss5.mean(0).item())
            loss6_arr.append(loss6.mean(0).item())
            loss7_arr.append(loss7.mean(0).item())
            loss8_arr.append(loss8.mean(0).item())
            loss9_arr.append(loss9.mean(0).item())
            loss10_arr.append(loss10.mean(0).item())
            loss11_arr.append(loss11.mean(0).item())

            torch.cuda.empty_cache()

        mean0 = sum(loss0_arr) / len(loss0_arr)
        mean1 = sum(loss1_arr) / len(loss1_arr)
        mean2 = sum(loss2_arr) / len(loss2_arr)
        mean3 = sum(loss3_arr) / len(loss3_arr)
        mean4 = sum(loss4_arr) / len(loss4_arr)
        mean5 = sum(loss5_arr) / len(loss5_arr)
        mean6 = sum(loss6_arr) / len(loss6_arr)
        mean7 = sum(loss7_arr) / len(loss7_arr)
        mean8 = sum(loss8_arr) / len(loss8_arr)
        mean9 = sum(loss9_arr) / len(loss9_arr)
        mean10 = sum(loss10_arr) / len(loss10_arr)
        mean11 = sum(loss11_arr) / len(loss11_arr)

        return (
            round(mean0, 2),
            round(mean1, 2),
            round(mean2, 2),
            round(mean3, 2),
            round(mean4, 2),
            round(mean5, 2),
            round(mean6, 2),
            round(mean7, 2),
            round(mean8, 2),
            round(mean9, 2),
            round(mean10, 2),
            round(mean11, 2),
        )

def accuracy(sent, label, model_, intervened_token_idx, indices, method):
    
    with torch.no_grad():
        acc_dict = {"Layer0": [0], 
                    "Layer1": [0], 
                    "Layer2": [0],
                    "Layer3": [0],
                    "Layer4": [0],
                    "Layer5": [0],
                    "Layer6": [0],
                    "Layer7": [0],
                    "Layer8": [0],
                    "Layer9": [0],
                    "Layer10": [0],
                    "Layer11": [0]}
        
        for i in tqdm(range(indices)):
            
            batch_size = 16

            samples = sent["input_ids"][i * 16 : (i + 1) * 16]
            labels = label[i * 16 : (i + 1) * 16]
            output_list = model_(samples)
            ground_truth_token_id = labels
            if method == "acc sae masking neel":

                for layer in range(12):
                    total_neel_samples_processed = 0 
                    matches_neel = 0
                    predicted_text_neel_ = output_list[f"Predicted_L{layer}"][1]
                    
                    # Calculate accuracy
                    predicted_text_neel = [word.split()[0] for word in predicted_text_neel_]
                    source_label_neel = [word.split()[0] for word in labels]
                    for i in range(len(predicted_text_neel)):
                        total_neel_samples_processed += 1
                        if predicted_text_neel[i] == source_label_neel[i]:
                            matches_neel += 1
                    value_neel = matches_neel/len(predicted_text_neel)
                    acc_dict[f"Layer{layer}"].append(value_neel)
                    
                torch.cuda.empty_cache()
            elif method == "acc sae masking openai":
                
                for layer in range(12):
                    total_openai_samples_processed = 0
                    matches_openai = 0
                    predicted_text_ = output_list[f"Predicted_L{layer}"][1]
                    
                    # Calculate accuracy
                    predicted_text = [word.split()[0] for word in predicted_text_]
                    source_label = [word.split()[0] for word in labels]

                    for i in range(len(predicted_text)):
                        total_openai_samples_processed += 1
                        if predicted_text[i] == source_label[i]:
                            matches_openai += 1
                        else:
                            print(f"Predicted: {predicted_text[i]}")
                            print(f"Ground label: {source_label[i]}")
                            print()
                    
                    
                    acc_dict[f"Layer{layer}"].append(matches_openai / total_openai_samples_processed)
                    
                torch.cuda.empty_cache()
            
            elif method == "acc sae masking apollo":
                for layer in range(6):
                    total_apollo_samples_processed = 0
                    matches_apollo = 0
                    predicted_text_ = output_list[f"Predicted_L{layer}"][1]
                    
                    # Calculate accuracy
                    predicted_text = [word.split()[0] for word in predicted_text_]
                    source_label = [word.split()[0] for word in labels]

                    for i in range(len(predicted_text)):
                        total_apollo_samples_processed += 1
                        if predicted_text[i] == source_label[i]:
                            matches_apollo += 1
                    
                    acc_dict[f"Layer{layer}"].append(matches_apollo / total_apollo_samples_processed)
                    torch.cuda.empty_cache()
                    
                acc_dict[f"Layer6"].append(0)
                acc_dict[f"Layer7"].append(0)
                acc_dict[f"Layer8"].append(0)
                acc_dict[f"Layer9"].append(0)
                acc_dict[f"Layer10"].append(0)
                acc_dict[f"Layer11"].append(0)
                
        print(acc_dict)
        acc_list = [(sum(acc_dict[f"Layer{i}"])/(len(acc_dict[f"Layer{i}"]) - 1)) for i in range(12)]
        print(acc_list)
    return acc_list
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--device", default="cuda:1")
    parser.add_argument("-met", "--method", required=True)
    parser.add_argument("-bs", "--batch_size", required=True)
    args = parser.parse_args()

    model, intervened_token_idx = config(args.device)
    batch_size = 16

    latexlist = []
    with open("comfy_continent.json", "r") as f:
        contdata = json.load(f)

    with open("comfy_country.json", "r") as f1:
        countdata = json.load(f1)

    contsent = [sent[0] for sent in contdata]
    contlabel = [" " + label[1].split()[0] for label in contdata]
    print(contlabel)

    countsent = [s[0] for s in countdata]
    countlabel = [" " + l[1].split()[0] for l in countdata]

    t_contsent = model.tokenizer(contsent, return_tensors="pt").to(args.device)
    t_contlabel = model.tokenizer(contlabel, return_tensors="pt").to(args.device)
    
    t_countsent = model.tokenizer(countsent, return_tensors="pt").to(args.device)
    t_countlabel = model.tokenizer(countlabel, return_tensors="pt").to(args.device)

    count_indices = int(len(countsent) / 16)
    print(f"Continent Indices: {count_indices}")
    
    cont_indices = int(len(contsent) / 16)
    print(f"Continent Indices: {cont_indices}")

    if args.method == "sae masking neel" or args.method == "sae masking openai" or args.method == "sae masking apollo":
        model_sae_eval = eval_sae(
        model=model,
        DEVICE=args.device,
        method=args.method,
        intervened_token_idx=intervened_token_idx,
        batch_size=args.batch_size
        )
        
        
        (
            loss0,
            loss1,
            loss2,
            loss3,
            loss4,
            loss5,
            loss6,
            loss7,
            loss8,
            loss9,
            loss10,
            loss11,
        ) = loss(
            sent=t_contsent,
            model=model_sae_eval,
            intervened_token_idx=-8,
            indices=cont_indices,
        )

        t_countsent = model.tokenizer(countsent, return_tensors="pt").to(args.device)
        t_countlabel = model.tokenizer(countlabel, return_tensors="pt").to(args.device)

        eval_acc = eval_sae_acc(
            model=model,
            DEVICE=args.device,
            method=args.method,
            intervened_token_idx=intervened_token_idx,
            batch_size=args.batch_size,
        )

        if args.method == "sae masking neel":
            latexbloomlist_country = [
                loss0,
                loss1,
                loss2,
                loss3,
                loss4,
                loss5,
                loss6,
                loss7,
                loss8,
                loss9,
                loss10,
                loss11,
            ]
        elif args.method == "sae masking openai":
            latexopenailist_country = [
                loss0,
                loss1,
                loss2,
                loss3,
                loss4,
                loss5,
                loss6,
                loss7,
                loss8,
                loss9,
                loss10,
                loss11,
            ]
        elif args.method == "sae masking apollo":
            latexapollolist_country = [
                loss0,
                loss1,
                loss2,
                loss3,
                loss4,
                loss5,
                loss6,
                loss7,
                loss8,
                loss9,
                loss10,
                loss11,
            ]

        t_countsent = model.tokenizer(countsent, return_tensors="pt").to(args.device)

        count_indices = int(len(t_countsent["input_ids"]) / 16)
        print(f"Country Indices: {count_indices}")

        (
            count_loss0,
            count_loss1,
            count_loss2,
            count_loss3,
            count_loss4,
            count_loss5,
            count_loss6,
            count_loss7,
            count_loss8,
            count_loss9,
            count_loss10,
            count_loss11,
        ) = loss(
            sent=t_countsent,
            model=model_sae_eval,
            intervened_token_idx=-8,
            indices=count_indices,
        )

        if args.method == "sae masking neel":
            latexbloomlist_continent = [
                count_loss0,
                count_loss1,
                count_loss2,
                count_loss3,
                count_loss4,
                count_loss5,
                count_loss6,
                count_loss7,
                count_loss8,
                count_loss9,
                count_loss10,
                count_loss11,
            ]
            with open("latex_table.txt", "w") as f:
                for item in latexbloomlist_country:
                    f.write(f"{item}\n")

            with open("latex_table.txt", "a") as f:
                for item in latexbloomlist_continent:
                    f.write(f"{item}\n")

        elif args.method == "sae masking openai":
            latexopenailist_continent = [
                count_loss0,
                count_loss1,
                count_loss2,
                count_loss3,
                count_loss4,
                count_loss5,
                count_loss6,
                count_loss7,
                count_loss8,
                count_loss9,
                count_loss10,
                count_loss11,
            ]
            with open("latex_table.txt", "a") as f:
                for item in latexopenailist_country:
                    f.write(f"{item}\n")

            with open("latex_table.txt", "a") as f:
                for item in latexopenailist_continent:
                    f.write(f"{item}\n")

        elif args.method == "sae masking apollo":
            latexapollolist_continent = [
                count_loss0,
                count_loss1,
                count_loss2,
                count_loss3,
                count_loss4,
                count_loss5,
                count_loss6,
                count_loss7,
                count_loss8,
                count_loss9,
                count_loss10,
                count_loss11,
            ]
            with open("latex_table.txt", "a") as f:
                for item in latexapollolist_country:
                    f.write(f"{item}\n")

            with open("latex_table.txt", "a") as f:
                for item in latexapollolist_continent:
                    f.write(f"{item}\n")

    elif args.method == "acc sae masking neel":
        model_sae_acc = eval_sae_acc(model=model,
                                    DEVICE=args.device,
                                    method=args.method,
                                    intervened_token_idx=intervened_token_idx,
                                    batch_size=args.batch_size)
        acc_list_count_neel = accuracy(
            sent=t_countsent,
            label = countlabel,
            model_=model_sae_acc,
            intervened_token_idx=-8,
            indices=count_indices,
            method=args.method,
        )
        acc_list_cont_neel= accuracy(
            sent=t_contsent,
            label = contlabel,
            model_=model_sae_acc,
            intervened_token_idx=-8,
            indices=cont_indices,
            method=args.method,
        )
        
        assert len(acc_list_count_neel) == len(acc_list_cont_neel) == 12
        
        with open("latex_table_acc.txt", "w") as f:
            for _item in acc_list_count_neel:
                f.write(f"{_item}\n")

        with open("latex_table_acc.txt", "a") as f:
            for item___ in acc_list_cont_neel:
                f.write(f"{item___}\n")
        

    elif args.method == "acc sae masking openai":
        model_sae_acc = eval_sae_acc(model, 
                            args.device, 
                            args.method, 
                            intervened_token_idx, 
                            batch_size)
        acc_list_count_openai = accuracy(
            sent=t_countsent,
            label = countlabel,
            model_=model_sae_acc,
            intervened_token_idx=-8,
            indices=count_indices,
            method=args.method,
        )
        acc_list_cont_openai = accuracy(
            sent=t_contsent,
            label = contlabel,
            model_=model_sae_acc,
            intervened_token_idx=-8,
            indices=cont_indices,
            method=args.method,
        )

        assert len(acc_list_count_openai) == len(acc_list_cont_openai) == 12

        with open("latex_table_acc.txt", "a") as f:
            for item__ in acc_list_count_openai:
                f.write(f"{item__}\n")

        with open("latex_table_acc.txt", "a") as f:
            for item_ in acc_list_cont_openai:
                f.write(f"{item_}\n")
        

    elif args.method == "acc sae masking apollo":
        model_sae_acc = eval_sae_acc(model, 
                                    args.device, 
                                    args.method, 
                                    intervened_token_idx, 
                                    batch_size)
        acc_list_count_apollo = accuracy(
            sent=t_countsent,
            label = countlabel,
            model_=model_sae_acc,
            intervened_token_idx=-8,
            indices=count_indices,
            method=args.method,
        )
        acc_list_cont_apollo = accuracy(
            sent=t_contsent,
            label = contlabel,
            model_=model_sae_acc,
            intervened_token_idx=-8,
            indices=cont_indices,
            method=args.method,
        )
        
        assert len(acc_list_count_apollo) == len(acc_list_cont_apollo) == 12
        
        with open("latex_table_acc.txt", "a") as f:
            for itemacc_list_count in acc_list_count_apollo:
                f.write(f"{itemacc_list_count}\n")

        with open("latex_table_acc.txt", "a") as f:
            for itemacc_list_cont in acc_list_cont_apollo:
                f.write(f"{itemacc_list_cont}\n")
    
