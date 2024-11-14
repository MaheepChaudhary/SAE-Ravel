import copy
from imports import *

def analyse(model, data, attribute):
    new_data = []
    for sample_no in tqdm(range(len(data))):
        sample = data[sample_no]
        if len(model.tokenizer.tokenize(sample[0])) == 61 and attribute == "continent":
            with model.trace(sample[0]) as tracer:
                output = model.lm_head.output[0].save()
        
            prediction = model.tokenizer.decode(output.argmax(dim = -1)[-1]).split()[0]
            city = sample[0].split()[-8]
            if prediction == sample[1].split()[0]:
                new_data.append([sample[0], sample[1]])

        if len(model.tokenizer.tokenize(sample[0])) == 59 and attribute == "country":

            with model.trace(sample[0]) as tracer:
                output = model.lm_head.output[0].save()
        
            prediction = model.tokenizer.decode(output.argmax(dim = -1)[-1]).split()[0]
            city = sample[0].split()[-8]
            if prediction == sample[1].split()[0]:
                new_data.append([sample[0], sample[1]])

        else:
            continue
    
    with open(f"comfy_{attribute}.json", "w") as f:
        json.dump(new_data, f)




def overlap_measure(country_data, continent_data):

    list_of_country_cities = []
    list_of_continent_cities = []

    for sample in country_data:
        list_of_country_cities.append(sample[0].split()[-8])

    for sample in continent_data:
        list_of_continent_cities.append(sample[0].split()[-8])

    overlapping_cities = set(list_of_country_cities) & set(list_of_continent_cities)

    print(f"The total number of overlapping cities are {len(overlapping_cities)}")

    return overlapping_cities


def final_data(country_data, continent_data):

    overlapping_cities = overlap_measure(country_data, continent_data)

    new_country_data = []
    new_continent_data = []

    copied_country_data = copy.deepcopy(country_data)
    copied_continent_data = copy.deepcopy(continent_data)

    for data in country_data:
        city = data[0].split()[-8]
        if city in overlapping_cities:
            for data1 in copied_country_data:
                city1 = data1[0].split()[-8]
                if city1 in overlapping_cities:
                    new_country_data.append([[data[0], data[1]], [data1[0], data1[1]]])
                else:
                    continue
        else:
            continue


    for data in continent_data:
        city = data[0].split()[-8]
        if city in overlapping_cities:
            for data1 in copied_continent_data:
                city1 = data1[0].split()[-8]
                if city1 in overlapping_cities:
                    new_continent_data.append([[data[0], data[1]], [data1[0], data1[1]]])
                else:
                    continue
        else:
            continue

    with open("final_data_country.json","w") as f1:
        json.dump(new_country_data, f1)

    with open("final_data_continent.json", "w") as f2:
        json.dump(new_continent_data, f2)

if __name__ == "__main__":  
    model = LanguageModel("openai-community/gpt2", device_map="cuda:1")
    print(model)
    
    with open("data/continent_data.json", "r") as f:
        continent_data = json.load(f)


    with open("data/country_data.json", "r") as f:
        country_data = json.load(f)


    analyse(model, continent_data,"continent")
    analyse(model, country_data, "country")



    with open("data/comfy_continent.json","r") as f:
        continent_data = json.load(f)

    with open("data/comfy_country.json", "r") as f:
        country_data = json.load(f)

    
    final_data(continent_data = continent_data, country_data = country_data)
