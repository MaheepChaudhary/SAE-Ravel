'''
Country Prompt: 
Toronto is in Canada. {E} is in, 

Continent:
{E} is a city in the continent of, 

Language:
[{"city": "Beijing", "lang": "Chinese"}, {"city": "{E}", "lang": " "}]

'''


from imports import *


def country_prompt(data, cities):

    country_data = []

    for city in cities:
        label = data[city]["Country"]
        # country_data.append([f"Toronto is in Canada. {city} is in", label])
        country_data.append([f"Toronto is a city in the country of Canada. Beijing is a city in the country of China. Miami is a city in the country of United States. Santiago is a city in the country of Chile. London is a city in the country of England. {city} is a city in the country of", label])
    return country_data


def continent_prompt(data, cities):

    continent_data = []

    for city in cities:
        label = data[city]["Continent"]
        # continent_data.append([f"{city} is a city in the continent of", label])
        continent_data.append([f"Toronto is a city in the continent of North America. Beijing is a city in the continent of Asia. Miami is a city in the continent of North America. Santiago is a city in the continent of South America. London is a city in the continent of Europe. {city} is a city in the continent of", label])

    return continent_data

def language_prompt(data, cities):

    language_data = []

    for city in cities:

        label = data[city]["Language"]
        language_data.append([{"city": "Beijing", "lang": "Chinese"}, {"city": f"{city}", "lang": " "}, label])

    return language_data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-p", "--path_json", default = "ravel/data/ravel_city_entity_attributes.json", help='Prompting for Ravel Data')
    
    args = parser.parse_args()
    
    with open(args.path_json, 'r') as file:
        data = json.load(file)

    cities = list(data.keys())

    country_data = country_prompt(data, cities)
    continent_data = continent_prompt(data, cities)
    language_data = language_prompt(data, cities)

    with open('processed_data/country_data.json', 'w') as file:
        json.dump(country_data, file)
    
    with open('processed_data/continent_data.json', 'w') as file:
        json.dump(continent_data, file)
    
    with open('processed_data/language_data.json', 'w') as file:
        json.dump(language_data, file)
