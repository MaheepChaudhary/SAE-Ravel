from matplotlib import pyplot as plt

import matplotlib as mpl
# making the graph for continent


# def filter_non_zero_values(layer_list):
#     """
#     Filter out zero values from the layer list and return the filtered list
#     along with the indices of the non-zero values.
#     """
#     filtered_values = [value for value in layer_list if value != 0]
#     indices = [index for index, value in enumerate(layer_list) if value != 0]


neuron_masking_continent = [
    0.72,
    0.72,
    0.73,
    0.71,
    0.73,
    0.72,
    0.72,
    0.74,
    0.75,
    0.72,
    0.69,
    0.68,
]
das_masking_continent = [
    0.94,
    0.96,
    0.92,
    0.95,
    0.97,
    0.96,
    0.96,
    0.95,
    0.89,
    0.81,
    0.77,
    0.68,
]
sae_neel_masking_continent = [
    0.63,
    0.45,
    0.4,
    0.53,
    0.6,
    0.62,
    0.6,
    0.64,
    0.67,
    0.52,
    0.21,
    0.68,
]
sae_openai_masking_continent = [
    0.72,
    0.73,
    0.7,
    0.67,
    0.69,
    0.70,
    0.64,
    0.61,
    0.66,
    0.68,
    0.67,
    0.68,
]
sae_apollo_e2eds_masking_continent = [
    0.57,
    0.39,
    0.64,
]
sae_apollo_masking_continent = [
    0.53,
    0.39,
    0.72,
]

baseline = [0.565625] * len(sae_openai_masking_continent)

layer = list(range(12))
plt.plot(layer, das_masking_continent, label="DAS", marker="o", color = "green")
plt.plot(layer, neuron_masking_continent, label="Neuron", marker="o", color = "red")
plt.plot(layer, sae_openai_masking_continent, label="OpenAI SAE", marker="o", color = "gray")
plt.plot(layer, baseline, linestyle='--', label='Baseline', color = "black")
plt.plot(layer, sae_neel_masking_continent, label="Bloom SAE", marker="o", color = "blue")
layer_for_apollo = [1, 5, 9]
plt.plot(
    layer_for_apollo,
    sae_apollo_e2eds_masking_continent,
    label="Apollo e2eds SAE",
    marker="o",
    color = "violet"
)
plt.plot(layer_for_apollo, sae_apollo_masking_continent, label="Apollo SAE", marker="o", color = "pink")

plt.title("Disentangle Score for country-intervened")
plt.xlabel("Layer")
plt.ylabel("Disentangle Score")
plt.legend()
plt.savefig("continent.png")
plt.close()


neuron_masking_country = [
    0.74,
    0.71,
    0.74,
    0.73,
    0.73,
    0.73,
    0.71,
    0.73,
    0.54,
    0.55,
    0.58,
    0.56,
]
das_masking_country = [
    0.93,
    0.94,
    0.91,
    0.94,
    0.95,
    0.94,
    0.95,
    0.94,
    0.74,
    0.70,
    0.68,
    0.56,
]
sae_neel_masking_country = [
    0.64,
    0.42,
    0.38,
    0.57,
    0.65,
    0.64,
    0.63,
    0.65,
    0.60,
    0.49,
    0.22,
    0.56,
]
sae_openai_masking_country = [
    0.71,
    0.73,
    0.71,
    0.67,
    0.71,
    0.7,
    0.63,
    0.59,
    0.42,
    0.51,
    0.55,
    0.56,
]
sae_apollo_e2eds_masking_country = [
    0.59,
    0.44,
    0.6,
]
sae_apollo_masking_country = [
    0.54,
    0.44,
    0.57,
]

baseline = [0.565625] * len(sae_openai_masking_country)

layer = list(range(12))
plt.plot(layer, das_masking_country, label="DAS", marker="o", color = "green")
plt.plot(layer, neuron_masking_country, label="Neuron", marker="o", color = "red")
plt.plot(layer, sae_openai_masking_country, label="OpenAI SAE", marker="o", color = "gray")
plt.plot(layer, baseline, linestyle='--', label='Baseline', color = "black")
plt.plot(layer, sae_neel_masking_country, label="Bloom SAE", marker="o", color = "blue")
layer_for_apollo = [1, 5, 9]
plt.plot(
    layer_for_apollo,
    sae_apollo_e2eds_masking_country,
    label="Apollo e2eds SAE",
    marker="o",
    color = "violet"
)
plt.plot(layer_for_apollo, sae_apollo_masking_country, label="Apollo SAE", marker="o", color = "pink")

plt.title("Disentangle Score for country-intervened")
plt.xlabel("Layer")
plt.ylabel("Disentangle Score")
mpl.rcParams["legend.loc"] = 4
plt.legend()
plt.savefig("country.png")
