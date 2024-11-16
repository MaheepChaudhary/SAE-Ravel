from imports import *
from models import *
from data.ravel_data_prep import *

random.seed(2)


def config(learning_rate, token_length):

    model = LanguageModel("openai-community/gpt2", device_map=DEVICE)

    intervened_token_idx = -8
    intervention_token_length = token_length

    return model, intervened_token_idx


def data_processing(
    model, samples, token_length_allowed, attribute, DEVICE, batch_size
):

    # print(np.array(samples).shape)
    bases = list(np.array(samples)[:, 0, 0])
    sources = list(np.array(samples)[:, 1, 0])
    base_labels = list(np.array(samples)[:, 0, 1])
    source_labels = list(np.array(samples)[:, 1, 1])
    assert (
        len(bases)
        == len(sources)
        == len(base_labels)
        == len(source_labels)
        == batch_size
    )

    base_ids = model.tokenizer(bases, return_tensors="pt").to(DEVICE)
    source_ids = model.tokenizer(sources, return_tensors="pt").to(DEVICE)
    # print(base_ids["input_ids"].shape, source_ids["input_ids"].shape)
    source_tokens = model.tokenizer(sources)
    base_tokens = model.tokenizer(bases)
    source_label_token = model.tokenizer(source_labels)
    base_label_token = model.tokenizer(base_labels)

    # The model has the vocab with words with space along side them, so we are making the tokens s.t. they do not split and correspond to their word with integrated space.
    source_label_mods = [" " + label.split()[0] for label in source_labels]
    base_label_mods = [" " + label.split()[0] for label in base_labels]

    base_label_ids = model.tokenizer(base_label_mods, return_tensors="pt").to(DEVICE)
    source_label_ids = model.tokenizer(source_label_mods, return_tensors="pt").to(
        DEVICE
    )

    allowed_token_length = 59 if attribute == "country" else 61

    proceed = True
    return (
        proceed,
        base_ids,
        source_ids,
        base_label_ids,
        source_label_ids,
        source_labels,
        base_labels,
    )


def train_data_processing(task, intervention_divided_data, batch_size):

    with open("data/final_data_continent.json", "r") as file:
        continent_data = json.load(file)

    with open("data/final_data_country.json", "r") as file:
        country_data = json.load(file)

    random.shuffle(country_data)
    random.shuffle(continent_data)

    if task == "train" or task == "test":
        if intervention_divided_data == "continent":
            data1 = country_data
            data2 = continent_data
            print("Inside continent intevention")
        if intervention_divided_data == "country":
            data1 = continent_data
            data2 = country_data

        new_data1 = [[[None, None], [None, None]] for _ in range(len(data1))]

        for sample_no in range(len(data1)):
            sample = data1[sample_no]
            new_data1[sample_no][0][0] = sample[0][0]
            new_data1[sample_no][0][1] = sample[0][1]
            new_data1[sample_no][1][0] = sample[1][0]
            new_data1[sample_no][1][1] = sample[0][1]

        c = 0
        all_data_ = data2 +new_data1
        for data in all_data_:
            assert np.array(data).shape == (2, 2)
            intervention_label = data[1][1]
            if intervention_label == data[0][1]:
                c+=1
        print(f"Baseline Accuracy: {c/len(all_data_)}")

        data1_num_batches = np.array(new_data1).shape[0] // batch_size
        data2_num_batches = np.array(data2).shape[0] // batch_size
        data1_batch_data = [
            new_data1[i * batch_size : (i + 1) * batch_size]
            for i in range(data1_num_batches)
        ]
        data2_batch_data = [
            data2[i * batch_size : (i + 1) * batch_size]
            for i in range(data2_num_batches)
        ]
        assert np.array(data1_batch_data).shape == (data1_num_batches, batch_size, 2, 2)
        assert np.array(data2_batch_data).shape == (data2_num_batches, batch_size, 2, 2)
        #        data = data1_batch_data + data2_batch_data

        if intervention_divided_data == "continent":
            country_batch_data = data1_batch_data
            continent_batch_data = data2_batch_data
        elif intervention_divided_data == "country":
            continent_batch_data = data1_batch_data
            country_batch_data = data2_batch_data

        random.shuffle(country_batch_data)
        random.shuffle(continent_batch_data)

        train_country_data = country_batch_data[0 : int(0.7 * len(country_batch_data))]
        train_continent_data = continent_batch_data[
            0 : int(0.7 * len(continent_batch_data))
        ]

        val_country_data = country_batch_data[
            int(0.7 * len(country_batch_data)) : int(0.8 * len(country_batch_data))
        ]
        val_continent_data = continent_batch_data[
            int(0.7 * len(continent_batch_data)) : int(0.8 * len(continent_batch_data))
        ]

        test_country_data = country_batch_data[
            int(0.8 * len(country_batch_data)) : len(country_batch_data)
        ]
        test_continent_data = continent_batch_data[
            int(0.8 * len(continent_batch_data)) : len(continent_batch_data)
        ]

    elif task == "total_iia_train":

        country_num_batches = np.array(country_data).shape[0] // batch_size
        continent_num_batches = np.array(continent_data).shape[0] // batch_size
        country_batch_data = [
            country_data[i * batch_size : (i + 1) * batch_size]
            for i in range(country_num_batches)
        ]
        continent_batch_data = [
            continent_data[i * batch_size : (i + 1) * batch_size]
            for i in range(continent_num_batches)
        ]

        assert np.array(country_batch_data).shape == (
            country_num_batches,
            batch_size,
            2,
            2,
        )
        assert np.array(continent_batch_data).shape == (
            continent_num_batches,
            batch_size,
            2,
            2,
        )

    #     data = country_batch_data + continent_batch_data
    #
    # random.shuffle(data)
    #
    # train_data = data[: int(0.7 * len(data))]
    # val_data = data[int(0.7 * len(data)) : int(0.8 * len(data))]
    # test_data = data[int(0.8 * len(data)) :]
    train_data = train_country_data + train_continent_data
    val_data = val_country_data + val_continent_data
    test_data = test_country_data + test_continent_data

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return (
        train_country_data,
        train_continent_data,
        train_data,
        val_country_data,
        val_continent_data,
        val_data,
        test_country_data,
        test_continent_data,
        test_data,
    )


def train(
    train_continent_data,
    train_country_data,
    val_country_data,
    val_continent_data,
    training_model,
    model,
    train_data,
    optimizer,
    loss_fn,
    epochs,
    token_length_allowed,
    attribute,
    temperature_schedule,
    batch_size,
    DEVICE,
    wndb,
):
    training_model.train()

    def sanity_data_check(train_data, train_continent_data, train_country_data):

        for element_no in tqdm(range(len(train_data))):
            element = train_data[element_no]
            if (
                element not in train_continent_data
                and element not in train_country_data
            ):
                print(f"Nope")
        print("Finsihed")

    sanity_data_check(train_data, train_continent_data, train_country_data)

    temp_idx = 0

    for epoch in tqdm(range(epochs)):

        correct = {i: [] for i in range(0, 12)}
        total_samples_processed = 0
        total_loss = 0.0

        # for sample_no in tqdm(range(len(data))):
        i = 0
        matches = 0
        for sample_no in range(np.array(train_data).shape[0]):

            samples = train_data[sample_no]
            assert np.array(samples).shape == (batch_size, 2, 2)
            # samples = train_data[i*batch_size:(i+1)*batch_size]

            # Data Processing
            (
                proceed,
                base_ids,
                source_ids,
                base_label_ids,
                source_label_ids,
                source_label,
                base_label,
            ) = data_processing(
                model=model,
                samples=samples,
                token_length_allowed=token_length_allowed,
                attribute=attribute,
                DEVICE=DEVICE,
                batch_size=batch_size,
            )

            if not proceed:
                continue

            # training the model
            optimizer.zero_grad()

            temperature = temperature_schedule[temp_idx]

            intervened_base_output, predicted_text = training_model(
                source_ids, base_ids, temperature
            )
            ground_truth_token_id = source_label_ids
            # ground_truth_token_id = base_label_ids
            vocab_size = model.tokenizer.vocab_size
            ground_truth_one_hot = F.one_hot(
                ground_truth_token_id["input_ids"], num_classes=vocab_size
            )
            ground_truth_one_hot = ground_truth_one_hot.to(dtype=torch.long)
            last_token_output = intervened_base_output[:, -1, :]
            assert ground_truth_one_hot.squeeze(1).shape == last_token_output.shape
            ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
            ground_truth_indices = ground_truth_indices.to(dtype=torch.long)
            loss = loss_fn(last_token_output, ground_truth_indices)
            # loss = loss_fn(predicted_logit.view(-1, predicted_logit.size(-1)), ground_truth_token_id.view(-1))
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted_text = [word.split()[0] for word in predicted_text]
            source_label = [word.split()[0] for word in source_label]
            for i in range(len(predicted_text)):
                if predicted_text[i] == source_label[i]:
                    matches += 1
            total_samples_processed += batch_size
            temp_idx += 1
            i += 1
            # if sample_no % \100 == 0 and sample_no != 0:
            # print(f"Epoch: {epoch}, Sample: {sample_no}, Accuracy: {matches / total_samples_processed:.4f}, Loss: {total_loss / total_samples_processed:.4f}")
        if wndb == "True":
            wandb.log(
                {
                    f"GPT-2 Token Sub-Space Intervention Accuracy {args.layer_intervened}": matches
                    / total_samples_processed,
                    f"GPT-2 Token Sub-Space Intervention Loss {args.layer_intervened}": total_loss
                    / total_samples_processed,
                }
            )
        print(
            f"Epoch: {epoch}, Accuracy: {matches / total_samples_processed:.4f}, Loss: {total_loss / total_samples_processed:.4f}"
        )
        torch.save(
            training_model.state_dict(),
            f"saved_models/saved_model_{args.intervention_divided_data}_{args.method}_{args.model}_e{epoch}_lr{args.learning_rate}_layer{args.layer_intervened}.pth",
        )

        # eval_model = my_model(
        #     model=model,
        #     DEVICE=DEVICE,
        #     method=args.method,
        #     token_length_allowed=args.token_length_allowed,
        #     expansion_factor=args.expansion_factor,
        #     layer_intervened=layer_intervened,
        #     intervened_token_idx=intervened_token_idx,
        #     batch_size=args.batch_size,
        # )
        #
        # # Load the state_dict from the saved file
        # eval_model.load_state_dict(
        #     torch.load(
        #         f"models/saved_model_{args.intervention_divided_data}_{args.method}_{args.model}_e{epoch}_lr{args.learning_rate}_layer{args.layer_intervened}.pth"
        #     ), strict = False
        # )
        #
        # val(
        #     eval_model=eval_model,
        #     val_country_data=val_country_data,
        #     val_continent_data=val_continent_data,
        #     model=model,
        #     val_data=val_data,
        #     loss_fn=loss_fn,
        #     batch_size=batch_size,
        #     token_length_allowed=token_length_allowed,
        #     attribute=attribute,
        #     val_temperature=temperature,
        #     DEVICE=DEVICE,
        #     wndb=wndb,
        # )
        # continent_acc = calculate_accuracy(
        #     eval_model,
        #     model,
        #     train_continent_data,
        #     token_length_allowed,
        #     attribute,
        #     batch_size,
        #     DEVICE,
        #     temperature,
        # )
        # country_acc = calculate_accuracy(
        #     eval_model,
        #     model,
        #     train_country_data,
        #     token_length_allowed,
        #     attribute,
        #     batch_size,
        #     DEVICE,
        #     temperature,
        # )
        # all_data_acc = calculate_accuracy(
        #     eval_model,
        #     model,
        #     train_data,
        #     token_length_allowed,
        #     attribute,
        #     batch_size,
        #     DEVICE,
        #     temperature,
        # )
        # print(
        #     f"Train Continent Accuracy: {continent_acc}, Train Country Accuracy: {country_acc}"
        # )
        # if wndb == "True":
        #     wandb.log(
        #         {
        #             f"Train Continent Accuracy {args.layer_intervened}": continent_acc,
        #             f"Train Country Accuracy {args.layer_intervened}": country_acc,
        #             # f"Train Accuracy {args.layer_intervened}": all_data_acc,
        #         }
        #     )
        # Log accuracy and loss to wandb
        # epoch_accuracy = matches / total_samples_processed

        # print(f"Epoch {epoch} finished with accuracy {epoch_accuracy:.4f} and average loss {total_loss / total_samples_processed:.4f}")


def calculate_accuracy(
    eval_model,
    model,
    cal_data,
    token_length_allowed,
    attribute,
    batch_size,
    DEVICE,
    cal_temperature,
):
    eval_model.eval()
    correct_predictions = 0
    total_predictions = 0
    cal_total_samples_processed = 0
    cal_matches = 0
    for cal_sample_no in range(np.array(cal_data).shape[0]):
        cal_samples = cal_data[cal_sample_no]
        assert np.array(cal_samples).shape == (batch_size, 2, 2)
        # samples = train_data[i*batch_size:(i+1)*batch_size]
        # Data Processing
        (
            proceed,
            cal_base_ids,
            cal_source_ids,
            base_label_ids,
            cal_source_label_ids,
            cal_source_label_,
            base_label,
        ) = data_processing(
            model=model,
            samples=cal_samples,
            token_length_allowed=token_length_allowed,
            attribute=attribute,
            DEVICE=DEVICE,
            batch_size=batch_size,
        )

        if not proceed:
            continue
        cal_intervened_base_output, cal_predicted_text_ = eval_model(
            cal_source_ids, cal_base_ids, cal_temperature
        )
        cal_ground_truth_token_id = cal_source_label_ids
        # ground_truth_token_id = base_label_ids
        cal_ground_truth_one_hot = F.one_hot(
            cal_ground_truth_token_id["input_ids"],
            num_classes=model.tokenizer.vocab_size,
        )
        # print(ground_truth_one_hot.shape)
        ground_truth_one_hot = cal_ground_truth_one_hot.to(dtype=torch.long)
        cal_last_token_output = cal_intervened_base_output[:, -1, :]
        assert ground_truth_one_hot.squeeze(1).shape == cal_last_token_output.shape
        cal_ground_truth_indices = torch.argmax(ground_truth_one_hot.squeeze(1), dim=1)
        cal_ground_truth_indices = cal_ground_truth_indices.to(dtype=torch.long)
        loss = loss_fn(cal_last_token_output, cal_ground_truth_indices)

        # Calculate accuracyk
        cal_predicted_text = [word.split()[0] for word in cal_predicted_text_]
        # source_label = [word.split()[0] for word in base_label]
        cal_source_label = [word.split()[0] for word in cal_source_label_]
        for i in range(len(cal_predicted_text)):
            if cal_predicted_text[i] == cal_source_label[i]:
                cal_matches += 1
        cal_total_samples_processed += batch_size
    return cal_matches / cal_total_samples_processed


def val(
    eval_model,
    model,
    val_data,
    val_continent_data,
    val_country_data,
    loss_fn,
    batch_size,
    token_length_allowed,
    attribute,
    val_temperature,
    DEVICE,
    wndb,
):
    eval_model.eval()
    matches_val = 0
    total_val_samples_processed = 0
    total_val_loss = 0

    correct_val = {i: [] for i in range(0, 12)}
    for val_sample_no in range(np.array(val_data).shape[0]):
        val_samples = val_data[val_sample_no]
        assert np.array(val_samples).shape == (batch_size, 2, 2)
        # Data Processing
        (
            val_proceed,
            val_base_ids,
            val_source_ids,
            base_label_ids,
            val_source_label_ids,
            val_source_label_,
            base_label,
        ) = data_processing(
            model=model,
            samples=val_samples,
            token_length_allowed=token_length_allowed,
            attribute=attribute,
            DEVICE=DEVICE,
            batch_size=batch_size,
        )

        if not val_proceed:
            continue

        val_intervened_base_output, val_predicted_text_ = eval_model(
            val_source_ids, val_base_ids, val_temperature
        )

        val_ground_truth_token_id = val_source_label_ids
        vocab_size = model.tokenizer.vocab_size
        val_ground_truth_one_hot = F.one_hot(
            val_ground_truth_token_id["input_ids"], num_classes=vocab_size
        )
        val_ground_truth_one_hot = val_ground_truth_one_hot.to(dtype=torch.long)
        val_last_token_output = val_intervened_base_output[:, -1, :]
        assert val_ground_truth_one_hot.squeeze(1).shape == val_last_token_output.shape
        val_ground_truth_indices = torch.argmax(
            val_ground_truth_one_hot.squeeze(1), dim=1
        )
        val_ground_truth_indices = val_ground_truth_indices.to(dtype=torch.long)
        loss = loss_fn(val_last_token_output, val_ground_truth_indices)
        total_val_loss += loss.item()

        # Calculate accuracy
        val_predicted_text = [word.split()[0] for word in val_predicted_text_]
        val_source_label = [word.split()[0] for word in val_source_label_]

        total_val_samples_processed += batch_size
        for i in range(len(val_predicted_text)):
            if val_predicted_text[i] == val_source_label[i]:
                matches_val += 1

    if wndb == "True":
        wandb.log(
            {
                f"GPT-2 SS IIA Val {args.layer_intervened}": matches_val
                / total_val_samples_processed,
                f"GPT-2 SS IIA Val Loss {args.layer_intervened}": total_val_loss
                / total_val_samples_processed,
            }
        )
    print(
        f"Validation Accuracy: {matches_val / total_val_samples_processed:.4f}, Validation Loss: {total_val_loss / total_val_samples_processed:.4f}"
    )

    continent_acc = calculate_accuracy(
        eval_model,
        model,
        val_continent_data,
        token_length_allowed,
        attribute,
        batch_size,
        DEVICE,
        val_temperature,
    )
    country_acc = calculate_accuracy(
        eval_model,
        model,
        val_country_data,
        token_length_allowed,
        attribute,
        batch_size,
        DEVICE,
        val_temperature,
    )
    print(f"Continent Accuracy: {continent_acc}, Country Accuracy: {country_acc}")
    if wndb == "True":
        wandb.log(
            {
                f"Val Continent Accuracy {args.layer_intervened}": continent_acc,
                f"Val Country Accuracy {args.layer_intervened}": country_acc,
            }
        )


def test(
    eval_model,
    model,
    test_data,
    test_country_data,
    test_continent_data,
    loss_fn,
    attribute,
    token_length_allowed,
    batch_size,
    temperature_end,
    DEVICE,
    wndb,
):

    eval_model.eval()

    total_test_samples_processed = 0
    total_test_loss = 0.0

    matches_test = 0
    total_test_samples_processed = 0
    total_test_loss = 0

    correct_test = {i: [] for i in range(0, 12)}
    for test_sample_no in range(np.array(test_data).shape[0]):
        test_samples = test_data[test_sample_no]
        assert np.array(test_samples).shape == (batch_size, 2, 2)
        # Data Processing
        (
            proceed,
            test_base_ids,
            test_source_ids,
            test_base_label_ids,
            test_source_label_ids,
            test_source_label_,
            test_base_label,
        ) = data_processing(
            model=model,
            samples=test_samples,
            token_length_allowed=token_length_allowed,
            attribute=attribute,
            DEVICE=DEVICE,
            batch_size=batch_size,
        )

        if not proceed:
            continue

        test_temperature = temperature_end

        test_intervened_base_output, test_predicted_text_ = eval_model(
            test_source_ids, test_base_ids, test_temperature
        )

        test_ground_truth_token_id = test_source_label_ids
        vocab_size = model.tokenizer.vocab_size
        test_ground_truth_one_hot = F.one_hot(
            test_ground_truth_token_id["input_ids"], num_classes=vocab_size
        )
        test_ground_truth_one_hot = test_ground_truth_one_hot.to(dtype=torch.long)
        test_last_token_output = test_intervened_base_output[:, -1, :]
        assert (
            test_ground_truth_one_hot.squeeze(1).shape == test_last_token_output.shape
        )
        test_ground_truth_indices = torch.argmax(
            test_ground_truth_one_hot.squeeze(1), dim=1
        )
        test_ground_truth_indices = test_ground_truth_indices.to(dtype=torch.long)
        test_loss = loss_fn(test_last_token_output, test_ground_truth_indices)
        total_test_loss += test_loss.item()

        # Calculate accuracy
        test_predicted_text = [word.split()[0] for word in test_predicted_text_]
        test_source_label = [word.split()[0] for word in test_source_label_]
        total_test_samples_processed += batch_size
        for i in range(len(test_predicted_text)):
            if test_predicted_text[i] == test_source_label[i]:
                matches_test += 1

    if wndb == "True":
        wandb.log(
            {
                f"GPT-2 SS IIA Test Acc {args.layer_intervened}": matches_test
                / total_test_samples_processed,
                f"GPT-2 SS IIA Test Loss {args.layer_intervened}": total_test_loss
                / total_test_samples_processed,
            }
        )

    continent_acc = calculate_accuracy(
        eval_model,
        model,
        test_continent_data,
        token_length_allowed,
        attribute,
        batch_size,
        DEVICE,
        temperature_end,
    )
    country_acc = calculate_accuracy(
        eval_model,
        model,
        test_country_data,
        token_length_allowed,
        attribute,
        batch_size,
        DEVICE,
        temperature_end,
    )
    print(
        f"Test Continent Accuracy: {continent_acc}, Test Country Accuracy: {country_acc}"
    )
    if wndb == "True":
        wandb.log(
            {
                f"Test Continent Accuracy {args.layer_intervened}": continent_acc,
                f"Test Country Accuracy {args.layer_intervened}": country_acc,
            }
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--device", default="cuda:1", help="Device to run the model on"
    )
    parser.add_argument(
        "-a",
        "--attribute",
        required=True,
        help="name of the attribute on which evaluation is being performned",
    )
    parser.add_argument(
        "-tla",
        "--token_length_allowed",
        required=True,
        type=int,
        help="insert the length you would allow the model to train mask",
    )
    parser.add_argument(
        "-method",
        "--method",
        required=True,
        help="to let know if you want neuron masking, das masking or SAE masking",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=1,
        type=int,
        help="# of epochs on which mask is to be trained",
    )
    parser.add_argument(
        "-ef", "--expansion_factor", default=1, help="expansion factor for SAE"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.001,
        type=float,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "-t",
        "--task",
        required=True,
        help="task to perform, i.e. train or test or total_iia_train",
    )
    parser.add_argument(
        "-svd",
        "--saved_model_path",
        default="gpt2/models/saved_model.pth",
        help="path to the saved model",
    )
    parser.add_argument(
        "-n",
        "--notes",
        default="",
        help="Any notes you want to write for the wandb graph",
    )
    parser.add_argument(
        "-idd",
        "--intervention_divided_data",
        help="The data which is divided for intervention",
    )
    parser.add_argument(
        "-bs", "--batch_size", default=32, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "-lid",
        "--layer_intervened",
        default=0,
        type=int,
        help="Layer intervened for the SAE masking",
    )
    parser.add_argument(
        "-wb", "--wndb", default=False, help="Whether to log the data to wandb or not"
    )

    args = parser.parse_args()
    args.model = "gpt2"
    if args.wndb == "True":
        wandb.init(project="sae_concept_eraser")
        wandb.run.name = f"{args.method}-{args.intervention_divided_data}_intervened-e{args.epochs}-b{args.batch_size}-{args.notes}"
    DEVICE = args.device
    layer_intervened = args.layer_intervened

    (
        model,
        intervened_token_idx,
    ) = config(learning_rate=args.learning_rate, token_length=args.token_length_allowed)
    # model.to(DEVICE)
    training_model = my_model(
        model=model,
        DEVICE=DEVICE,
        method=args.method,
        token_length_allowed=args.token_length_allowed,
        expansion_factor=args.expansion_factor,
        layer_intervened=layer_intervened,
        intervened_token_idx=intervened_token_idx,
        batch_size=args.batch_size,
    )

    training_model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    for name, param in training_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    optimizer = optim.Adam(training_model.parameters(), lr=args.learning_rate)

    (
        train_country_data,
        train_continent_data,
        train_data,
        val_country_data,
        val_continent_data,
        val_data,
        test_country_data,
        test_continent_data,
        test_data,
    ) = train_data_processing(
        args.task, args.intervention_divided_data, args.batch_size
    )

    # Inserting the temperature
    total_step = 0
    # target_total_step = len(batches) * args.epochs
    # TODO: The total number of batches is total_no_samples/batch_len
    batch_size = args.batch_size
    target_total_step = len(train_data) * args.epochs
    temperature_start = 10.0
    temperature_end = 0.1
    temperature_schedule = (
        t.linspace(
            t.tensor(temperature_start),
            t.tensor(temperature_end),
            int(target_total_step),
        )
        .to(t.bfloat16)
        .to(DEVICE)
    )
    print(f"Temp Schedule lenght is {len(temperature_schedule)}")

    temp_idx = 0

    with torch.autograd.set_detect_anomaly(True):
        if args.task == "total_iia_train":
            """
            This correponds to the fact when we are training the model with total intervention and not partial, either on continent or country.
            """
            train(
                train_continent_data=train_continent_data,
                train_country_data=train_country_data,
                val_continent_data=val_continent_data,
                val_country_data=val_country_data,
                training_model=training_model,
                model=model,
                train_data=train_data,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=args.epochs,
                token_length_allowed=args.token_length_allowed,
                attribute=args.attribute,
                temperature_schedule=temperature_schedule,
                batch_size=batch_size,
                DEVICE=DEVICE,
                wndb=args.wndb,
            )

        elif args.task == "train":

            train(
                train_continent_data=train_continent_data,
                train_country_data=train_country_data,
                val_country_data=val_country_data,
                val_continent_data=val_continent_data,
                training_model=training_model,
                model=model,
                train_data=train_data,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=args.epochs,
                token_length_allowed=args.token_length_allowed,
                attribute=args.attribute,
                temperature_schedule=temperature_schedule,
                batch_size=batch_size,
                DEVICE=DEVICE,
                wndb=args.wndb,
            )
            # Assuming training_model.l4_mask is your tensor
            l4_mask_cpu = training_model.l4_mask.to("cpu")  # Move tensor to CPU

            # Create a boolean mask where the condition is true
            mask_greater_than_0_5 = l4_mask_cpu > 0
            mask_equal_to_0 = l4_mask_cpu == 0
            mask_less_than_0 = l4_mask_cpu < 0

            # Sum the mask to get the number of elements satisfying the conditions
            num_elements_greater_than_0_5 = mask_greater_than_0_5.sum().item()
            num_elements_equal_to_0 = mask_equal_to_0.sum().item()
            num_elements_less_than_0 = mask_less_than_0.sum().item()

            print(
                f"Number of elements in l4_mask greater than 0.5: {num_elements_greater_than_0_5}"
            )
            print(
                f"Number of elements in l4_mask equal to 0: {num_elements_equal_to_0}"
            )
            print(
                f"Number of elements in l4 mask less than 0: {num_elements_less_than_0}"
            )

            try:
                with open("masking_stats.json", "r") as f:
                    data = json.load(f)

                data[
                    f"[GPT2-{args.intervention_divided_data}-{args.method}-{args.learning_rate}-{args.batch_size}-{args.layer_intervened}] Number of elements in l4_mask > 0.5"
                ] = num_elements_greater_than_0_5
                data[
                    f"[GPT2-{args.intervention_divided_data}-{args.method}-{args.learning_rate}-{args.batch_size}-{args.layer_intervened}] num elements in l4 masks = 0"
                ] = num_elements_equal_to_0

                data[
                    f"[GPT2-{args.intervention_divided_data}-{args.method}-{args.learning_rate}-{args.batch_size}-{args.layer_intervened}] num elements in l4 masks < 0"
                ] = num_elements_less_than_0

                with open("masking_stats.json", "w") as f:
                    json.dump(data, f)

            except:

                data = {}
                data[
                    f"[GPT2-{args.intervention_divided_data}-{args.method}-{args.learning_rate}-{args.batch_size}-{args.layer_intervened}] Number of elements in l4_mask > 0.5"
                ] = num_elements_greater_than_0_5
                data[
                    f"[GPT2-{args.intervention_divided_data}-{args.method}-{args.learning_rate}-{args.batch_size}-{args.layer_intervened}] num elements in l4 masks = 0"
                ] = num_elements_equal_to_0

                data[
                    f"[GPT2-{args.intervention_divided_data}-{args.method}-{args.learning_rate}-{args.batch_size}-{args.layer_intervened}] num elements in l4 masks < 0"
                ] = num_elements_less_than_0

                with open("masking_stats.json", "w") as f:
                    json.dump(data, f)

            # Save the model
            # torch.save(
            #     training_model.state_dict(),
            #     f"models/saved_model_{args.intervention_divided_data}_{args.method}_{args.model}_e{args.epochs}_lr{args.learning_rate}_layer{args.layer_intervened}.pth",
            # )
            eval_model = my_model(
                model=model,
                DEVICE=DEVICE,
                method=args.method,
                token_length_allowed=args.token_length_allowed,
                expansion_factor=args.expansion_factor,
                layer_intervened=layer_intervened,
                intervened_token_idx=intervened_token_idx,
                batch_size=args.batch_size,
            )

            # Load the state_dict from the saved file
            eval_model.load_state_dict(
                torch.load(
                    f"models/saved_model_{args.intervention_divided_data}_{args.method}_{args.model}_e{args.epochs-1}_lr{args.learning_rate}_layer{args.layer_intervened}.pth"
                ),
                strict=False,
            )

            # model_path = args.saved_model_path
            test(
                eval_model=eval_model,
                model=model,
                test_data=test_data,
                test_country_data=test_country_data,
                test_continent_data=test_continent_data,
                loss_fn=loss_fn,
                token_length_allowed=args.token_length_allowed,
                batch_size=batch_size,
                temperature_end=temperature_end,
                DEVICE=DEVICE,
                attribute=args.attribute,
                wndb=args.wndb,
            )
