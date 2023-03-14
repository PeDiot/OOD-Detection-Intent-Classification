import random
from typing import Optional, Dict, Tuple

from tqdm import tqdm

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd 

# important to keep the full path
# choose your own path 
# used to load ATIS and BITEXT datasets

def prep_dataset(
    config_name,
    config,
    tokenizer,
    train_max_size=-1,
    validation_max_size=-1,
    test_max_size=-1,
    ood_label: Optional[int]=None, 
    data_path: Optional[str]=None
) -> Tuple[Dataset, Dataset, Dataset]:
    sentence1_key, sentence2_key = config["keys"]

    datasets = None

    if config_name in ("mnli", "rte"):
        datasets = load_glue(config_name)
    elif config_name in ("snli",):
        datasets = load_snli()
    elif config_name == "tweet_eval":
        datasets = load_tweet_eval()
    elif config_name == "amazon_reviews_multi":
        datasets = load_amazon_reviews_multi(config["language"])
    elif config_name == "go_emotions":
        datasets = load_go_emotions()
    elif config_name == "sst2":
        datasets = load_sst2()
    elif config_name == "20ng":
        datasets = load_20ng()
    elif config_name == "trec":
        datasets = load_trec()
    elif config_name == "trec_fine":
        datasets = load_trec(labels="label-fine")
    elif config_name == "imdb":
        datasets = load_imdb()
    elif config_name == "yelp":
        datasets = load_yelp()
    # ===== Datasets for the OOD project =====
    elif config_name == "b77":
        datasets = load_b77()
    elif config_name == "atis": 
        datasets = load_atis(data_path)
    elif config_name == "bitext": 
        datasets = load_bitext(data_path)
    # =========================================
    elif config_name == "massive":
        datasets = load_massive()
    elif config_name == "emotion":
        datasets = load_emotion()
    elif config_name == "twitterfin":
        datasets = load_twitterfin()
    elif config_name in ("fr_xnli", "fr_pawsx", "fr_cls"):
        datasets = load_flue(config_name)
    elif config_name == "fr_book_reviews":
        datasets = load_fr_book_reviews()
    elif config_name == "fr_allocine":
        datasets = load_fr_allocine()
    elif config_name in ("fr_xstance", "es_xstance", "de_xstance"):
        datasets = load_xstance(config_name)
    elif config_name in ("fr_swiss_judgement", "de_swiss_judgement"):
        datasets = load_swiss_judgement(config_name)
    elif config_name in (
        "fr_tweet_sentiment",
        "es_tweet_sentiment",
        "de_tweet_sentiment",
    ):
        datasets = load_tweet_multil_sentiments(config_name)
    elif config_name in ("fr_pawsx", "es_pawsx", "de_pawsx"):
        datasets = load_pawsx(config_name)
    elif config_name == "de_lexarg":
        datasets = load_german_arg_mining(config_name)
    elif config_name == "es_tweet_inde":
        datasets = load_twitter_catinde(config_name)
    elif config_name == "es_cine":
        datasets = load_muchocine(config_name)
    elif config_name == "es_sst2":
        datasets = load_sst2_es(config_name)
    else:
        raise ValueError(f"Unknown dataset {config_name}")

    def preprocess_function(examples):
        result = {}
        inputs = (
            examples[sentence1_key]
            if sentence2_key is None
            else examples[sentence1_key] + " " + examples[sentence2_key]
        )

        result["text"] = inputs

        result["labels"] = examples["label"] if "label" in examples else 0

        if ood_label != None: 
            result["ood"] = ood_label

        return result

    train_dataset = (
        list(map(preprocess_function, datasets["train"]))
        if "train" in datasets
        else None
    )
    dev_dataset = (
        list(map(preprocess_function, datasets["validation"]))
        if "validation" in datasets
        else None
    )
    test_dataset = (
        list(map(preprocess_function, datasets["test"])) if "test" in datasets else None
    )

    if train_max_size > 0: 
        train_dataset = Dataset.from_list(train_dataset[:train_max_size])
    if validation_max_size > 0: 
        dev_dataset = Dataset.from_list(dev_dataset[:validation_max_size])
    if test_max_size > 0: 
        test_dataset = Dataset.from_list(test_dataset[:test_max_size])

    return train_dataset, dev_dataset, test_dataset


def load_swiss_judgement(task_name):
    lang = task_name.split("_")[0]
    datasets = load_dataset("swiss_judgment_prediction", lang)

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_twitter_catinde(task_name):
    datasets = load_dataset("catalonia_independence", "spanish")

    train = [{"text": x["TWEET"], "label": x["LABEL"]} for x in datasets["train"]]
    validation = [
        {"text": x["TWEET"], "label": x["LABEL"]} for x in datasets["validation"]
    ]
    test = [{"text": x["TWEET"], "label": x["LABEL"]} for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_muchocine(task_name):
    datasets = load_dataset("muchocine")
    dataset = [
        {"text": s["review_summary"], "label": int(s["star_rating"])}
        for s in datasets["train"]
    ]

    train = dataset[:2000] + dataset[:2000] + dataset[:2000]
    validation = dataset[1000:2500]
    test = dataset[2500:]

    return {"train": train, "validation": validation, "test": test}


def load_pawsx(task_name):
    lang = task_name.split("_")[0]
    datasets = load_dataset("paws-x", lang)

    def mk_sample(sample):
        return {
            "text": sample["sentence1"] + " " + sample["sentence2"],
            "label": sample["label"],
        }

    datasets = datasets.map(mk_sample, remove_columns=["sentence1", "sentence2"])

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_sst2_es(task_name=None):
    datasets = load_dataset("mrm8488/sst2-es-mt")

    def mk_sample(x):
        return {"text": x["sentence_es"], "label": x["label"]}

    datasets = datasets.map(mk_sample, remove_columns=["sentence_es", "label"])

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_tweet_multil_sentiments(task_name):
    lang = task_name.split("_")[0]
    lang_map = {"en": "english", "es": "spanish", "fr": "french", "de": "german"}
    datasets = load_dataset("cardiffnlp/tweet_sentiment_multilingual", lang_map[lang])

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_german_arg_mining(task_name):
    dataset = load_dataset("joelito/german_argument_mining", ignore_verifications=True)

    label_map = {"conclusion": 0, "definition": 1, "subsumption": 2, "other": 3}
    train = [
        {"text": x["input_sentence"], "label": label_map[x["label"]]}
        for x in dataset["train"]
    ]
    validation = [
        {"text": x["input_sentence"], "label": label_map[x["label"]]}
        for x in dataset["validation"]
    ]
    test = [
        {"text": x["input_sentence"], "label": label_map[x["label"]]}
        for x in dataset["test"]
    ]

    return {"train": train, "validation": validation + train[:2000], "test": test}


def load_xstance(task_name):
    lang = task_name.split("_")[0]

    datasets = load_dataset("strombergnlp/x-stance", lang)

    def mk_sample(sample):
        return {
            "text": sample["question"] + " " + sample["comment"],
            "label": sample["label"],
        }

    datasets = datasets.map(mk_sample)

    train = [x for x in datasets["train"]]
    validation = [x for x in datasets["validation"]] + train[:2000]
    test = [x for x in datasets["test"]]

    return {"train": train, "validation": validation, "test": test}


def load_flue(task_name):
    task_map = {
        "fr_xnli": "XNLI",
        "fr_cls": "CLS",
        "fr_pawsx": "PAWS-X",
    }

    if task_name == "fr_pawsx":
        datasets = load_dataset("flue", task_map[task_name])
        ds = {}

        def mk_sample(x):
            return {
                "text": x["sentence1"] + " " + x["sentence2"],
                "label": x["label"],
            }

        datasets = datasets.map(mk_sample, remove_columns=["sentence1", "sentence2"])
        ds["train"] = [x for x in datasets["train"]][:30000]
        ds["validation"] = [x for x in datasets["validation"]] + [
            x for x in datasets["test"]
        ][30000 : 30000 + 2000]
        ds["test"] = [x for x in datasets["test"]]

    elif task_name == "fr_xnli":

        datasets = load_dataset("flue", "XNLI")

        def mk_sample(x):
            return {
                "text": x["premise"] + " " + x["hypo"],
                "label": x["label"],
            }

        ds = {}
        datasets = datasets.map(mk_sample, remove_columns=["premise", "hypo"])
        ds["train"] = [x for x in datasets["train"]][:50000]
        ds["validation"] = [x for x in datasets["validation"]] + [
            x for x in datasets["test"]
        ][50000 : 50000 + 2000]
        ds["test"] = [x for x in datasets["test"]]

    elif task_name == "fr_cls":
        datasets = load_dataset("flue", "CLS")
        ds = {}
        ds["train"] = [x for x in datasets["train"]]
        ds["validation"] = [x for x in datasets["test"]][:2000] + [
            x for x in datasets["train"]
        ][:2000]
        ds["test"] = [x for x in datasets["test"]][:2000]

    else:
        raise ValueError("Unknown task {}".format(task_name))

    return ds


def load_emotion():
    dataset = load_dataset("emotion")
    dd = {
        "train": [x for x in dataset["train"]],
        "validation": [x for x in dataset["validation"]],
        "test": [x for x in dataset["test"]],
    }

    return dd


def load_b77():
    label_mapping = {
        'Refund_not_showing_up': 0,
        'activate_my_card': 1,
        'age_limit': 2,
        'apple_pay_or_google_pay': 3,
        'atm_support': 4,
        'automatic_top_up': 5,
        'balance_not_updated_after_bank_transfer': 6,
        'balance_not_updated_after_cheque_or_cash_deposit': 7,
        'beneficiary_not_allowed': 8,
        'cancel_transfer': 9,
        'card_about_to_expire': 10,
        'card_acceptance': 11,
        'card_arrival': 12,
        'card_delivery_estimate': 13,
        'card_linking': 14,
        'card_not_working': 15,
        'card_payment_fee_charged': 16,
        'card_payment_not_recognised': 17,
        'card_payment_wrong_exchange_rate': 18,
        'card_swallowed': 19,
        'cash_withdrawal_charge': 20,
        'cash_withdrawal_not_recognised': 21,
        'change_pin': 22,
        'compromised_card': 23,
        'contactless_not_working': 24,
        'country_support': 25,
        'declined_card_payment': 26,
        'declined_cash_withdrawal': 27,
        'declined_transfer': 28,
        'direct_debit_payment_not_recognised': 29,
        'disposable_card_limits': 30,
        'edit_personal_details': 31,
        'exchange_charge': 32,
        'exchange_rate': 33,
        'exchange_via_app': 34,
        'extra_charge_on_statement': 35,
        'failed_transfer': 36,
        'fiat_currency_support': 37,
        'get_disposable_virtual_card': 38,
        'get_physical_card': 39,
        'getting_spare_card': 40,
        'getting_virtual_card': 41,
        'lost_or_stolen_card': 42,
        'lost_or_stolen_phone': 43,
        'order_physical_card': 44,
        'passcode_forgotten': 45,
        'pending_card_payment': 46,
        'pending_cash_withdrawal': 47,
        'pending_top_up': 48,
        'pending_transfer': 49,
        'pin_blocked': 50,
        'receiving_money': 51,
        'request_refund': 52,
        'reverted_card_payment?': 53,
        'supported_cards_and_currencies': 54,
        'terminate_account': 55,
        'top_up_by_bank_transfer_charge': 56,
        'top_up_by_card_charge': 57,
        'top_up_by_cash_or_cheque': 58,
        'top_up_failed': 59,
        'top_up_limits': 60,
        'top_up_reverted': 61,
        'topping_up_by_card': 62,
        'transaction_charged_twice': 63,
        'transfer_fee_charged': 64,
        'transfer_into_account': 65,
        'transfer_not_received_by_recipient': 66,
        'transfer_timing': 67,
        'unable_to_verify_identity': 68,
        'verify_my_identity': 69,
        'verify_source_of_funds': 70,
        'verify_top_up': 71,
        'virtual_card_not_working': 72,
        'visa_or_mastercard': 73,
        'why_verify_identity': 74,
        'wrong_amount_of_cash_received': 75,
        'wrong_exchange_rate_for_cash_withdrawal': 76
    }
    datasets = load_dataset("banking77")  
    datasets = datasets.align_labels_with_mapping(label_mapping, "label")
    
    return datasets

def load_atis(data_path: str):
    """Description. ATIS Airline Travel Information System."""
    dataset = pd.read_csv(f"{data_path}atis.csv")
    dataset = DatasetDict({"test": Dataset.from_pandas(dataset)})

    return dataset

def load_bitext(data_path: str): 
    """Description. Bitext - Customer Service Tagged Training Dataset for Intent Detection"""
    dataset = pd.read_csv(f"{data_path}/bitext.csv")
    dataset = dataset\
        .rename(columns={"utterance": "text", "intent": "label"})\
        .loc[:, ["text", "label"]]

    dataset = DatasetDict({"test": Dataset.from_pandas(dataset)})

    return dataset


def load_twitterfin():
    datasets = load_dataset("zeroshot/twitter-financial-news-sentiment")

    _train = [x for x in datasets["train"]]
    _val = [x for x in datasets["validation"]]

    train = _train[:6000]
    val = _train[6000:]
    test = _val

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def load_massive(lang="en-US"):
    datasets = load_dataset("AmazonScience/massive", lang)

    train_dataset = [
        {"text": x["utt"], "label": x["intent"]} for x in datasets["train"]
    ]
    dev_dataset = [
        {"text": x["utt"], "label": x["intent"]} for x in datasets["validation"]
    ]
    test_dataset = [{"text": x["utt"], "label": x["intent"]} for x in datasets["test"]]

    return {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}


def load_glue(task):
    datasets = load_dataset("glue", task)
    if task == "mnli":
        test_dataset = [d for d in datasets["test_matched"]] + [
            d for d in datasets["test_mismatched"]
        ]
        datasets["test"] = test_dataset

    if task == "rte":
        datasets = {
            "validation": list(datasets["validation"]) + list(datasets["train"])[:2000],
            "test": list(datasets["test"]),
            "train": list(datasets["train"]),
        }
    if task == "sst2":
        datasets = {
            "validation": list(datasets["validation"]) + list(datasets["train"])[:2000],
            "test": list(datasets["test"]),
            "train": list(datasets["train"]),
        }
    return datasets


def load_snli():
    datasets = load_dataset("snli")
    return datasets


def load_20ng():
    all_subsets = (
        "18828_alt.atheism",
        "18828_comp.graphics",
        "18828_comp.os.ms-windows.misc",
        "18828_comp.sys.ibm.pc.hardware",
        "18828_comp.sys.mac.hardware",
        "18828_comp.windows.x",
        "18828_misc.forsale",
        "18828_rec.autos",
        "18828_rec.motorcycles",
        "18828_rec.sport.baseball",
        "18828_rec.sport.hockey",
        "18828_sci.crypt",
        "18828_sci.electronics",
        "18828_sci.med",
        "18828_sci.space",
        "18828_soc.religion.christian",
        "18828_talk.politics.guns",
        "18828_talk.politics.mideast",
        "18828_talk.politics.misc",
        "18828_talk.religion.misc",
    )
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for i, subset in enumerate(all_subsets):
        dataset = load_dataset("newsgroup", subset)["train"]
        examples = [{"text_scr": d["text"], "label": i} for d in dataset]
        random.shuffle(examples)
        num_train = int(0.8 * len(examples))
        num_dev = int(0.1 * len(examples))
        train_dataset += examples[:num_train]
        dev_dataset += examples[num_train : num_train + num_dev]
        test_dataset += examples[num_train + num_dev :]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_fr_book_reviews():
    datasets = load_dataset("Abirate/french_book_reviews", ignore_verifications=True)
    dd = [
        {"text": x["reader_review"], "label": int(x["label"] + 1)}
        for x in datasets["train"]
    ]
    train = dd[:5000]
    val = dd[4000:7000]
    test = dd[7000:]

    return {"train": train, "validation": val, "test": test}


def load_fr_allocine():
    datasets = load_dataset("allocine")
    train = [{"text": x["review"], "label": x["label"]} for x in datasets["train"]][
        :50000
    ]
    val = [{"text": x["review"], "label": x["label"]} for x in datasets["validation"]][
        :5000
    ]
    test = [{"text": x["review"], "label": x["label"]} for x in datasets["test"]][
        :20000
    ]

    return {"train": train, "validation": val, "test": test}


def load_trec(labels="label-coarse"):  # or fine-label
    datasets = load_dataset("trec")
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [
        {
            "text_scr": train_dataset[i]["text"],
            "label": train_dataset[i][labels],
        }
        for i in idxs[-num_reserve:]
    ]
    train_dataset = [
        {
            "text_scr": train_dataset[i]["text"],
            "label": train_dataset[i][labels],
        }
        for i in idxs[:-num_reserve]
    ]
    test_dataset = [{"text_scr": d["text"], "label": d[labels]} for d in test_dataset]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_yelp():
    datasets = load_dataset("yelp_polarity")
    train_dataset = datasets["train"]
    idxs = list(range(len(train_dataset) // 10))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) // 10 * 0.1)
    dev_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[-num_reserve:]
    ]
    train_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[:-num_reserve]
    ]
    test_dataset = datasets["test"]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_imdb():
    datasets = load_dataset("imdb", ignore_verifications=True)  # /plain_text')
    train_dataset = datasets["train"]
    unsup_dataset = datasets["unsupervised"]
    idxs = list(range(len(train_dataset)))
    random.shuffle(idxs)
    num_reserve = int(len(train_dataset) * 0.1)
    dev_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[-num_reserve:]
    ] + [
        {"text": unsup_dataset[i]["text"], "label": unsup_dataset[i]["label"]}
        for i in range(8000)
    ]
    train_dataset = [
        {"text": train_dataset[i]["text"], "label": train_dataset[i]["label"]}
        for i in idxs[:-num_reserve]
    ]
    test_dataset = datasets["test"]
    datasets = {"train": train_dataset, "validation": dev_dataset, "test": test_dataset}
    return datasets


def load_amazon_reviews_multi(language):
    dataset = load_dataset(
        "amazon_reviews_multi", language
    )  # all_languages de fr en es ja zh
    # product_category
    labels = sorted(list(set(dataset["train"]["product_category"])))[::-1]
    print("Numbers of labels Amazon", len(labels))
    dict_labels = {label: i for i, label in enumerate(labels)}
    datasets = {
        "train": dataset["train"],
        "validation": dataset["train"],
        "test": dataset["test"],
    }
    new_datasets = {"train": [], "validation": [], "test": []}
    for split_name in new_datasets.keys():
        arr = datasets[split_name]["product_category"]
        for k, v in tqdm(dict_labels.items()):
            arr = [x.replace(k, str(v)) for x in arr]
        arr = [int(i) for i in arr]
        review_body = datasets[split_name]["review_body"]
        for i in tqdm(range(len(arr))):
            new_datasets[split_name].append(
                {
                    "label": arr[i],
                    "review_body": review_body[i],
                }
            )
    return new_datasets


def load_tweet_eval():
    datasets = load_dataset("tweet_eval", "emoji")  # /plain_text')
    datasets = {
        "train": datasets["train"],
        "validation": datasets["train"],
        "test": datasets["test"],
    }
    return datasets


def load_go_emotions():
    datasets = load_dataset("go_emotions", "simplified")  # label = json
    datasets = {
        "train": datasets["train"],
        "validation": datasets["train"],
        "test": datasets["test"],
    }
    new_datasets = {"train": [], "validation": [], "test": []}
    for split_name in new_datasets.keys():
        arr = datasets[split_name]["labels"]
        arr = [x[0] for x in arr]
        texts = datasets[split_name]["text"]
        for i in tqdm(range(len(arr))):
            new_datasets[split_name].append(
                {
                    "label": arr[i],
                    "text": texts[i],
                }
            )

    new_datasets["validation"] += new_datasets["train"][:5000]
    return new_datasets


def load_sst2():
    datasets = load_dataset("glue", "sst2", ignore_verifications=True)

    train = [{"text": x["sentence"], "label": x["label"]} for x in datasets["train"]]
    validation = [
        {"text": x["sentence"], "label": x["label"]} for x in datasets["validation"]
    ]
    test = [{"text": x["sentence"], "label": x["label"]} for x in datasets["test"]]

    datasets = {"train": train, "validation": validation, "test": test}
    return datasets


def prep_model(model_name, config: Optional[Dict] = None):
    if config is None:
        config = {"label": 2}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config["label"]
    )

    return model, tokenizer
