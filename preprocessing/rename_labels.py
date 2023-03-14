"""Description. Rename labels in CLINC150 for texts from banking domain."""

import json
from typing import Dict

DATA_DIR = "./../datasets/"

new_banking_labels = {"pin_change": "change_pin"}

new_credit_cards_labels = {
    "replacement_card_duration": "card_delivery_estimate", 
    "expiration_date": "card_about_to_expire", 
    "report_lost_card": "lost_or_stolen_card", 
    "card_declined": "declined_card_payment"
}

def update_labels(domains: Dict, domain_name: str, mapping: Dict): 
    domains[domain_name] = [
        mapping[label] 
        if label in mapping.keys()
        else label
        for label in mapping
    ]

if __name__ == "__main__": 

    with open(f"{DATA_DIR}CLINC150.json") as f: 
        ds = json.load(f)

    with open(f"{DATA_DIR}CLINC150_domains.json") as f: 
        domains = json.load(f)

    with open(f"{DATA_DIR}b77_label_mapping.json") as f: 
        b77_label_mapping = json.load(f)

    update_labels(domains, "banking", new_banking_labels)
    update_labels(domains, "credit_cards", new_credit_cards_labels)

    assert "change_pin" in domains["banking"]
    assert "card_about_to_expire" in domains["credit_cards"]

    with open(f"{DATA_DIR}CLINC150_domains.json", "w") as f: 
        json.dump(domains, f)