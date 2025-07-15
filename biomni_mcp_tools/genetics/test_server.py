import requests


def test_get_rna_seq_archs4():
    url = "http://localhost:8000/get_rna_seq_archs4"
    payload = {"gene_name": "TP53", "K": 5}
    response = requests.post(url, json=payload)
    print(response.json())


if __name__ == "__main__":
    test_get_rna_seq_archs4()
