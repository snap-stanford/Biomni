import requests

# Replace "TP53" and 5 with own values if needed
payload = {
    "gene_name": "TP53",
    "K": 5
}

response = requests.post("http://localhost:5090/get_rna_seq_archs4", json=payload)

print("Status code:", response.status_code)
print("Response:", response.json())
