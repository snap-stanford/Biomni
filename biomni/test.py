from biomni.agent import A1

# Initialize the agent with data path, Data lake will be automatically downloaded on first run (~11GB)
agent = A1(llm='gpt-5-mini', expected_data_lake_files = [])

# Execute biomedical tasks using natural language
# Execute biomedical tasks using natural language
agent.go("Is YES1 amplified as ecDNA or BFB in CCLE? In which samples? Use the amplicon_table tool and CCLE.csv. Be efficient.")
# agent.go("show me all BFBs in CCLE at breast cancer samples using amplicon_table and CCLE.csv")
# agent.go("Summarize the size distribution of all BFB amplifications (Captured interval length using amplicon_table and CCLE.csv")