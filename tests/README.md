# Biomni Agent Test Suite

This directory contains comprehensive test suites for the Biomni agent, demonstrating the integration of ChEMBL Beaker cheminformatic tools, database APIs, and drug safety analysis capabilities.

## Test Files

### 1. `agent_test_chembl_cheminformatics.py`
**Comprehensive Cheminformatic Analysis for Drug Discovery**

This test demonstrates the full capabilities of ChEMBL Beaker integration for drug discovery workflows.

#### Test 1: Comprehensive Cheminformatic Analysis
- **Chemical Structure Validation**: Canonicalization and standardization using ChEMBL Beaker
- **Physicochemical Property Analysis**: Comprehensive property calculation with ChEMBL and RDKit descriptors
- **Format Conversion**: Multi-format molecular representation (SMILES, InChI, InChIKey, MOL files)
- **Structural Alert Analysis**: Safety assessment using toxicophore detection
- **Database Integration**: Cross-reference with PubChem for additional chemical information
- **Drug Discovery Assessment**: Evaluation of drug-likeness and development potential

#### Test 2: Multi-Compound Comparison
- **Individual Compound Analysis**: Systematic evaluation of multiple compounds
- **Comparative Analysis**: Molecular weight, lipophilicity, and structural alert comparison
- **Database Cross-Reference**: PubChem queries for biological activity and safety data
- **Lead Optimization Assessment**: Ranking and structural modification suggestions
- **Formulation Analysis**: Solubility and bioavailability assessment

### 2. `agent_test_drug_safety_analysis.py`
**Drug Safety Analysis and Risk Assessment**

This test demonstrates comprehensive drug safety analysis using multiple data sources and cheminformatic tools.

#### Test 1: Comprehensive Drug Safety Analysis
- **Cheminformatic Safety Assessment**: Structural alert analysis using ChEMBL Beaker
- **FDA Safety Database Analysis**: Adverse event reports and safety signal analysis
- **Clinical Trial Safety Data**: Safety outcomes from clinical studies
- **Drug-Drug Interaction Analysis**: Interaction mechanism and severity assessment
- **Risk Assessment and Mitigation**: Systematic safety evaluation and monitoring strategies
- **Regulatory Compliance Analysis**: Safety guideline adherence and compliance assessment

#### Test 2: Drug Combination Safety Analysis
- **Structural Interaction Analysis**: Molecular compatibility assessment
- **Drug-Drug Interaction Assessment**: Comprehensive interaction analysis
- **Safety Signal Analysis**: Combination-specific adverse event analysis
- **Clinical Trial Safety Review**: Combination therapy safety outcomes
- **Patient Population Risk Assessment**: Population-specific safety concerns
- **Risk Mitigation and Monitoring**: Monitoring protocols and safety measures
- **Regulatory and Clinical Guidelines**: Compliance and guideline assessment

## Key Features Demonstrated

### ChEMBL Beaker Integration (Cheminformatic Tools)
- **Structure Standardization**: Chemical structure validation and canonicalization
- **Property Calculation**: Comprehensive physicochemical property analysis
- **Format Conversion**: Multi-format molecular representation
- **Structural Alert Analysis**: Safety assessment using toxicophore detection
- **Molecular Image Generation**: SVG and 2D coordinate generation
- **Clear Distinction**: Separate from ChEMBL Database bioactivity queries

### Enhanced ChEMBL Database Integration
- **Multiple Query Types**: Molecule name, ChEMBL ID, SMILES similarity, natural language, direct endpoint
- **Reliable API Access**: Robust error handling and fallback mechanisms
- **Bioactivity Data**: Comprehensive bioactivity and safety information retrieval
- **Molecular Properties**: Drug-likeness scores and physicochemical properties
- **Cross-Reference Capabilities**: Integration with other databases and tools
- **Clear Distinction**: Separate from ChEMBL Beaker cheminformatic tools
- **Improved API Patterns**: Full-text search, substructure/similarity search, drug metadata, ATC classifications
- **Efficient Queries**: Support for 'only=' parameter to reduce response fields
- **Practical Usage**: Based on official ChEMBL documentation and real-world usage patterns

### Database Integration
- **PubChem Queries**: Chemical information retrieval and cross-referencing
- **FDA Database Access**: Adverse event reports and safety signal analysis
- **Clinical Trial Data**: Safety outcomes from clinical studies
- **Drug-Drug Interaction Data**: Comprehensive interaction assessment

### Workflow Automation
- **Systematic Analysis**: Structured workflows for drug discovery and safety assessment
- **Multi-Source Integration**: Combined analysis from multiple databases
- **Decision Support**: Clear recommendations and risk assessment
- **Research Documentation**: Detailed logs and comprehensive reports

## Running the Tests

### Prerequisites
- Biomni agent with ChEMBL Beaker integration
- Access to OpenAI API (or other LLM provider)
- Required Python packages installed

### Execution
```bash
# Run ChEMBL cheminformatic analysis tests
python tests/agent_test_chembl_cheminformatics.py

# Run drug safety analysis tests
python tests/agent_test_drug_safety_analysis.py

# Test improved ChEMBL database function
python tests/test_improved_chembl.py

# Test ChEMBL Database vs Beaker distinction
python tests/test_chembl_distinction.py

# Test improved ChEMBL API patterns
python tests/test_improved_chembl_api.py
```

### Expected Outputs
Each test provides:
- **Detailed Research Logs**: Step-by-step analysis with timestamps
- **Comprehensive Reports**: Structured analysis results and recommendations
- **Risk Assessments**: Clear risk levels and mitigation strategies
- **Decision Support**: Actionable recommendations for drug development

## Use Cases

### Drug Discovery
- Lead compound identification and optimization
- Structure-activity relationship analysis
- Drug-likeness assessment
- Formulation strategy development

### Drug Safety
- Preclinical safety assessment
- Clinical trial safety monitoring
- Post-marketing safety surveillance
- Risk-benefit analysis

### Regulatory Compliance
- Safety guideline adherence
- Regulatory submission support
- Compliance monitoring
- Risk communication

### Clinical Practice
- Drug combination safety assessment
- Patient-specific risk evaluation
- Monitoring protocol development
- Safety guideline implementation

## Technical Capabilities

### Cheminformatic Analysis
- **Molecular Structure Processing**: SMILES, InChI, MOL file handling
- **Property Calculation**: LogP, molecular weight, drug-likeness scores
- **Safety Assessment**: Structural alert and toxicophore detection
- **Format Conversion**: Multi-format molecular representation

### Database Integration
- **Multi-Source Queries**: PubChem, FDA, ClinicalTrials.gov
- **Cross-Reference Analysis**: Integrated data from multiple sources
- **Real-Time Updates**: Current safety and regulatory information
- **Comprehensive Coverage**: Chemical, biological, and clinical data

### Workflow Automation
- **Systematic Analysis**: Structured workflows for complex analyses
- **Decision Support**: Clear recommendations and risk assessment
- **Documentation**: Comprehensive research logs and reports
- **Scalability**: Handles multiple compounds and complex scenarios

## Integration Benefits

### For Drug Discovery
- **Accelerated Analysis**: Automated cheminformatic workflows
- **Comprehensive Assessment**: Multi-source data integration
- **Risk Mitigation**: Early safety assessment and alert detection
- **Decision Support**: Clear recommendations for lead optimization

### For Drug Safety
- **Proactive Monitoring**: Early detection of safety signals
- **Comprehensive Assessment**: Multi-dimensional safety analysis
- **Risk Communication**: Clear risk levels and mitigation strategies
- **Regulatory Support**: Compliance and guideline adherence

### For Clinical Practice
- **Patient Safety**: Individualized risk assessment
- **Drug Combination Safety**: Polypharmacy risk evaluation
- **Monitoring Protocols**: Evidence-based safety monitoring
- **Clinical Decision Support**: Informed therapeutic decisions

These tests demonstrate the full potential of the Biomni agent for drug discovery, safety analysis, and clinical decision support, showcasing the power of integrated cheminformatic tools and database APIs.
