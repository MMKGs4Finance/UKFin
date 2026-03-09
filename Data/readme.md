# Fannie Mae Data Processing and Document Extraction Pipeline

This repository implements a three-stage pipeline for processing Fannie Mae monthly performance data and extracting structured information from financial documents using LLM/VLM models.

The overall goal is to:
1. Label loan performance into three discrete risk categories (Good / Average / Bad)
2. Select and curate subsets of Fannie Mae data for downstream use
3. Extract structured entities from images and PDFs into CSV tables

The raw Fannie Mae monthly performance data used in this pipeline is stored under `./CIRT-122014-102025`.

---

## 1. Fannie Mae Data Processing

### ExtractLabel.ipynb
**Purpose**  
Label Fannie Mae monthly performance records and aggregate loan-level risk into three categories: **Good**, **Average**, and **Bad**.

**Method**  
- Use delinquency status and zero balance codes from the pipe-delimited format  
- Normalize delinquency values (e.g. `R`, `NA`, `0` → `00`)  
- Record-level rules:
  - **Bad**: severe zero balance code or delinquency ≥ 3 months  
  - **Average**: delinquency of 1–2 months  
  - **Good**: otherwise current and non-terminated  

**Loan-level aggregation**  
- **Bad** if any record is Bad  
- **Average** if no Bad records and at least one Average  
- **Good** if all records are Good  

**Outputs**  
- `loan_records.csv`: raw monthly records for selected loan IDs  
- `loan_labeled.csv`: record-level and loan-level labels (`Good`, `Average`, `Bad`)


## 2. Tabular to CSV Tables

### SelectData.ipynb
**Purpose**  
Extract and transform selected Fannie Mae monthly performance records into structured CSV tables for downstream use.

**Method**  
- Select target loans and reporting periods from Fannie Mae monthly data  
- Parse records in the pipe-delimited format  
- Clean and standardize identifiers to ensure consistency across tables  
- Construct stable entity identifiers (e.g. account, contract, payment, transaction)  
- Extract financial, temporal, and status-related fields without value imputation  

**Outputs**  
A set of normalized CSV tables representing key entities and relationships,
stored under the `./Result/` directory, including:
- LoanAccount
- Account
- Contract
- PaymentSchedule
- Transaction
- Payments
- Application
- Asset and assessment-related tables
- …
---

## 3. Document Extraction with LLM / VLM
This stage extracts structured information from financial documents (images and PDFs) using large language and vision-language models.

### 3.1 Directory Structure

- `DOC_ROOT = ./Doc/`  
  Root directory containing input documents, organized by folder name  
  Example folders:
  - Passport
  - DriverLicence
  - Person
  - BirthCertificate

- `RESULT_ROOT = ./Result/`  
  Output directory for generated CSV tables

---

### 3.2 Processing Workflow

For each folder under `DOC_ROOT`:

#### Image files (PNG, JPG, JPEG)

1. Run VLM-based OCR to extract raw text
2. Construct a strict JSON extraction prompt
3. Use an LLM to extract required fields
4. Validate JSON structure and required keys
5. Save results to a folder-specific CSV file

#### PDF files

1. Load PDF text using `build_medical_documents`
2. Apply the same strict JSON extraction logic
3. Save extracted fields (e.g. income) to CSV

---

### 3.3 Models

- `ImageOcrTwin`  
  Vision-language model for OCR from images
- `TextKeywordTwin`  
  Language model for structured field extraction
- `OCR_extract`  
  Retry-based extraction with validation:
  - Output must be a JSON object
  - Required keys must exist
  - Field-specific constraints enforced (e.g. numeric ID formats)

---

### 3.4 Supported Document Types

| Folder Name        | Output CSV Function        |
|--------------------|----------------------------|
| Passport           | `json_to_Passport`         |
| DriverLicence      | `json_to_DriverLicence`    |
| Person             | `json_to_Person`           |
| BirthCertificate   | `json_to_BirthCertificate` |
| PDF (Income)       | `json_to_Income`           |

---

### 3.5 Extraction Rules

All extraction prompts enforce the following constraints:

- Return **ONE valid JSON object only**
- Use values only if **explicitly and clearly shown**
- If missing or uncertain, output `null`
- Do **not** guess, infer, normalize, convert, or reformat values

**Date fields**
- Accepted format: `DD/MM/YYYY`
- Any other format or ambiguous value must be returned as `null`

---

### 3.6 Runtime Environment

- Python version: **Python 3.11**
- Hardware: **2 GPUs** (used for LLM/VLM inference)
- GPU acceleration is required for efficient OCR and document-level extraction

---

## End-to-End Usage Summary

1. Run `1ExtractLabel.ipynb` to label records and loans as Good / Average / Bad
2. Run `2SelectData.ipynb` to curate the dataset
3. Run `3main.py` to populate structured CSV tables from images and PDFs

This pipeline produces clean, interpretable, and ontology-ready tabular data suitable for downstream analytics, risk modeling, and knowledge graph construction.


### Environment Setup

This project is tested with Python 3.11 in an Apptainer container.

Create and activate a Conda environment:
conda create -n envi python=3.11
conda activate envi
pip install -r requirements.txt



