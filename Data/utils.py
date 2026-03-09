import json
import re
import csv
import os

def balance_braces(s: str) -> str:
    """Trim trailing '}' if number of '}' exceeds number of '{'."""
    while s.count('{') < s.count('}'):
        s = s.rstrip('}')
    s = s.replace('\n', '')
    s = re.sub(r'\s+', ' ', s)
    return s

def preprocess_response_string(response_text: str) -> str:
    if response_text.startswith('```json'):
        response_text = response_text[7:-3].strip()
    elif response_text.startswith('```') and response_text.endswith('```'):
        response_text = response_text[3:-3].strip()
    response_text = response_text.replace("```", "").replace("json", "").strip()
    # Remove trailing commas
    response_text = re.sub(r',\s*}', '}', response_text)
    response_text = re.sub(r',\s*]', ']', response_text)
    response_text = balance_braces(response_text)
    return response_text

def json_to_Passport(csv_path, js):
    if isinstance(js, str):
        js = json.loads(js)

    fn = js.get("first_name")
    ln = js.get("last_name")
    pid = js.get("id_number")

    if not fn or not ln:
        return

    header = [
        "lastName",
        "firstName",
        "accountID",
        "documentID",
        "personID",
        "passportNumber"
    ]

    row = [
        ln,
        fn,
        "",
        f"{ln[0]}{fn[0]}_Doc1",
        f"{ln[0]}{fn[0]}0",
        pid
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(csv_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def json_to_DriverLicence(csv_path, js):
    if isinstance(js, str):
        js = json.loads(js)

    last_name = js.get("last_name")
    first_name = js.get("first_name")
    birth_date = js.get("birthDate") or js.get("dob") or js.get("birth_date")
    account_id = js.get("accountID") or js.get("account_id") or ""
    issue_date = js.get("issueDate") or js.get("issue_date")
    expiry_date = js.get("expieryDate") or js.get("expiryDate") or js.get("expiry_date")

    if not first_name or not last_name:
        return

    header = [
        "lastName",
        "firstName",
        "birthDate",
        "accountID",
        "issueDate",
        "expieryDate",
        "documentID",
        "personID"
    ]

    row = [
        last_name,
        first_name,
        birth_date,
        account_id,
        issue_date,
        expiry_date,
        f"{last_name[0]}{first_name[0]}_Doc0",
        f"{last_name[0]}{first_name[0]}0"
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(csv_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def json_to_Person(csv_path, js):
    if isinstance(js, str):
        js = json.loads(js)

    first_name = js.get("first_name")
    last_name = js.get("last_name")
    role = js.get("role") or ""

    if not first_name or not last_name:
        return

    header = [
        "fullName",
        "personID",
        "assetID",
        "documentID",
        "addressID",
        "role"
    ]

    full_name = f"{first_name} {last_name}"

    row = [
        full_name,
        f"{last_name[0]}{first_name[0]}0",
        "",
        f"{last_name[0]}{first_name[0]}_Doc0",
        "",
        role
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(csv_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def json_to_BirthCertificate(csv_path, js):
    if isinstance(js, str):
        js = json.loads(js)

    last_name = js.get("last_name")
    first_name = js.get("first_name")
    birth_date = js.get("birthDate") 

    if not last_name or not first_name:
        return

    header = [
        "lastName",
        "firstName",
        "birthDate",
        "documentID",
        "personID"
    ]

    row = [
        last_name,
        first_name,
        birth_date,
        f"{last_name[0]}{first_name[0]}_Doc2",
        f"{last_name[0]}{first_name[0]}0"
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open(csv_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)





def json_to_Income(csv_path, js):
    if isinstance(js, str):
        js = json.loads(js)

    fn = js.get("first_name")
    ln = js.get("last_name")
    income = js.get("incomeAmount")

    if not fn or not ln or income in (None, ""):
        return

    # Build personID and incomeID from initials (same style as your Passport function)
    person_id = f"{ln[0]}{fn[0]}0"
    income_id = f"{person_id}_Income"

    # If documentID not provided, fall back to "<LN><FN>_Doc0"
    doc_id = f"{ln[0]}{fn[0]}_Doc0"

    header = ["personID", "incomeAmount", "incomeID", "documentID"]
    row = [person_id, income, income_id, doc_id]

    # Create file with header if not exists
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Ensure the file ends with a newline before appending
    with open(csv_path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")

    # Append row
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
