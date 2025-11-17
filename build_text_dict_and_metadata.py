#!/usr/bin/env python3
"""
build_text_dict_and_metadata.py

Scans the Dataset folder and builds:
 - out/text_dict.json    -> mapping text_code -> text (first non-empty line)
 - out/metadata.csv      -> rows: group,subject,session,text_code,txt_path,wav_path,wav_missing

Usage:
    python build_text_dict_and_metadata.py ./Dataset ./out
"""
import os
import sys
import json
import csv

def first_line_of_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s:
                    return s
            return ""
    except Exception:
        return ""

def scan_dataset(root):
    text_dict = {}
    rows = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    for group_name in sorted(os.listdir(root)):
        group_path = os.path.join(root, group_name)
        if not os.path.isdir(group_path):
            continue
        for subject_name in sorted(os.listdir(group_path)):
            subject_path = os.path.join(group_path, subject_name)
            if not os.path.isdir(subject_path):
                continue
            for session_name in sorted(os.listdir(subject_path)):
                session_path = os.path.join(subject_path, session_name)
                if not os.path.isdir(session_path):
                    continue

                txt_map = {}
                wav_map = {}
                for dirpath, _, filenames in os.walk(session_path):
                    for fn in filenames:
                        base, ext = os.path.splitext(fn)
                        ext = ext.lower()
                        if ext == '.txt':
                            txt_map[base] = os.path.join(dirpath, fn)
                        elif ext == '.wav':
                            wav_map[base] = os.path.join(dirpath, fn)

                all_codes = sorted(set(txt_map.keys()) | set(wav_map.keys()))
                if not all_codes:
                    rows.append({
                        'group': group_name,
                        'subject': subject_name,
                        'session': session_name,
                        'text_code': None,
                        'txt_path': None,
                        'wav_path': None,
                        'wav_missing': None
                    })
                else:
                    for code in all_codes:
                        txt_path = txt_map.get(code)
                        wav_path = wav_map.get(code)
                        wav_missing = (wav_path is None)
                        if txt_path:
                            txt_text = first_line_of_file(txt_path)
                            if code not in text_dict and txt_text:
                                text_dict[code] = txt_text
                        rows.append({
                            'group': group_name,
                            'subject': subject_name,
                            'session': session_name,
                            'text_code': code,
                            'txt_path': txt_path,
                            'wav_path': wav_path,
                            'wav_missing': wav_missing
                        })
    return text_dict, rows

def save_outputs(text_dict, rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'text_dict.json')
    csv_path = os.path.join(out_dir, 'metadata.csv')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(text_dict, f, indent=2, ensure_ascii=False)

    fieldnames = ['group','subject','session','text_code','txt_path','wav_path','wav_missing']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return json_path, csv_path

def print_summary(text_dict, rows):
    total_codes = len(text_dict)
    total_entries = len([r for r in rows if r.get('text_code')])
    missing_count = len([r for r in rows if r.get('text_code') and r.get('wav_missing')])
    print("Dataset summary:")
    print(f"  distinct text codes found: {total_codes}")
    print(f"  subject/session/text entries: {total_entries}")
    print(f"  entries with missing .wav: {missing_count}")
    sample = sorted(text_dict.keys())[:20]
    print(f"  sample codes: {sample}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python build_text_dict_and_metadata.py /path/to/Dataset /path/to/output_dir")
        sys.exit(1)
    dataset_root = sys.argv[1]
    out_dir = sys.argv[2]

    text_dict, rows = scan_dataset(dataset_root)
    json_path, csv_path = save_outputs(text_dict, rows, out_dir)
    print("Saved text dictionary to:", json_path)
    print("Saved metadata CSV to:", csv_path)
    print_summary(text_dict, rows)

if __name__ == '__main__':
    main()
