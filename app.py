#!/usr/bin/env python3
"""
Streamlit app for iCalendar to CSV Extractor with Patient Matching
Complete version with all sophisticated matching logic
"""

import streamlit as st
import csv
import re
from datetime import datetime, timedelta
from icalendar import Calendar
import pytz
from typing import List, Dict, Optional, Tuple
import pandas as pd
from difflib import SequenceMatcher
import logging
from pathlib import Path
import unicodedata
import io
import zipfile
import tempfile
import os

# Configure page
st.set_page_config(
    page_title="iCalendar Appointment Tool",
    page_icon="📅",
    layout="wide"
)

# Setup logging for the app
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize session state for storing results
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'files_to_download' not in st.session_state:
    st.session_state.files_to_download = {}

# Helper functions from original script
_WS_RE = re.compile(r"\s+")

def _normalize_token(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")  # NBSP
    s = s.replace("\u200B", "")   # zero-width space
    s = s.replace("\u200C", "")   # ZWNJ
    s = s.replace("\u200D", "")   # ZWJ
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    return s

_CLEAN_NOISE_RE = re.compile(r"(?:\bTMS\b|#\d+|\d+/\d+|\bF\d{2}\.\d\b|\b[FR]\d{2}\.\d\b|[()])", re.IGNORECASE)

def _clean_person_token(s: str) -> str:
    s = _CLEAN_NOISE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _surname_key(s: str) -> str:
    n = _normalize_token(s)

    # Armenian harmonization:
    if n.endswith("ielyan"):
        n = n[:-6] + "ilyan"
    elif n.endswith("yelyan"):
        n = n[:-6] + "lyan"
    elif n.endswith("elyan"):
        n = n[:-5] + "lyan"

    # General Slavic/Armenian feminine/masculine harmonization
    repl = [
        ("skaya", "sky"), ("tskaya", "tsky"), ("vskaya", "vsky"), ("zkaya", "zky"),
        ("ckaya", "cky"), ("shaya", "shay"), ("chaya", "chay"), ("zhaya", "zhay"),
        ("aya", "y"), ("ova", "ov"), ("eva", "ev"), ("ina", "in"), ("kina", "kin"), ("yina", "yin"),
    ]
    for suf, to in repl:
        if n.endswith(suf):
            return n[: -len(suf)] + to
    return n

class PatientData:
    """Patient information"""
    def __init__(self, first_name: str, last_name: str, prn: str, insurance: str, doctor: str):
        self.first_name = first_name.strip()
        self.last_name = last_name.strip()
        self.prn = prn.strip()
        self.insurance = insurance.strip()
        self.doctor = doctor.strip()
    
    def full_name(self) -> str:
        """Return name in 'Last Name, First Name' format"""
        return f"{self.last_name}, {self.first_name}"

def parse_ical_file(ical_content: bytes) -> Calendar:
    """Parse the iCalendar file and return Calendar object"""
    try:
        calendar = Calendar.from_ical(ical_content)
        return calendar
    except Exception as e:
        raise Exception(f"Error parsing iCalendar file: {str(e)}")

def is_target_day(dt: datetime, target_days: List[int]) -> bool:
    """Check if the datetime falls on one of the target days
    Days: Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6"""
    return dt.weekday() in target_days

def is_in_target_months(dt: datetime, target_months: List[int], target_year: int) -> bool:
    """Check if the datetime is in the target months and year"""
    return dt.year == target_year and dt.month in target_months

def extract_appointments(calendar: Calendar, months: List[int], year: int, days: List[int] = None) -> List[Dict]:
    """Extract appointments that fall on specified days in specified months
    
    Args:
        calendar: iCalendar object
        months: List of months (1-12)
        year: Target year
        days: List of weekdays (0=Monday, 6=Sunday). Default is Monday and Friday.
    """
    if days is None:
        days = [0, 4]  # Default to Monday and Friday
    
    appointments = []
    
    for component in calendar.walk():
        if component.name == "VEVENT":
            summary = str(component.get('summary', 'No Title'))
            description = str(component.get('description', ''))
            location = str(component.get('location', ''))
            
            dtstart = component.get('dtstart')
            if dtstart is None:
                continue
                
            start_dt = dtstart.dt
            
            if isinstance(start_dt, datetime):
                event_datetime = start_dt
            else:
                event_datetime = datetime.combine(start_dt, datetime.min.time())
            
            if event_datetime.tzinfo is None:
                event_datetime = pytz.UTC.localize(event_datetime)
            
            dtend = component.get('dtend')
            if dtend:
                end_dt = dtend.dt
                if isinstance(end_dt, datetime):
                    end_datetime = end_dt
                else:
                    end_datetime = datetime.combine(end_dt, datetime.min.time())
                
                if end_datetime.tzinfo is None:
                    end_datetime = pytz.UTC.localize(end_datetime)
            else:
                end_datetime = event_datetime + timedelta(hours=1)
            
            rrule = component.get('rrule')
            if rrule:
                from dateutil.rrule import rrulestr
                
                rrule_str = f"DTSTART:{event_datetime.strftime('%Y%m%dT%H%M%SZ')}\n"
                rrule_str += f"RRULE:{rrule.to_ical().decode('utf-8')}"
                
                try:
                    rule = rrulestr(rrule_str)
                    start_of_year = datetime(year, 1, 1, tzinfo=pytz.UTC)
                    end_of_year = datetime(year + 1, 1, 1, tzinfo=pytz.UTC)
                    
                    for occurrence in rule.between(start_of_year, end_of_year, inc=True):
                        if is_target_day(occurrence, days) and is_in_target_months(occurrence, months, year):
                            duration = end_datetime - event_datetime
                            occurrence_end = occurrence + duration
                            
                            appointments.append({
                                'title': summary,
                                'start_date': occurrence.strftime('%m/%d/%Y'),
                                'start_time': occurrence.strftime('%H:%M'),
                                'end_time': occurrence_end.strftime('%H:%M'),
                                'day_of_week': occurrence.strftime('%A'),
                                'location': location,
                                'description': description.replace('\n', ' ').replace('\r', '')
                            })
                except Exception as e:
                    logger.warning(f"Could not process recurring event '{summary}': {str(e)}")
            else:
                if is_target_day(event_datetime, days) and is_in_target_months(event_datetime, months, year):
                    appointments.append({
                        'title': summary,
                        'start_date': event_datetime.strftime('%m/%d/%Y'),
                        'start_time': event_datetime.strftime('%H:%M'),
                        'end_time': end_datetime.strftime('%H:%M'),
                        'day_of_week': event_datetime.strftime('%A'),
                        'location': location,
                        'description': description.replace('\n', ' ').replace('\r', '')
                    })
    
    appointments.sort(key=lambda x: (datetime.strptime(x['start_date'], '%m/%d/%Y'), x['start_time']))
    logger.info(f"Extracted {len(appointments)} appointments")
    return appointments

def parse_patient_name(name: str) -> Tuple[str, str]:
    """Parse 'LastName, FirstName' format with support for complex names."""
    name = str(name).strip()
    
    if ',' in name:
        parts = name.split(',', 1)
        last_name = parts[0].strip()
        first_name = parts[1].strip() if len(parts) > 1 else ""
        
        # Handle complex last names like "Russell (Kwon)"
        if '(' in last_name:
            main_last_name = last_name.split('(')[0].strip()
            return first_name, main_last_name
        
        return first_name, last_name
    else:
        # Fallback: split and assume last word is last name
        parts = name.split()
        if len(parts) >= 2:
            first_name = ' '.join(parts[:-1])
            last_name = parts[-1]
        else:
            first_name = ""
            last_name = name
        return first_name, last_name

def load_patient_list(excel_content: bytes) -> Dict[str, PatientData]:
    """Load patient list from Excel file with enhanced indexing"""
    try:
        df = pd.read_excel(io.BytesIO(excel_content), header=None)
        logger.info(f"Loaded patient list with {len(df)} records")
        patients: Dict[str, PatientData] = {}

        for _, row in df.iterrows():
            # Assuming columns: 0=Name, 1=PRN, 2=Insurance, 4=Doctor
            name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            prn = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            insurance = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
            doctor = str(row.iloc[4]).strip() if len(row) > 4 and pd.notna(row.iloc[4]) else ""

            if not name or name.lower() == "nan":
                continue

            # parse name
            first_name, last_name = parse_patient_name(name)

            # build the patient object
            patient = PatientData(first_name, last_name, prn, insurance, doctor)
            logger.info(f"Loaded patient: {patient.full_name()} (PRN: {prn}, Insurance: {insurance}, Doctor: {doctor})")

            # Enhanced indexing for better matching
            first_key = _normalize_token(first_name)
            last_key = _normalize_token(last_name)
            name_key = f"{first_key}_{last_key}"
            patients[name_key] = patient
            
            # Index by display "last, first" too
            disp_key = f"{_normalize_token(last_name)}, {_normalize_token(first_name)}"
            if f"display_{disp_key}" not in patients:
                patients[f"display_{disp_key}"] = patient

            # Index by surname key to help last-name-only resolution
            surname_key = _surname_key(last_name)
            patients.setdefault(f"lnk_{surname_key}", []).append(patient)

            # Index by PRN for quick lookup
            if prn:
                patients[f"prn_{_normalize_token(prn)}"] = patient

        logger.info(f"Successfully loaded {len(patients)} patient records")
        return patients
    except Exception as e:
        raise Exception(f"Error loading patient list: {str(e)}")

def calculate_string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher"""
    s1 = _normalize_token(s1)
    s2 = _normalize_token(s2)
    return SequenceMatcher(None, s1, s2).ratio()

def match_patient_by_lastname_only(candidate_last: str,
                                   patients: Dict[str, PatientData],
                                   min_sim: float = 0.65,
                                   max_results: int = 5) -> List[Tuple[PatientData, float]]:
    cand = _normalize_token(candidate_last)
    cand_key = _surname_key(candidate_last)

    patient_to_score: Dict[PatientData, float] = {}

    # Fast path from surname bucket (start with 0.95)
    bucket = patients.get(f"lnk_{cand_key}", [])
    for p in bucket:
        patient_to_score[p] = max(patient_to_score.get(p, 0), 0.95)

    # Fuzzy sweep over real PatientData entries, update max score per patient
    for key, p in patients.items():
        if key.startswith(("prn_", "lnk_")): 
            continue
        if not isinstance(p, PatientData): 
            continue

        pl = _normalize_token(p.last_name or "")
        best = calculate_string_similarity(cand, pl)
        for token in pl.split():
            best = max(best, calculate_string_similarity(cand, token))

        # Armenian tail nudge
        if cand.endswith("yan") and pl.endswith("yan"):
            core_sim = calculate_string_similarity(cand[:-3], pl[:-3])
            if core_sim >= 0.65:
                best = max(best, 0.90)

        # Additional nudge for ian/yan variations
        if (cand.endswith("yan") and pl.endswith("ian")) or (cand.endswith("ian") and pl.endswith("yan")):
            core_sim = calculate_string_similarity(cand[:-3], pl[:-3])
            if core_sim >= 0.65:
                best = max(best, 0.90)

        if best >= min_sim:
            patient_to_score[p] = max(patient_to_score.get(p, 0), best)

    # Sort by score desc
    hits = sorted(patient_to_score.items(), key=lambda x: x[1], reverse=True)[:max_results]
    logger.info(f"Lastname-only hits for '{candidate_last}': {[(h[0].full_name(), h[1]) for h in hits]}")
    return hits

def match_patient(name: str, patients: Dict[str, PatientData]) -> Tuple[Optional[PatientData], bool, float]:
    first_name, last_name = parse_patient_name(name)

    # Normalized exacts FIRST
    first_lower = _normalize_token(first_name)
    last_lower = _normalize_token(last_name)
    has_full_name = bool(first_lower) and bool(last_lower)

    if has_full_name:
        # Exact by display key
        disp_key = f"display_{last_lower}, {first_lower}"
        if disp_key in patients and isinstance(patients[disp_key], PatientData):
            logger.info(f"Exact display match for '{name}' -> '{patients[disp_key].full_name()}'")
            return patients[disp_key], False, 1.0

        # Exact by canonical key
        name_key = f"{first_lower}_{last_lower}"
        if name_key in patients and isinstance(patients[name_key], PatientData):
            logger.info(f"Exact canonical match for '{name}' -> '{patients[name_key].full_name()}'")
            return patients[name_key], False, 1.0

    # Fuzzy matching
    first_parts = [part.strip() for part in first_lower.replace(',', ' ').split() if part.strip()]
    last_parts = [part.strip() for part in last_lower.replace(',', ' ').split() if part.strip()]
    matches = []
    
    for key, patient in patients.items():
        if key.startswith(('prn_', 'lnk_')):
            continue
        if not isinstance(patient, PatientData):
            continue
            
        patient_first = _normalize_token(patient.first_name)
        patient_last = _normalize_token(patient.last_name)
        
        # Calculate similarity scores
        first_name_score = 0
        last_name_score = 0
        
        if first_parts and patient_first:
            max_first_score = 0
            for first_part in first_parts:
                sim = calculate_string_similarity(first_part, patient_first)
                max_first_score = max(max_first_score, sim)
                
                for patient_word in patient_first.split():
                    sim = calculate_string_similarity(first_part, patient_word)
                    max_first_score = max(max_first_score, sim)
            
            first_name_score = max_first_score
        
        if last_parts and patient_last:
            max_last_score = 0
            # Check surname key equality
            if last_parts:
                for last_part in last_parts:
                    if _surname_key(last_part) == _surname_key(patient_last):
                        max_last_score = max(max_last_score, 0.98)
            for last_part in last_parts:
                sim = calculate_string_similarity(last_part, patient_last)
                max_last_score = max(max_last_score, sim)
                
                for patient_word in patient_last.split():
                    sim = calculate_string_similarity(last_part, patient_word)
                    max_last_score = max(max_last_score, sim)
            
            last_name_score = max_last_score
        
        # Acceptance gate
        if (first_name_score >= 0.60 and last_name_score >= 0.60) or \
            (last_name_score >= 0.90 and first_name_score >= 0.50):
            overall_score = (first_name_score + last_name_score) / 2
            matches.append((patient, overall_score, first_name_score, last_name_score))
            if overall_score >= 0.80:
                logger.info(f"High fuzzy score for '{name}' -> '{patient.full_name()}' (Overall: {overall_score:.1%}, First: {first_name_score:.1%}, Last: {last_name_score:.1%})")

    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        best_match = matches[0]
        logger.info(f"Fuzzy match: '{name}' -> '{best_match[0].full_name()}' "
                   f"(Overall: {best_match[1]:.1%}, First: {best_match[2]:.1%}, Last: {best_match[3]:.1%})")
        is_ambiguous = len([m for m in matches if m[1] >= 0.85]) > 1
        return best_match[0], is_ambiguous, best_match[1]
    
    # Fallback: first-initial + last-name
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", name)
    for tok in raw_tokens:
        cand_ln = _normalize_token(tok)
        if patients.get(f"lnk_{_surname_key(cand_ln)}"):
            ln_matches = match_patient_by_lastname_only(cand_ln, patients)
            if ln_matches:
                top = ln_matches[0]
                logger.info(f"Lastname-only bucket hit: '{name}' -> '{top[0].full_name()}' ({top[1]:.1%})")
                return top[0], False, max(0.90, top[1])

    # Fallback: lastname-only when no full name
    if not has_full_name:
        for candidate_last in filter(None, [last_lower, first_lower]):
            ln_matches = match_patient_by_lastname_only(candidate_last, patients)
            if ln_matches:
                top = ln_matches[0]
                is_ambiguous = False
                if len(ln_matches) > 1:
                    diff = top[1] - ln_matches[1][1]
                    is_ambiguous = diff < 0.10
                    logger.info(f"Lastname-only fuzzy: '{name}' -> '{top[0].full_name()}' ({top[1]:.1%}) ambiguous={is_ambiguous}")
                    return top[0], is_ambiguous, top[1]
    
    # Last-resort: exact display equality
    display_key = f"{_normalize_token(last_name)}, {_normalize_token(first_name)}"
    for key, p in patients.items():
        if key.startswith(("prn_", "lnk_")) or not isinstance(p, PatientData):
            continue
        disp = f"{_normalize_token(p.last_name)}, {_normalize_token(p.first_name)}"
        if disp == display_key:
            logger.info(f"Last-resort exact display match for '{name}' -> '{p.full_name()}'")
            return p, False, 1.0

    logger.warning(f"No patient match found for: {name}")
    return None, False, 0.0

def extract_metadata(title: str) -> Tuple[str, str]:
    """Extract CPT codes and TMS session numbers from title"""
    metadata = ""
    
    # Extract TMS session numbers
    tms_pattern = r'TMS\s*#?\s*(\d+)'
    tms_match = re.search(tms_pattern, title, re.IGNORECASE)
    if tms_match:
        metadata = f"TMS #{tms_match.group(1)}"
        title = re.sub(tms_pattern, '', title, flags=re.IGNORECASE).strip()
    
    # Extract CPT-like codes
    cpt_pattern = r'\b(\d{2}/\d{2})\b'
    cpt_match = re.search(cpt_pattern, title)
    if cpt_match:
        if metadata:
            metadata += f" | CPT {cpt_match.group(1)}"
        else:
            metadata = f"CPT {cpt_match.group(1)}"
        title = re.sub(cpt_pattern, '', title).strip()
    
    return title.strip(), metadata

def normalize_name_format(name: str) -> str:
    """Convert 'FirstName LastName' to 'LastName, FirstName' format"""
    name = name.strip()

    # Already in "Last, First" format
    if ',' in name:
        return name
    
    # Split by whitespace
    parts = [p for p in name.split() if p] # Remove empty strings
    
    if len(parts) == 0:
        return name
    elif len(parts) == 1:
        # Single name - assume it's last name
        return name
    elif len(parts) == 2:
        # "First Last" → "Last, First"
        # OR "Initial Last"
        return f"{parts[1]}, {parts[0]}"
    elif len(parts) == 3:
        # Could be:
        # "First Middle Last" → "Last, First Middle"
        # "First Last1 Last2" → "Last1 Last2, First"
        # Heuristic: Use last word as last name, rest as first names
        last_name = parts[-1]
        first_names = ' '.join(parts[:-1])
        return f"{last_name}, {first_names}"
    else:
        # Multiple parts - assume last word is last name
        last_name = parts[-1]
        first_names = ' '.join(parts[:-1])
        return f"{last_name}, {first_names}"

def split_combined_names(title: str, patients: Dict[str, PatientData]) -> List[Tuple[str, str]]:
    """
    Return [(normalized_name, metadata)] from a raw appointment title.
    Handles:
      - "First Last, First Last" (two full people)
      - "Last First, First" (one last with two firsts)
      - "Last, First First" (after normalize_name_format)
      - "Last1, Last2, Last3, Last4" (all bare last names)
      - "A and B" form for last names
      - "Last First First" (same last, 2 given names)
      - noise like '13/36', 'TMS #29', diagnoses -> stripped
    """
    clean_title, metadata = extract_metadata(title)
    clean_title = _clean_person_token(clean_title)

    names: List[Tuple[str, str]] = []

    # Split around " and " as a last-name joiner when there's no commas
    if " and " in clean_title and "," not in clean_title:
        parts = [p.strip() for p in clean_title.split(" and ") if p.strip()]
        if len(parts) == 2 and all(" " not in p for p in parts):
            return [(p, metadata) for p in parts]

    # Comma-based split first (it’s most informative)
    parts = [p.strip() for p in clean_title.split(",") if p.strip()]
    if len(parts) == 0:
        return []
    if len(parts) == 1:
        # Single segment -> normalize to "Last, First First..."
        normalized = normalize_name_format(parts[0])
        # If it's "LN FN FN" and LN exists in patients, split as two under same last
        words = parts[0].split()
        if len(words) == 3:
            ln, f1, f2 = words[0], words[1], words[2]
            if f"lnk_{_surname_key(ln)}" in patients:
                return [(f"{ln}, {f1}", metadata), (f"{ln}, {f2}", metadata)]

        if normalized and "," in normalized:
            last, firsts = [p.strip() for p in normalized.split(",", 1)]
            # Expand multi-firsts (e.g., "Larisa Igor" => two rows)
            first_parts = [f for f in firsts.split() if f.isalpha() and len(f) > 1]
            if len(first_parts) > 1:
                for fp in first_parts:
                    names.append((f"{last}, {fp}", metadata))
                return names
            if firsts:
                return [(f"{last}, {firsts}", metadata)]
        # Couldn’t normalize; fall back as-is
        return [(parts[0], metadata)]
    
    # Special-case: "Last First, First" -> same last, two people
    if len(parts) >= 2:
        seg0_words = parts[0].split()
        if len(seg0_words) == 2 and parts[1]:
            last_candidate, first0 = seg0_words[0].strip(), seg0_words[1].strip()
            if f"lnk_{_surname_key(last_candidate)}" in patients:
                out = [(f"{last_candidate}, {first0}", metadata)]
                # split any additional first names in subsequent segments
                for j in range(1, len(parts)):
                    for fn in parts[j].split():
                        fn = _clean_person_token(fn)
                        if fn.isalpha() and len(fn) > 1:
                            out.append((f"{last_candidate}, {fn}", metadata))
                return out

    # Check if all tokens are single words
    tokens = [_clean_person_token(p) for p in parts if p]
    # All tokens are single alphabetic words
    if all((" " not in t and t.replace("-", "").replace("'", "").isalpha()) for t in tokens):

        # First: do a surname check BEFORE pairing
        def _looks_like_lastname(t: str) -> bool:
            n = _normalize_token(t)
            return (f"lnk_{_surname_key(n)}" in patients) or any(
                (k.endswith(f"_{n}")) for k in patients.keys()
                if not k.startswith(("prn_", "lnk_"))
            )

        lname_hits = sum(_looks_like_lastname(t) for t in tokens)

        # If 3+ look like last names, treat as list of last names
        if lname_hits >= max(3, len(tokens)):
            return [(t, metadata) for t in tokens]

        # Otherwise, if even count, fall back to alternating Last, First pairing
        if len(tokens) % 2 == 0:
            out = []
            for i in range(0, len(tokens), 2):
                out.append((f"{tokens[i]}, {tokens[i+1]}", metadata))
            return out
    
    # Detect “Last, First [First...]” (same last name, multiple firsts)
    first_part = tokens[0]
    if " " not in first_part and len(tokens) >= 2:
        last = first_part
        out = []
        for i in range(1, len(tokens)):
            for fn in tokens[i].split():
                fn = _clean_person_token(fn)
                if fn.isalpha() and len(fn) > 1:
                    out.append((f"{last}, {fn}", metadata))
        if out:
            return out

    # Handle each segment independently
    out = []
    for seg in tokens:
        seg = seg.strip()
        if not seg:
            continue
        words = [w for w in seg.split() if w.isalpha()]
        if len(words) == 2:
            # Try "First Last" then "Last First"
            f1, f2 = words
            cand1 = f"{f2}, {f1}"
            cand2 = f"{f1}, {f2}"
            # Pick the one whose last name exists in patients (surname key)
            k1 = f"lnk_{_surname_key(f2)}"
            k2 = f"lnk_{_surname_key(f1)}"
            if k1 in patients:
                out.append((cand1, metadata))
            elif k2 in patients:
                out.append((cand2, metadata))
            else:
                out.append((cand1, metadata)) # default
        elif len(words) == 3:
            # Likely "Last First First"
            ln, f1, f2 = words[0], words[1], words[2]
            out.append((f"{ln}, {f1}", metadata))
            out.append((f"{ln}, {f2}", metadata))
        else:
            # Fallback: try normalize_name_format (may produce "Last, First First...")
            norm = normalize_name_format(seg)
            if norm and "," in norm:
                out.append((norm, metadata))
            else:
                out.append((seg, metadata))
    return out

def process_appointments_with_patients(appointments: List[Dict], 
                                      patients: Dict[str, PatientData]) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Process appointments and match with patient records
    Returns: (matched_appointments, unmatched_appointments, summary_stats)
    """
    matched = []
    unmatched = []
    
    stats = {
        'total_appointments': len(appointments),
        'total_names_extracted': 0,
        'matched_patients': 0,
        'unmatched_entries': 0,
        'fuzzy_matches': 0,
        'ambiguous_matches': 0,
        'split_entries': 0,
        'metadata_extracted': 0,
        'rejected_low_confidence': 0
    }
    
    # Minimum confidence threshold
    MIN_CONFIDENCE = 0.75
    
    for apt in appointments:
        # Strip the title and use that as the canonical original_title
        original_title = apt['title'].strip()
        names_with_metadata = split_combined_names(original_title, patients)
        
        # --- De-dup names per appointment (by normalized "last,first") ---
        seen = set()
        deduped = []
        for nm, md in names_with_metadata:
            fn, ln = parse_patient_name(nm)
            key = f"{_normalize_token(ln)},{_normalize_token(fn)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append((nm, md))
        names_with_metadata = deduped

        # If no valid names extracted, add to unmatched for review
        if not names_with_metadata:
            # These are skipped entries (Michigan, Oregon, etc.) or parsing failures
            unmatched_entry = {
                'original_name': original_title,
                'date': apt['start_date'],
                'start_time': apt['start_time'],
                'end_time': apt['end_time'],
                'day_of_week': apt['day_of_week'],
                'full_title': original_title,
                'codes': ''
            }
            unmatched.append(unmatched_entry)
            continue
        
        stats['total_names_extracted'] += len(names_with_metadata)
        
        if len(names_with_metadata) > 1:
            stats['split_entries'] += 1
            logger.info(f"Split '{original_title}' into {len(names_with_metadata)} names")
        
        # After names_with_metadata = deduped
        seen_patients = set() # de-dup AFTER matching by PRN or normalized full name

        for name_tuple in names_with_metadata:
            # Unpack the tuple - name_tuple is (name, metadata)
            name, metadata = name_tuple
            
            if metadata:
                stats['metadata_extracted'] += 1
            
            patient, is_ambiguous, confidence = match_patient(name, patients)
            
            # CRITICAL: Reject matches below minimum confidence threshold
            if patient and confidence < MIN_CONFIDENCE:
                patient = None
                stats['rejected_low_confidence'] += 1
            
            if patient:
                # Always use normalized full name for de-dup to avoid skipping different patients with same PRN
                pid = f"{_normalize_token(patient.last_name)},{_normalize_token(patient.first_name)}"
                logger.info(f"Computed PID for '{patient.full_name()}' (from '{name}'): {pid} (PRN was '{patient.prn}')")
                if pid in seen_patients:
                    logger.warning(f"Skipping duplicate PID {pid} for name '{name}' matched to '{patient.full_name()}'")
                    continue  # skip duplicate patient within this appointment
                seen_patients.add(pid)
                logger.info(f"Appending matched entry for name '{name}' matched to '{patient.full_name()}'")
                matched_entry = {
                    'name': patient.full_name(),
                    'date': apt['start_date'],
                    'start_time': apt['start_time'],
                    'end_time': apt['end_time'],
                    'day_of_week': apt['day_of_week'],
                    'prn': patient.prn,
                    'insurance': patient.insurance,
                    'doctor': patient.doctor,
                    'codes': metadata,  # CPT codes and TMS session info
                    'original_title': original_title,  # Use stripped version
                    'confidence': f"{confidence:.1%}"
                }
                matched.append(matched_entry)
                stats['matched_patients'] += 1
                
                if confidence < 1.0:
                    stats['fuzzy_matches'] += 1
                if is_ambiguous:
                    stats['ambiguous_matches'] += 1
            else:
                unmatched_entry = {
                    'original_name': name,
                    'date': apt['start_date'],
                    'start_time': apt['start_time'],
                    'end_time': apt['end_time'],
                    'day_of_week': apt['day_of_week'],
                    'full_title': original_title, # Use stripped version
                    'codes': metadata
                }
                unmatched.append(unmatched_entry)
                stats['unmatched_entries'] += 1
    
    return matched, unmatched, stats

def create_csv_content(matched: List[Dict]) -> str:
    """Create CSV content from matched appointments"""
    output = io.StringIO()
    
    if not matched:
        logger.warning("No matched appointments to write")
        return ""
    
    fieldnames = ['name', 'date', 'start_time', 'end_time', 'day_of_week', 
                  'prn', 'insurance', 'doctor', 'codes', 'original_title', 'confidence']
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(matched)

    logger.info(f"Successfully wrote {len(matched)} matched appointments to {output}")

    return output.getvalue()

def create_excel_content(unmatched: List[Dict]) -> bytes:
    """Create Excel content from unmatched entries"""
    if not unmatched:
        logger.info("No unmatched entries to write")
        return b""
    
    df = pd.DataFrame(unmatched)
    output = io.BytesIO()
    df.to_excel(output, index=False, sheet_name='Unmatched')
    logger.info(f"Successfully wrote {len(unmatched)} unmatched entries to {output}")
    return output.getvalue()

def _c(x): 
    try: return float(x.strip('%'))
    except: return 0.0

def generate_summary_report(stats: Dict, matched: List[Dict]) -> str:
    """Generate summary report"""
    report_lines = [
        "=" * 60,
        "APPOINTMENT PROCESSING SUMMARY REPORT",
        "=" * 60,
        "",
        f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OVERALL STATISTICS:",
        f"  Total Appointments Processed: {stats['total_appointments']}",
        f"  Total Names Extracted: {stats['total_names_extracted']}",
        f"  Entries Split (multiple patients): {stats['split_entries']}",
        f"  Metadata Extracted (CPT/TMS): {stats['metadata_extracted']}",
        "",
        "MATCHING RESULTS:",
        f"  Successfully Matched Patients: {stats['matched_patients']}",
        f"  Unmatched Entries: {stats['unmatched_entries']}",
        f"  Rejected (Low Confidence <75%): {stats.get('rejected_low_confidence', 0)}",
        f"  Fuzzy Matches: {stats['fuzzy_matches']}",
        f"  Ambiguous Matches: {stats['ambiguous_matches']}",
        "",
    ]
    
    if stats['total_names_extracted'] > 0:
        match_rate = stats['matched_patients']/stats['total_names_extracted']*100
        report_lines.append(f"  Match Rate: {match_rate:.1f}%")
    
    report_lines.append("")
    
    if matched:
        # Group by day
        by_day = {}
        for m in matched:
            day = m['day_of_week']
            by_day[day] = by_day.get(day, 0) + 1
        
        report_lines.append("APPOINTMENTS BY DAY:")
        for day, count in sorted(by_day.items()):
            report_lines.append(f"  {day}: {count}")
        report_lines.append("")
        
        # Group by doctor
        by_doctor = {}
        for m in matched:
            doc = m['doctor'] if m['doctor'] else "Unknown"
            by_doctor[doc] = by_doctor.get(doc, 0) + 1
        
        report_lines.append("APPOINTMENTS BY DOCTOR:")
        for doc, count in sorted(by_doctor.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {doc}: {count}")
        report_lines.append("")
        
        # Show CPT/TMS statistics
        with_codes = sum(1 for m in matched if m.get('codes'))
        if with_codes > 0:
            report_lines.append("CODES/PROCEDURES:")
            report_lines.append(f"  Appointments with CPT/TMS codes: {with_codes}")
            report_lines.append("")
        
        # Confidence distribution
        report_lines.append("CONFIDENCE DISTRIBUTION:")
        perfect = sum(1 for m in matched if _c(m['confidence']) == 100.0)
        high = sum(1 for m in matched if 90.0 <= _c(m['confidence']) < 100.0)
        medium = sum(1 for m in matched if 80.0 <= _c(m['confidence']) < 90.0)
        acceptable = sum(1 for m in matched if 75.0 <= _c(m['confidence']) < 80.0)
        report_lines.append(f"  Perfect matches (100%): {perfect}")
        report_lines.append(f"  High confidence (90-99%): {high}")
        report_lines.append(f"  Medium confidence (80-89%): {medium}")
        report_lines.append(f"  Acceptable confidence (75-79%): {acceptable}")
        report_lines.append("")
    
    report_lines.append("=" * 60)

    logger.info(f"Summary report written to {report_lines}")
    return "\n".join(report_lines)

def create_log_content(log_messages: List[str]) -> str:
    """Create log file content"""
    current_date = datetime.now().strftime("%Y%m%d")
    log_content = f"Processing Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_content += "=" * 60 + "\n\n"
    log_content += "\n".join(log_messages)
    logger.info(f"Log of this run is written to {log_content}")
    return log_content

def create_zip_file(files_dict: Dict[str, bytes]) -> bytes:
    """Create a zip file containing all the output files"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, content in files_dict.items():
            zip_file.writestr(filename, content)
    logger.info(f"All files are zipped into this file: {zip_buffer}")
    return zip_buffer.getvalue()

# Main Streamlit App
def main():
    st.title("📅 iCalendar Appointment Tool")
    st.markdown("Extract and process appointments with advanced patient matching")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Year selection
        current_year = datetime.now().year
        year = st.number_input(
            "Select Year",
            min_value=2020,
            max_value=2030,
            value=2025,
            help="Select the year to process appointments for"
        )
        
        # Month selection
        month_options = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        
        selected_month_names = st.multiselect(
            "Select Months",
            options=list(month_options.keys()),
            default=["September", "October"],
            help="Select one or more months to process"
        )
        
        selected_months = [month_options[month] for month in selected_month_names]
        
# Day selection
        st.markdown("### Select Days of Week")
        
        # Quick select buttons
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("All Days", use_container_width=True):
                st.session_state["day_selector"] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        with col2:
            if st.button("Weekdays", use_container_width=True):
                st.session_state["day_selector"] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        with col3:
            if st.button("Weekends", use_container_width=True):
                st.session_state["day_selector"] = ["Saturday", "Sunday"]
        with col4:
            if st.button("M/F", use_container_width=True):
                st.session_state["day_selector"] = ["Monday", "Friday"]
        with col5:
            if st.button("M/W/F", use_container_width=True):
                st.session_state["day_selector"] = ["Monday", "Wednesday", "Friday"]
        with col6:
            if st.button("T/Th", use_container_width=True):
                st.session_state["day_selector"] = ["Tuesday", "Thursday"]
        
        
        day_options = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }
        
        selected_day_names = st.multiselect(
            "Select Days",
            options=list(day_options.keys()),
            default=["Monday", "Friday"],
            key="day_selector",
            help="Select which days of the week to extract appointments from"
        )
        
        selected_days = [day_options[day] for day in selected_day_names]
        
        st.markdown("---")
        st.info("📌 Select any days of the week to extract appointments. The app uses advanced fuzzy matching with Armenian/Slavic name harmonization to match appointments with patient records.")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Upload Files")
        
        # iCalendar file upload
        ical_file = st.file_uploader(
            "Upload iCalendar File (.ics)",
            type=['ics'],
            help="Upload your calendar file in iCalendar format"
        )
        
        # Patient list upload
        patient_file = st.file_uploader(
            "Upload Patient List (list_of_patients_mutual.xlsx)",
            type=['xlsx', 'xls'],
            help="Upload the Excel file without a header with columns: [LastName, FirstName], [Diagnosis Codes], [Insurance], [Secondary Insurance], [Doctor], [Notes]"
        )
    
    with col2:
        st.header("📊 Selected Parameters")
        st.write(f"**Year:** {year}")
        st.write(f"**Months:** {', '.join(selected_month_names)}")
        st.write(f"**Days:** {', '.join(selected_day_names)}")
        st.write(f"**Matching Threshold:** 75% minimum confidence")
    
    # Process button
    if st.button("🚀 Process Appointments", type="primary", use_container_width=True):
        if not ical_file:
            st.error("Please upload an iCalendar file")
        elif not patient_file:
            st.error("Please upload the patient list Excel file")
        elif not selected_months:
            st.error("Please select at least one month")
        elif not selected_days:
            st.error("Please select at least one day of the week")
        else:
            with st.spinner("Processing appointments..."):
                log_messages = []
                try:
                    # Step 1: Parse iCalendar
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting iCalendar processing...")
                    calendar = parse_ical_file(ical_file.read())
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully parsed iCalendar file")
                    
                    # Step 2: Extract appointments
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting {', '.join(selected_day_names)} appointments for {', '.join(selected_month_names)} {year}")
                    appointments = extract_appointments(calendar, selected_months, year, selected_days)
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Extracted {len(appointments)} appointments")
                    
                    if not appointments:
                        st.warning("No appointments found for the specified criteria")
                        return
                    
                    # Step 3: Load patient list
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loading patient list...")
                    patients = load_patient_list(patient_file.read())
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len([p for p in patients.values() if isinstance(p, PatientData)])} patient records")
                    
                    # Step 4: Process and match
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Processing appointments and matching patients...")
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Using advanced fuzzy matching with surname harmonization...")
                    matched, unmatched, stats = process_appointments_with_patients(appointments, patients)
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Processing complete: {stats['matched_patients']} matched, {stats['unmatched_entries']} unmatched")
                    
                    # Step 5: Generate output files
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Generating output files...")
                    
                    # Create file contents
                    csv_content = create_csv_content(matched)
                    excel_content = create_excel_content(unmatched)
                    summary_content = generate_summary_report(stats, matched)
                    log_content = create_log_content(log_messages)
                    
                    # Store files in session state
                    current_date = datetime.now().strftime("%Y%m%d")
                    st.session_state.files_to_download = {
                        f"appointments_processed_{current_date}.csv": csv_content.encode('utf-8'),
                        f"appointments_not_patients_{current_date}.xlsx": excel_content,
                        f"processing_summary_{current_date}.txt": summary_content.encode('utf-8'),
                        f"run_log_{current_date}.txt": log_content.encode('utf-8')
                    }
                    
                    st.session_state.processed = True
                    st.session_state.stats = stats
                    st.session_state.summary = summary_content
                    
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] All files generated successfully!")
                    
                    st.success("✅ Processing completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(e)}")
    
    # Display results if processed
    if st.session_state.processed:
        st.markdown("---")
        st.header("📈 Processing Results")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Appointments", st.session_state.stats['total_appointments'])
        with col2:
            st.metric("Matched Patients", st.session_state.stats['matched_patients'])
        with col3:
            st.metric("Unmatched Entries", st.session_state.stats['unmatched_entries'])
        with col4:
            match_rate = (st.session_state.stats['matched_patients'] / 
                         st.session_state.stats['total_names_extracted'] * 100 
                         if st.session_state.stats['total_names_extracted'] > 0 else 0)
            st.metric("Match Rate", f"{match_rate:.1f}%")
        
        # Additional statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fuzzy Matches", st.session_state.stats['fuzzy_matches'])
        with col2:
            st.metric("Split Entries", st.session_state.stats['split_entries'])
        with col3:
            st.metric("Low Confidence Rejected", st.session_state.stats.get('rejected_low_confidence', 0))
        
        # Display summary report
        with st.expander("📋 View Summary Report"):
            st.text(st.session_state.summary)
        
        # Download section
        st.header("⬇️ Download Results")
        
        # Create zip file
        zip_content = create_zip_file(st.session_state.files_to_download)
        
        current_date = datetime.now().strftime("%Y%m%d")
        st.download_button(
            label="📦 Download All Files (ZIP)",
            data=zip_content,
            file_name=f"appointment_processing_{current_date}.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        # Individual file downloads
        with st.expander("Download Individual Files"):
            col1, col2 = st.columns(2)
            
            with col1:
                for filename, content in list(st.session_state.files_to_download.items())[:2]:
                    if content:
                        mime = "text/csv" if filename.endswith('.csv') else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        st.download_button(
                            label=f"📄 {filename}",
                            data=content,
                            file_name=filename,
                            mime=mime
                        )
            
            with col2:
                for filename, content in list(st.session_state.files_to_download.items())[2:]:
                    if content:
                        st.download_button(
                            label=f"📄 {filename}",
                            data=content,
                            file_name=filename,
                            mime="text/plain"
                        )

if __name__ == "__main__":
    main()
