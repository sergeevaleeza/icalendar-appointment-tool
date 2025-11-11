# iCalendar Appointment Tool

This is a Streamlit app that extracts appointments from an iCalendar file, matches them with a patient list from an Excel file, and generates output files including matched appointments, unmatched entries, and a summary report. The app allows configuration of the year, months, and days to extract appointments for.

## Features
- Upload iCalendar (.ics) file and patient list (.xlsx).
- Select year, months, and days via sidebar.
- Process data and generate output files.
- Download results as a ZIP file, including a date-based log file.

## Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. Create a new GitHub repository
2. Upload the following files:
   - `app.py` (the main Streamlit application)
   - `requirements.txt` (dependency list)
   - This `README.md`

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and main file (`app.py`)
5. Click "Deploy"

### Step 3: Configuration (Optional)

You can set up secrets in Streamlit Cloud if needed:
1. Go to your app's settings
2. Navigate to "Secrets"
3. Add any required environment variables

## Local Development

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Select Year, Months and Days**: Use the sidebar to configure which year, months and days to process

2. **Upload Files**:
   - **iCalendar File**: Upload your `.ics` calendar file
   - **Patient List**: Upload the `list_of_patients_mutual.xlsx` file with columns:
      - Column A: Last Name, First Name 
      - Column B: ICD-10-CM Codes
      - Column C: Insurance
      - Column E: Doctor

3. **Process**: Click the "Process Appointments" button

4. **Download Results**: After processing, download all files as a zip or individually:
   - `appointments_processed_YYYYMMDD.csv`: Matched appointments with patient data
   - `appointments_not_patients_YYYYMMDD.xlsx`: Unmatched entries for review
   - `processing_summary_YYYYMMDD.txt`: Summary statistics and analysis
   - `run_log_YYYYMMDD.txt`: Processing log for debugging

## File Formats

### Input: Patient List Excel
The Excel file should have no header row and contain:
- Column A: Last Name, First Name 
- Column B: ICD-10-CM Codes
- Column C: Insurance
- Column E: Doctor

### Output: Matched Appointments CSV
Contains:
- Patient name (Last, First format)
- Appointment date, time, day of week
- ICD-10-CM Codes, insurance, doctor information
- Procedure codes (CPT/TMS)
- Match confidence percentage

### Output: Unmatched Entries Excel
Contains appointments that couldn't be matched to patient records for manual review.

## Features in Detail

### Intelligent Name Matching
- Fuzzy matching with configurable threshold (75% default)
- Handles name variations and misspellings
- Armenian and Slavic name harmonization
- Supports multiple patients per appointment slot

### Code Extraction
- Automatically extracts CPT codes, TMS numbers, and F-codes from appointment titles
- Preserves metadata for billing and reporting

### Comprehensive Reporting
- Statistics on total appointments, match rates, and confidence levels
- Breakdown by day of week and doctor
- Confidence distribution analysis

## Troubleshooting

### Common Issues

1. **No appointments found**: Check that:
   - Selected months and year are correct
   - Calendar contains events on Monday/Friday
   - iCalendar file is valid

2. **Low match rate**: May indicate:
   - Names in calendar don't match patient list format
   - Missing patients in the Excel file
   - Need to adjust fuzzy matching threshold

3. **File upload errors**: Ensure:
   - iCalendar file has `.ics` extension
   - Excel file is `.xlsx` or `.xls` format
   - Files aren't corrupted

## Support

For issues or questions, please create an issue in the GitHub repository.

## License
MIT License
Copyright (c) 2025 Leeza Sergeeva

This project is provided as-is for medical appointment processing purposes.
