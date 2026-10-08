# ATS Resume Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and tells you exactly how to improve it. It uses Google's Gemini Flash model.

## Features

- Upload a resume as **PDF, DOCX or TXT** (max 5 MB)
- Optional **job description** box for keyword matching
- **ATS score out of 100** with a breakdown:
  - Keywords & relevance (25)
  - Content & impact (25)
  - ATS-friendly formatting (20)
  - Structure & sections (15)
  - Readability & length (15)
- Strengths, weaknesses, matched and missing keywords
- Prioritised improvements (High / Medium / Low)
- Suggested bullet-point rewrites
- Instant rule-based checks (email, phone, sections, word count, metrics)
- Download the full report as Markdown

The total score is calculated in code from the category scores, so the maths is always consistent.

## Run locally

```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

- paste it into the sidebar when the app opens, **or**
- set it as an environment variable:
  - macOS/Linux: `export GEMINI_API_KEY="your-key"`
  - Windows (PowerShell): `$env:GEMINI_API_KEY="your-key"`

Start the app:

```bash
streamlit run app.py
```

## Configuration

| Setting | How | Default |
|---|---|---|
| API key | Sidebar, `GEMINI_API_KEY` env var, or Streamlit secret | none |
| Model | Sidebar box or `GEMINI_MODEL` env var | `gemini-3.5-flash` |

If Google renames or retires a model, type the new model name in the sidebar (for example `gemini-3-flash-preview` or `gemini-2.5-flash`). See https://ai.google.dev/gemini-api/docs/models for current names.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository (see below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**. You get a public `https://<name>.streamlit.app` link.

## Push to GitHub

```bash
git init
git add app.py requirements.txt README.md
git commit -m "Initial commit: ATS resume checker"
git branch -M main
git remote add origin https://github.com/<your-username>/ats-resume-checker.git
git push -u origin main
```

Create the empty repository first at https://github.com/new (do not tick "Add a README").

**Never commit your API key.** Add a `.gitignore` containing:

```
venv/
__pycache__/
.env
.streamlit/secrets.toml
```

## Notes and limitations

- Scanned (image-only) PDFs cannot be read. ATS systems cannot read them either, so export a text-based PDF or DOCX.
- The score is an AI-based estimate, not the output of any real employer's ATS. Use it as guidance.
- Resume text is sent to the Gemini API for analysis. The app does not store it. Do not upload resumes you are not allowed to share.
- The free Gemini tier has rate limits. If you see a quota error, wait a minute and retry.

## Project structure

```
app.py             Streamlit app (UI, file parsing, Gemini call, scoring)
requirements.txt   Python dependencies
README.md          This file
```
