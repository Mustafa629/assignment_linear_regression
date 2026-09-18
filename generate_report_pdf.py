"""Converts report.md into report.pdf (tables + embedded figures)."""
import os
import markdown
from xhtml2pdf import pisa

ROOT = os.path.dirname(__file__)
MD_PATH = os.path.join(ROOT, "report.md")
PDF_PATH = os.path.join(ROOT, "report.pdf")

CSS = """
<style>
  body { font-family: Helvetica, sans-serif; font-size: 10.5pt; line-height: 1.4; }
  h1 { font-size: 20pt; margin-top: 0; }
  h2 { font-size: 15pt; margin-top: 18px; border-bottom: 1px solid #888; padding-bottom: 3px; }
  h3 { font-size: 12.5pt; margin-top: 14px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 12px 0; }
  th, td { border: 1px solid #999; padding: 4px 6px; font-size: 9.5pt; text-align: left; }
  th { background-color: #eee; }
  img { max-width: 480px; margin: 8px 0; }
  code, pre { font-family: Courier, monospace; font-size: 9pt; background-color: #f4f4f4; }
  pre { padding: 6px; white-space: pre-wrap; }
  hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
</style>
"""

with open(MD_PATH, "r", encoding="utf-8") as f:
    md_text = f.read()

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
html = f"<html><head>{CSS}</head><body>{html_body}</body></html>"

with open(PDF_PATH, "wb") as f:
    result = pisa.CreatePDF(html, dest=f, path=ROOT)

print("PDF generation error code:", result.err)
print("Saved to:", PDF_PATH)
