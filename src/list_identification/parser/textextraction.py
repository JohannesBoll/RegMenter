import os
import argparse
import fitz
from pathlib import Path

def extract_text(pathlist):
	"""
	Transform a given list of paths to PDF document to the text representation of them
	:param pathlist: List of paths
	:return: List of string list, each list represents a document
	"""
	textlist = []
	for path in pathlist:
		doc = fitz.open(path)
		text = ""
		for page in doc:
			text += page.get_text()
		textlist.append(text)
	return textlist

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='This tool extract text from pdfs. It is based on PyMuPDF.')
	parser.add_argument('path', type=Path, help='The path to the input file')
	args = parser.parse_args()
	path = args.path
	if os.path.exists(path):
		extract_text(path)
	else:
		raise FileNotFoundError(path)
