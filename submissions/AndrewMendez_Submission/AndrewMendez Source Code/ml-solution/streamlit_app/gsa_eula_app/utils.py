# for pdf and docx extraction
import pandas as pd
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
import docx
from tqdm import tqdm
# for stripping and preprocessing text
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import io
stop_words = set(stopwords.words('english'))
no_nonsense_re = re.compile(r'^[a-zA-Z^508]+$')
def strip_nonsense(doc,remove_stop_words=False,port_stem=False):
    """
    Returns stemmed lowercased alpha-only substrings from a string
    
    Parameters:
        doc (str): the text of a single FBO document.
        
    Returns:
        words (str): a string of space-delimited lower-case alpha-only words (except for `508`)
    """
    
    doc = doc.lower()
    doc = doc.split()
    words = ''
    for word in doc:
        m = re.match(no_nonsense_re, word)
        if m:
            match = m.group()
            if remove_stop_words and match in stop_words:
                continue
            else:
                if port_stem == True:
                    match_len = len(match)
                    if match_len <= 17 and match_len >= 3:
                        porter = PorterStemmer()
                        stemmed = porter.stem(match)
                        words += stemmed + ' '
                else:
                    words+= match+ ' '
    return words
def extract_clauses_from_pdf(path_to_pdf):
    '''
    Extracts clauses from pdf
    First segments PDF into pages
    Then extracts clauses from all paragraphs in page
    '''
    fp = open(path_to_pdf, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    # print(type(retstr))
    codec = 'utf-8'
    laparams = LAParams(line_margin=0.1)
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    page_no = 0
    pages = []
    for pageNumber, page in enumerate(PDFPage.get_pages(fp)):
        print('Processing page {} from {}'.format(page_no,path_to_pdf))
        # if pageNumber == page_no:
        interpreter.process_page(page)

        data = retstr.getvalue()
        pages.append(data)

        # with open(os.path.join('Files/Company_list/0010/text_parsed/2017AR', f'pdf page {page_no}.txt'), 'wb') as file:
        #     file.write(data.encode('utf-8'))
        data = ''
        retstr.truncate(0)
        retstr.seek(0)

        page_no += 1
    # split pages into claues
    clauses_per_page = []
    for p in pages:
        clauses_unormalized = [i.replace("\n"," ") for i in p.split("\n\n")]
        clauses_per_page.append(clauses_unormalized)
    return clauses_per_page

# the underlying XML does not make it easy to identify page breaks
def get_text_from_docx(filename):
    '''
    Function that uses python-docx to extract clauses (sentences) from docx.
    Loops through document, finds paragraphs
    '''
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        clause = para.text
        clause = strip_nonsense(clause)
        if len(clause)>2:
            fullText.append(clause)
    return fullText
def preprocess_clauses_pdf(pages):
    '''
    preprocess clauses
    '''
    clauses = []
    for p in pages:
        for clause in p:
            # do not include if length < 2
            clause_normalized = strip_nonsense(clause)
            if len(clause_normalized) >2:# append if has at least one word
                clauses.append(clause_normalized)
            # stip nonsense
    return clauses