import os
import spacy
import io
import pandas as pd
import nltk
from spacy.symbols import nsubj,NOUN
nlp = spacy.load("en_core_web_sm")
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import io
import re
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
#import nltk
#nlp = spacy.load("en_core_web_sm")
from io import StringIO
import string
import pandas as pd
#from nameparser.parser import HumanName
#from nltk.corpus import wordnet
import pandas as pd
import sqlite3 as sql
import models as m

def extract_text_from_pdf(x):
    with open(x ,'rb') as fh:
        # iterate over all pages of PDF document
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resoure manager
            resource_manager = PDFResourceManager()

            # create a file handle
            fake_file_handle = io.StringIO()

            # creating a text converter object
            converter = TextConverter(
                resource_manager,
                fake_file_handle,
                codec='utf-8',
                laparams=LAParams()
            )

            # creating a page interpreter
            page_interpreter = PDFPageInterpreter(
                resource_manager,
                converter
            )

            # process current page
            page_interpreter.process_page(page)

            # extract text
            text = fake_file_handle.getvalue()
            # print(text)
            yield text

            # close open handles
            converter.close()
            fake_file_handle.close()











def main():
    con = sql.connect("my.db")
    con.execute("DROP TABLE IF EXISTS first");
    con.execute('create table if not exists first ("id" integer primary key autoincrement,"username" text not null, "email" text not null, "twitter" text not null, "score" integer not null)')
    con.close()
    r=[]
    df = pd.read_fwf('REQ FOLDER/rq.txt')
    df.to_csv('log.csv')
    #m.dele()
    with os.scandir('New folder/') as entries:
        for entry in entries:
            #print(entry.name)
            text=''
            for page in extract_text_from_pdf(entry):
                #print(page)
                text += ' ' + page



            # for foo in lst1:
            #     foo1=foo.strip('\n')
            #     #print("TWITTER---------------",foo.strip('\n'))
            lst = re.findall('\S+@\S+', text)
            #print(lst)
            doc = nlp(text)
            data = pd.read_csv("log.csv")
            tokens = [token.text for token in doc if not token.is_stop]

            skills = list(data.columns.values)
#print(skills)
            skillset = []
            score=0
            for token in tokens:
                if token.lower() in skills:
                    skills.remove(token.lower())
                    score+=100
                    skillset.append(token)

#print("sum------",s)
            print("MATCHED SKILLS---",skillset)
            print("CV_SCORE---",score)
            for y in lst:
                print(y)
                con = sql.connect("my.db")
                cur = con.cursor()
                cur.execute("SELECT distinct username,twitter FROM users where email = ?",(y,))
                users = cur.fetchall()
                con.close()
            for x in users:
                pname=str(x[0])
                te_id=str(x[1])
                r+=[(pname,score,y,te_id)]
                #print(r[0])
            print("---run---------")
            #m.insertFirst(pname,score)
           #print(ans)
    print("PRINTING LIST")
    print(r)
    #for q in r:
    #    print(q[0])
        #m.insertFirst(q,)
    con = sql.connect("my.db")
    cur = con.cursor()
    cur.execute("select username,score from first")
    ans = cur.fetchall()
    con.close()
    #users = m.retrieveFirst()
    #print(ans)
    for x in ans:
        print(x)
    #return ans



    def Sort_Tuple(tup):
     return (sorted(tup, key=lambda x: x[1], reverse=True))

    ans2 = Sort_Tuple(r)
    print(ans2)
    for item in ans2:
        m.insertFirst(item[0], item[2], item[3], item[1])


    #retrieving from the table
    con = sql.connect("my.db")
    cur = con.cursor()
    cur.execute("select * from first")
    ans = cur.fetchall()
    con.close()
    return ans
    # users = m.retrieveFirst()
    # print(ans)
    #for x in ans:
     #   print(x)

    #return render_template('addgo.html', users=users)
if __name__ == "__main__":
    main()