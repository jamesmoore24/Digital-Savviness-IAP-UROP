import requests as r
from bs4 import BeautifulSoup
import pandas as pd
import time
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
import os


def get_fin_data():
    """
    This function takes information from the company_npm_roa.csv and company_rev.csv and converts the data into a csv file named 'company_data_final.csv'
    Data is taken from 2021-2022 in an annual measurement from Compustat North America Fundamental and Financial Ratio data bases

    This 'company_data_final.csv' contains:
        - GVKEY: company GVKEY number
        - CIK: company CIK number
        - NAICS: company North American Industry Classification System (NAICS) code
        - ($M) MCAP: estimated company market capitalization based on closed share price and total shares outstanding
        - ($M) REVT: company revenue total
        - ROA: company return on assets
        - NPM: company net profit margin
    """
    #obtain naics relations from 'company_info.csv'
    naics_d = dict()
    naics_set = set()
    for _, row in pd.read_csv('csv_files/company_info.csv').iterrows():
        comp = row['gvkey']
        if comp not in naics_set:
            naics_d[comp] = row['naics']
            naics_set.add(comp)

    #obtain ratio data
    ratio_df = pd.read_csv('csv_files/company_npm_roa.csv')
    ratio_df = ratio_df.dropna()
    ratio_set = set()
    ratio_dict = {} # this relates gvkey to npm and roa values in list
    for _, row in ratio_df.iterrows():
        comp = row['gvkey']
        if comp not in ratio_set:
            ratio_dict[comp] = (row['roa'], row['npm'])
            ratio_set.add(comp)

    #set up dictionary to be used to create pandas data frame
    d = {
        'GVKEY': [],
        'CIK': [],
        'NAICS': [],
        'MCAP': [],
        'REVT': [],
        'ROA': [],
        'NPM': [],
    }

    #obtain fundamentals data and create dictionary
    fund_df = pd.read_csv('csv_files/company_rev.csv')
    fund_df = fund_df.dropna()
    fund_set = set()
    for _, row in fund_df.iterrows():
        comp = row['gvkey']
        if comp not in fund_set and comp in ratio_set:
            d['GVKEY'].append(comp)
            d['CIK'].append(row['cik'])
            d['NAICS'].append(naics_d[comp])
            d['REVT'].append(row['revt'])
            d['MCAP'].append(row['csho'] * row['prcc_f'])
            d['ROA'].append(ratio_dict[comp][0])
            d['NPM'].append(ratio_dict[comp][1])
            fund_set.add(comp)
    
    #create dictionary, drop rows with empty cells and save to csv file
    df = pd.DataFrame(d)
    df = df.dropna()
    df.to_csv('company_data_final.csv')

def get_directors(data_directory):
    """
    Parameters:
        - data_directory: the directory to data repository
    Returns:
        - None
    Purpose:
        - Upload company financial performance data (GVKEY, CIK, NAICS, MCAP, REVT, ROA, NPM) alongside the directors' string
        - Will check directors against COMPUSTAT data for directors later
    """
    #initialize database connection
    excluded = 0
    director_count = 0
    
    #load variables from private .env file (can't be seen publicly on github)
    load_dotenv() 
    #create connection to PostgreSQL DB
    engine = create_engine(os.environ.get("DB_URI"))
    df_final = pd.read_csv(data_directory)

    #go through each row
    for ix, row in df_final.iterrows():
        #Note the progress querying the api
        print("Progress:", 100*round(float(ix/len(df_final.index)), 3), "%")
        print(f"Searching for {row['GVKEY']}")

        #initialize constants
        directors = ''
        d_set = set()
        no_data = 0
        invalid_html = 0
        no_recent_filings = 0
        def14a_count = 0
        count = 0

        # Get the current date and time 3 years ago to this day
        now = datetime.now()
        three_years_ago = now - timedelta(days=365*3)

        #Initialize API call
        headers = {
            "User-Agent": "jmoore1@mit.edu"
        }
        zeroes = "0" * (10-len(str(int(row['CIK']))))
        url = f"https://data.sec.gov/submissions/CIK{zeroes}{int(row['CIK'])}.json"

        try:
            #get the response and index to recent filings in json
            response = r.get(url, headers=headers)
            info = response.json()['filings']['recent']
            
            for access_num, file, form, date in zip(info['accessionNumber'], info['primaryDocument'], info['form'], info['filingDate']):
                #only allowed 10 requests/sec
                time.sleep(0.1)

                #need to stop search because we don't want to search forms longer than 3 years old
                if datetime.strptime(date, "%Y-%m-%d") < three_years_ago:
                    if not def14a_count:
                        print(f"No recent filings for {int(row['cik'])}")
                        no_recent_filings += 1
                    break
                #don't check too many form 4's
                elif count > 30:
                    break
                #analyze the form
                elif form == "4":
                    #initialize unique file url
                    f_url = f"https://www.sec.gov/Archives/edgar/data/{int(row['CIK'])}/{access_num.replace('-', '')}/{file}"
                    try:
                        f_response = r.get(f_url, headers=headers)
                        #Code to scrape directors names from FORM 4
                        soup = BeautifulSoup(f_response.text, 'html.parser')
                        name = soup.find_all("table")[4].find_all("a")[0].text
                        #find whether or not there's the 'X' next to the 'Director' option in the form 4
                        if soup.find_all("table")[4].find_all("table")[5].find_all("td")[0].text == 'X' and name not in d_set:
                            print(name, "Director added")
                            directors += name.lower() + ';'
                            d_set.add(name)
                        count += 1
                    except Exception as e:
                        #Can't read the html file
                        invalid_html = 1
                        print("Failed to load url! Exception: ", repr(e))
        except Exception as e:
            #couldn't access the link
            no_data = 1
            print(f"Couldn't retrive CIK data for {int(row['CIK'])}", repr(e))
        
        #create new DF, get rid of index column and export to db
        df = row.to_frame().T
        df.columns = df_final.columns
        #Add extra columns to DF
        df['Directors'] = [directors]
        df['No Data'] = [no_data]
        df['Invalid HTML'] = [invalid_html]
        df['No Recent Filings'] = [no_recent_filings]

        #export to PostgreSQL DB
        df.iloc[:, 1:].to_sql('company_data', engine, if_exists='append')
    
    print("Excluded", excluded, "companies")
    print("Found directors list for", director_count, "companies")

def get_biographies(cik, directors):
    """
    Parameters:
        - CIK: the company's unique CIK code
        - directors: a string of directors names separated by ';' character
    Returns:
        - director_bios: a dictionary that relates a director's name (FIRST LAST) to its respective biography in the DEF 14A filing

    Purpose:
        - For use in the calculation of tech savviness
    """

    # Retrieve the DEF 14A file from the EDGAR API
    #Initialize API call
    headers = {
        "User-Agent": "jmoore1@mit.edu"
    }
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = r.get(url, headers=headers)
    info = response.json()['filings']['recent']

    #clues and their respective weights
    clues = {'director sinc': 20, 'director of the company sinc': 20, 'elected a director': 20, 'appointed a director': 20, 'university': 10, 'college': 10, 'studi at': 10, 'director': 4, 'served a': 8, 'brings to the board': 10, 'founder of the compani': 10, 'founder': 5, 'qualif': 10, 'chairman of the board': 5, 'select': 1, 'nomin': 3, 'other board': 5, 'experi': 2, 'understand': 1, 'knowledg': 1, 'termin': -10, 'proxi': -10}
    
    #get directors from string (Ex. smith john h.;doe jane r.; ...)
    #Structure looks like [[last, first], [last, first], ...]
    directors_l = []
    for director in directors.split(';')[:-1]:
        #append first and last name strings into list
        directors_l.append(director.split(' ')[:2])

    for access_num, file, form, date in zip(info['accessionNumber'], info['primaryDocument'], info['form'], info['filingDate']):
        if form == 'DEF 14A':
            #create url and get a response
            f_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{access_num.replace('-', '')}/{file}"
            response = r.get(f_url, headers=headers)

            #make into html object and parse text using tokenizer
            soup = BeautifulSoup(response.text, 'html.parser')
            #make into lowercase and tokenize text
            parts = sent_tokenize(soup.text.lower())
            
            #initialize stemmer
            stemmer = PorterStemmer()
            
            #dictionary used to relate director to their respective highest-ranked biography
            director_bios = {}
            for director in directors_l:
                #create list of sentences found in data structure: (sentence, rank)
                sentence_record = []
                #check every part in the file
                for part in parts:
                    #need to check for every director, flag used because director name split into first and last
                    flag_director = False
                    for name in director:
                        if name in part:
                            flag_director = True
                    #only want to check parts which have directors names (first or last) in it
                    if flag_director:
                        rank = 0
                        stemmed = ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(part)]) 
                        for clue in clues:
                            if clue in stemmed: 
                                #add the weight to the total
                                rank += clues[clue]
                        #append part with ranking and get rid of new line and non-breaking space characters
                        sentence_record.append((part.replace("\n", "").replace("\xa0", ""), rank))
                
                #sort list by second value in tuple
                sentence_record.sort(key = lambda x: x[1], reverse=True)
                #add director name (FIRST LAST) with sentence into dictionary
                director_bios[f"{director[1]} {director[0]}"] = sentence_record[0][0]

            #return dictionary of names and related biographies for a single company
            return director_bios

if __name__ == '__main__':
    None


    



    

    
    
    
        

    




